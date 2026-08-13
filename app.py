
from pathlib import Path
import math
import os
import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import streamlit as st
import pydeck as pdk

try:
    from google import genai
except ImportError:
    genai = None

from shapely.geometry import LineString, MultiLineString
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ============================================================
# Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="LOGI-COPILOT",
    page_icon="🚚",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ROAD_SHP_PATH = DATA_DIR / "seoul.shp"
DEPOT_PATH = DATA_DIR / "depot.csv"
DRIVERS_PATH = DATA_DIR / "drivers.csv"
DELIVERIES_PATH = DATA_DIR / "deliveries.csv"
KNOWLEDGE_PATH = DATA_DIR / "knowledge.csv"
ROUTES_BEFORE_PATH = DATA_DIR / "routes_before_v2.csv"

DEFAULT_SPEED_KMH = 30.0
DEFAULT_SERVICE_MIN = 5
DEFAULT_MAX_ROUTE_TIME_MIN = 65.0
MARGIN_M = 5000


# ============================================================
# 1. 유틸
# ============================================================

def check_required_files():
    required = [
        ROAD_SHP_PATH,
        DEPOT_PATH,
        DRIVERS_PATH,
        DELIVERIES_PATH,
        KNOWLEDGE_PATH,
        ROUTES_BEFORE_PATH
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        st.error("필수 파일을 찾을 수 없습니다.")
        st.code("\n".join(missing))
        st.stop()


def extract_delivery_names(route_string):
    return re.findall(r"배송지_\d+", str(route_string))


def round_node(x, y):
    return (round(float(x), 2), round(float(y), 2))


def add_line_to_graph(G, line, speed_kmh):
    coords = list(line.coords)

    if len(coords) < 2:
        return

    speed_mps = speed_kmh * 1000 / 3600

    for i in range(len(coords) - 1):
        u = round_node(coords[i][0], coords[i][1])
        v = round_node(coords[i + 1][0], coords[i + 1][1])

        if u == v:
            continue

        length_m = math.hypot(v[0] - u[0], v[1] - u[1])

        if length_m <= 0:
            continue

        travel_time_sec = length_m / speed_mps

        # MVP에서는 양방향으로 처리
        G.add_edge(u, v, length_m=length_m, travel_time_sec=travel_time_sec)
        G.add_edge(v, u, length_m=length_m, travel_time_sec=travel_time_sec)


# ============================================================
# 2. 기본 CSV
# ============================================================

@st.cache_data
def load_base_data():
    depot_df = pd.read_csv(DEPOT_PATH)
    drivers_df = pd.read_csv(DRIVERS_PATH)
    deliveries_df = pd.read_csv(DELIVERIES_PATH)
    knowledge_df = pd.read_csv(KNOWLEDGE_PATH)
    routes_before_df = pd.read_csv(ROUTES_BEFORE_PATH)

    return depot_df, drivers_df, deliveries_df, knowledge_df, routes_before_df


# ============================================================
# 3. 도로 그래프
# ============================================================

@st.cache_resource(show_spinner="도로 네트워크를 준비하는 중입니다...")
def build_network():
    depot_df = pd.read_csv(DEPOT_PATH)
    drivers_df = pd.read_csv(DRIVERS_PATH)
    deliveries_df = pd.read_csv(DELIVERIES_PATH)

    roads = gpd.read_file(ROAD_SHP_PATH)

    if roads.empty:
        raise ValueError("도로 SHP가 비어 있습니다.")

    if roads.crs is None:
        raise ValueError("도로 SHP의 CRS 정보가 없습니다.")

    # CSV의 node_x / node_y가 EPSG:5179 기준이므로
    # SHP가 이미 투영좌표계여도 반드시 EPSG:5179로 통일
    roads_m = roads.to_crs(epsg=5179)

    roads_m = roads_m[roads_m.geometry.notna() & ~roads_m.geometry.is_empty].copy()

    all_x = pd.concat([
        depot_df["node_x"],
        drivers_df["node_x"],
        deliveries_df["node_x"]
    ])

    all_y = pd.concat([
        depot_df["node_y"],
        drivers_df["node_y"],
        deliveries_df["node_y"]
    ])

    xmin = all_x.min() - MARGIN_M
    xmax = all_x.max() + MARGIN_M
    ymin = all_y.min() - MARGIN_M
    ymax = all_y.max() + MARGIN_M

    roads_m = roads_m.cx[xmin:xmax, ymin:ymax].copy()

    if roads_m.empty:
        raise ValueError("분석지역에 해당하는 도로가 없습니다.")

    speed_col = None

    for col in ["MAX_SPD", "MAX_SPEED", "SPEED", "SPD"]:
        if col in roads_m.columns:
            speed_col = col
            break

    G = nx.DiGraph()

    for _, row in roads_m.iterrows():
        speed_kmh = DEFAULT_SPEED_KMH

        if speed_col is not None:
            try:
                value = float(row[speed_col])

                if np.isfinite(value) and value > 0:
                    speed_kmh = value
            except:
                pass

        geom = row.geometry

        if isinstance(geom, LineString):
            add_line_to_graph(G, geom, speed_kmh)

        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                add_line_to_graph(G, line, speed_kmh)

    if G.number_of_nodes() == 0:
        raise ValueError("도로 그래프 생성에 실패했습니다.")

    components = list(nx.weakly_connected_components(G))
    largest_component = max(components, key=len)
    G = G.subgraph(largest_component).copy()

    graph_nodes = np.array(list(G.nodes), dtype=float)

    def nearest_graph_node(x, y):
        d2 = (graph_nodes[:, 0] - x) ** 2 + (graph_nodes[:, 1] - y) ** 2
        idx = np.argmin(d2)
        return tuple(graph_nodes[idx])

    depot_node = nearest_graph_node(
        depot_df.iloc[0]["node_x"],
        depot_df.iloc[0]["node_y"]
    )

    driver_nodes = {}

    for _, row in drivers_df.iterrows():
        vehicle_id = int(row["vehicle_id"])

        driver_nodes[vehicle_id] = nearest_graph_node(
            row["node_x"],
            row["node_y"]
        )

    delivery_nodes = {}

    for _, row in deliveries_df.iterrows():
        delivery_nodes[row["name"]] = nearest_graph_node(
            row["node_x"],
            row["node_y"]
        )

    coord_transformer = Transformer.from_crs(
        roads_m.crs,
        "EPSG:4326",
        always_xy=True
    )

    return G, depot_node, driver_nodes, delivery_nodes, coord_transformer


# ============================================================
# 4. 재배차 엔진
# ============================================================

def run_reassignment(
    G,
    depot_node,
    driver_nodes,
    delivery_nodes,
    deliveries_df,
    knowledge_df,
    routes_before_df,
    failed_vehicle,
    max_route_time_min
):
    extra_delay = {name: 0 for name in deliveries_df["name"]}

    for _, row in knowledge_df.iterrows():
        location = row["location"]

        if location in extra_delay:
            extra_delay[location] += int(row["extra_delay_min"])

    service_time_min = {}

    for _, row in deliveries_df.iterrows():
        name = row["name"]
        service_time_min[name] = DEFAULT_SERVICE_MIN + extra_delay.get(name, 0)

    vehicle_routes = {}

    for _, row in routes_before_df.iterrows():
        vehicle_id = int(row["vehicle_id"])
        vehicle_routes[vehicle_id] = extract_delivery_names(row["route"])

    if failed_vehicle not in vehicle_routes:
        raise ValueError(f"차량 {failed_vehicle}의 기존 경로를 찾을 수 없습니다.")

    original_vehicle_routes = {
        vehicle_id: route.copy()
        for vehicle_id, route in vehicle_routes.items()
    }

    failed_deliveries = vehicle_routes[failed_vehicle].copy()
    vehicle_routes.pop(failed_vehicle)

    path_cache = {}

    def shortest_metrics(start_node, end_node):
        key = (start_node, end_node)

        if key in path_cache:
            return path_cache[key]

        distance_m = nx.shortest_path_length(
            G,
            start_node,
            end_node,
            weight="length_m"
        )

        travel_time_sec = nx.shortest_path_length(
            G,
            start_node,
            end_node,
            weight="travel_time_sec"
        )

        result = (float(distance_m), float(travel_time_sec))
        path_cache[key] = result

        return result

    def calculate_route(vehicle_id, delivery_route):
        nodes = [driver_nodes[vehicle_id]]

        for delivery_name in delivery_route:
            nodes.append(delivery_nodes[delivery_name])

        nodes.append(depot_node)

        total_distance_m = 0
        total_travel_time_sec = 0

        for i in range(len(nodes) - 1):
            distance_m, travel_time_sec = shortest_metrics(
                nodes[i],
                nodes[i + 1]
            )

            total_distance_m += distance_m
            total_travel_time_sec += travel_time_sec

        total_service_min = sum(
            service_time_min[name]
            for name in delivery_route
        )

        travel_time_min = total_travel_time_sec / 60
        estimated_time_min = travel_time_min + total_service_min

        return {
            "distance_km": total_distance_m / 1000,
            "travel_time_min": travel_time_min,
            "service_time_min": total_service_min,
            "estimated_time_min": estimated_time_min
        }

    def find_best_feasible_insertion(delivery_name):
        feasible_candidates = []
        infeasible_candidates = []

        for vehicle_id, current_route in vehicle_routes.items():
            current_metrics = calculate_route(vehicle_id, current_route)

            for insert_position in range(len(current_route) + 1):
                candidate_route = current_route.copy()
                candidate_route.insert(insert_position, delivery_name)

                candidate_metrics = calculate_route(
                    vehicle_id,
                    candidate_route
                )

                added_time = (
                    candidate_metrics["estimated_time_min"]
                    - current_metrics["estimated_time_min"]
                )

                added_distance = (
                    candidate_metrics["distance_km"]
                    - current_metrics["distance_km"]
                )

                final_time = candidate_metrics["estimated_time_min"]

                candidate = {
                    "vehicle_id": vehicle_id,
                    "insert_position": insert_position,
                    "delivery": delivery_name,
                    "added_time_min": added_time,
                    "added_distance_km": added_distance,
                    "final_time_min": final_time,
                    "overload_min": max(0, final_time - max_route_time_min),
                    "new_route": candidate_route
                }

                if final_time <= max_route_time_min:
                    feasible_candidates.append(candidate)
                else:
                    infeasible_candidates.append(candidate)

        if feasible_candidates:
            feasible_candidates.sort(
                key=lambda x: (
                    x["added_time_min"],
                    x["added_distance_km"],
                    x["final_time_min"]
                )
            )

            return feasible_candidates[0], None

        if infeasible_candidates:
            infeasible_candidates.sort(
                key=lambda x: (
                    x["overload_min"],
                    x["added_time_min"],
                    x["added_distance_km"]
                )
            )

            return None, infeasible_candidates[0]

        return None, None

    reassignment_log = []
    unassigned_log = []

    for delivery_name in failed_deliveries:
        best, best_infeasible = find_best_feasible_insertion(delivery_name)

        if best is not None:
            vehicle_id = best["vehicle_id"]
            vehicle_routes[vehicle_id] = best["new_route"]

            reassignment_log.append({
                "delivery": delivery_name,
                "decision": "자동 재배차",
                "assigned_vehicle": vehicle_id,
                "insert_position": best["insert_position"] + 1,
                "added_distance_km": best["added_distance_km"],
                "added_time_min": best["added_time_min"],
                "final_vehicle_time_min": best["final_time_min"],
                "max_route_time_min": max_route_time_min,
                "status": "재배차 가능"
            })

        elif best_infeasible is not None:
            unassigned_log.append({
                "delivery": delivery_name,
                "decision": "추가 차량 투입 필요",
                "best_candidate_vehicle": best_infeasible["vehicle_id"],
                "best_insert_position": best_infeasible["insert_position"] + 1,
                "best_candidate_added_distance_km": best_infeasible["added_distance_km"],
                "best_candidate_added_time_min": best_infeasible["added_time_min"],
                "best_candidate_final_time_min": best_infeasible["final_time_min"],
                "max_route_time_min": max_route_time_min,
                "overload_min": best_infeasible["overload_min"],
                "status": "자동 재배차 불가"
            })

        else:
            unassigned_log.append({
                "delivery": delivery_name,
                "decision": "추가 차량 투입 필요",
                "best_candidate_vehicle": np.nan,
                "best_insert_position": np.nan,
                "best_candidate_added_distance_km": np.nan,
                "best_candidate_added_time_min": np.nan,
                "best_candidate_final_time_min": np.nan,
                "max_route_time_min": max_route_time_min,
                "overload_min": np.nan,
                "status": "경로 후보 없음"
            })

    reassignment_df = pd.DataFrame(reassignment_log)
    unassigned_df = pd.DataFrame(unassigned_log)

    after_rows = []

    for vehicle_id in sorted(vehicle_routes):
        route = vehicle_routes[vehicle_id]
        metrics = calculate_route(vehicle_id, route)

        route_string = f"차량 {vehicle_id} 현재위치"

        if route:
            route_string += " → " + " → ".join(route)

        route_string += " → 물류센터"

        after_rows.append({
            "vehicle_id": vehicle_id,
            "route": route_string,
            "deliveries": len(route),
            "distance_km": metrics["distance_km"],
            "travel_time_min": metrics["travel_time_min"],
            "service_time_min": metrics["service_time_min"],
            "estimated_time_min": metrics["estimated_time_min"],
            "within_limit": metrics["estimated_time_min"] <= max_route_time_min
        })

    after_df = pd.DataFrame(after_rows)

    before_rows = []

    for vehicle_id, route in original_vehicle_routes.items():
        metrics = calculate_route(vehicle_id, route)

        route_string = f"차량 {vehicle_id} 현재위치"

        if route:
            route_string += " → " + " → ".join(route)

        route_string += " → 물류센터"

        before_rows.append({
            "vehicle_id": vehicle_id,
            "route": route_string,
            "deliveries": len(route),
            "distance_km": metrics["distance_km"],
            "travel_time_min": metrics["travel_time_min"],
            "service_time_min": metrics["service_time_min"],
            "estimated_time_min": metrics["estimated_time_min"]
        })

    before_df = pd.DataFrame(before_rows)

    total_original_deliveries = int(before_df["deliveries"].sum())
    completed_after = int(after_df["deliveries"].sum())
    unassigned_count = len(unassigned_df)

    comparison_df = pd.DataFrame({
        "항목": [
            "운행 차량 수",
            "완료 배송건수",
            "재배차 대기 배송건수",
            "총 배송거리(km)",
            "전체 차량 운행시간 합(min)",
            "평균 차량 운행시간(min)",
            "최장 차량 운행시간(min)"
        ],
        "정상 운영": [
            len(before_df),
            total_original_deliveries,
            0,
            before_df["distance_km"].sum(),
            before_df["estimated_time_min"].sum(),
            before_df["estimated_time_min"].mean(),
            before_df["estimated_time_min"].max()
        ],
        "차량 고장 후": [
            len(after_df),
            completed_after,
            unassigned_count,
            after_df["distance_km"].sum(),
            after_df["estimated_time_min"].sum(),
            after_df["estimated_time_min"].mean(),
            after_df["estimated_time_min"].max()
        ]
    })

    decision_rows = []

    for _, row in reassignment_df.iterrows():
        decision_rows.append({
            "delivery": row["delivery"],
            "system_decision": f"차량 {int(row['assigned_vehicle'])} 자동 재배차",
            "reason": (
                f"재배차 후 예상 운행시간 "
                f"{row['final_vehicle_time_min']:.2f}분으로 "
                f"{max_route_time_min:.0f}분 기준 이내"
            )
        })

    for _, row in unassigned_df.iterrows():
        if pd.notna(row["best_candidate_vehicle"]):
            reason = (
                f"가장 유리한 차량 {int(row['best_candidate_vehicle'])}도 "
                f"재배차 후 {row['best_candidate_final_time_min']:.2f}분으로 "
                f"기준보다 {row['overload_min']:.2f}분 초과"
            )
        else:
            reason = "재배차 가능한 경로 후보 없음"

        decision_rows.append({
            "delivery": row["delivery"],
            "system_decision": "추가 차량 투입 필요",
            "reason": reason
        })

    decision_summary_df = pd.DataFrame(decision_rows)

    knowledge_check_df = pd.DataFrame({
        "배송지": deliveries_df["name"],
        "기본서비스시간_min": DEFAULT_SERVICE_MIN,
        "현장지식추가시간_min": deliveries_df["name"].map(extra_delay),
        "총서비스시간_min": deliveries_df["name"].map(service_time_min)
    })

    return {
        "before_df": before_df,
        "after_df": after_df,
        "comparison_df": comparison_df,
        "reassignment_df": reassignment_df,
        "unassigned_df": unassigned_df,
        "decision_summary_df": decision_summary_df,
        "knowledge_check_df": knowledge_check_df,
        "failed_deliveries": failed_deliveries
    }


# ============================================================
# 5. 앱 시작
# ============================================================

check_required_files()

depot_df, drivers_df, deliveries_df, base_knowledge_df, routes_before_df = load_base_data()

if "knowledge_df" not in st.session_state:
    st.session_state.knowledge_df = base_knowledge_df.copy()

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# 6. 헤더
# ============================================================

st.title("🚚 라스트마일 돌발상황 대응 최적화")
st.caption("생성형 AI로 복합 현장지시를 해석하고, 실시간 차량 상태와 제약조건을 반영해 재배차하는 라스트마일 운영 지원 시스템")

col1, col2, col3 = st.columns(3)

col1.metric("등록 배송지", f"{len(deliveries_df)}개")
col2.metric("운행 차량", f"{len(drivers_df)}대")
col3.metric("등록 현장정보", f"{len(st.session_state.knowledge_df)}건")


# ============================================================
# 7. 현장 지연정보 등록
# ============================================================

st.subheader("① 현장 지연정보 등록")

with st.form("knowledge_form", clear_on_submit=True):
    c1, c2 = st.columns(2)

    delivery_name = c1.selectbox(
        "배송지",
        deliveries_df["name"].tolist()
    )

    knowledge_type = c2.selectbox(
        "상황 유형",
        ["congestion", "unloading", "access", "delay", "other"],
        format_func=lambda x: {
            "congestion": "교통 혼잡",
            "unloading": "하역 지연",
            "access": "진입 제약",
            "delay": "기타 지연",
            "other": "기타"
        }[x]
    )

    c3, c4 = st.columns(2)

    extra_delay_min = c3.number_input(
        "추가 지연시간(분)",
        min_value=0,
        max_value=180,
        value=10,
        step=5
    )

    description = c4.text_input(
        "설명",
        placeholder="예: 하역장 공사"
    )

    add_knowledge = st.form_submit_button(
        "현장정보 등록",
        type="primary",
        use_container_width=True
    )


if add_knowledge:
    current_df = st.session_state.knowledge_df.copy()

    existing_numbers = (
        current_df["knowledge_id"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .dropna()
    )

    if len(existing_numbers) == 0:
        next_number = 1
    else:
        next_number = existing_numbers.astype(int).max() + 1

    new_row = pd.DataFrame([{
        "knowledge_id": f"K{next_number:03d}",
        "location": delivery_name,
        "knowledge_type": knowledge_type,
        "description": description if description else "사용자 직접 입력",
        "extra_delay_min": int(extra_delay_min)
    }])

    st.session_state.knowledge_df = pd.concat(
        [current_df, new_row],
        ignore_index=True
    )

    st.session_state.last_result = None

    st.success(
        f"{delivery_name}에 {int(extra_delay_min)}분 지연정보를 등록했습니다."
    )


c1, c2 = st.columns([4, 1])

with c1:
    with st.expander("현재 등록된 현장정보 보기"):
        st.dataframe(
            st.session_state.knowledge_df,
            use_container_width=True,
            hide_index=True
        )

with c2:
    if st.button("현장정보 초기화", use_container_width=True):
        st.session_state.knowledge_df = base_knowledge_df.copy()
        st.session_state.last_result = None
        st.rerun()



# ============================================================
# 8. 동적 배송 시뮬레이션 / 돌발상황 재배차
# ============================================================

st.divider()
st.subheader("② 실시간 라스트마일 배송 시뮬레이션")

SIM_MIN_PER_TICK = 1.0
AUTO_REFRESH_SEC = 1.0


def make_original_routes(routes_df):
    routes = {}

    for _, row in routes_df.iterrows():
        vehicle_id = int(row["vehicle_id"])
        routes[vehicle_id] = extract_delivery_names(row["route"])

    return routes


def make_service_times(deliveries_df, knowledge_df):
    extra_delay = {
        name: 0
        for name in deliveries_df["name"]
    }

    for _, row in knowledge_df.iterrows():
        location = row["location"]

        if location in extra_delay:
            extra_delay[location] += int(row["extra_delay_min"])

    return {
        name: DEFAULT_SERVICE_MIN + extra_delay[name]
        for name in deliveries_df["name"]
    }


PATH_CACHE = {}


def get_path_metrics(G, start_node, end_node):
    key = (start_node, end_node)

    if key in PATH_CACHE:
        return PATH_CACHE[key]

    path = nx.shortest_path(
        G,
        start_node,
        end_node,
        weight="travel_time_sec"
    )

    distance_m = 0.0
    travel_time_sec = 0.0

    for i in range(len(path) - 1):
        edge = G[path[i]][path[i + 1]]
        distance_m += float(edge["length_m"])
        travel_time_sec += float(edge["travel_time_sec"])

    result = {
        "path": path,
        "distance_m": distance_m,
        "travel_time_sec": travel_time_sec
    }

    PATH_CACHE[key] = result
    return result


def get_node_at_elapsed_seconds(G, path_nodes, elapsed_sec):
    if not path_nodes:
        return None

    if len(path_nodes) == 1 or elapsed_sec <= 0:
        return path_nodes[0]

    cumulative = 0.0

    for i in range(len(path_nodes) - 1):
        edge = G[path_nodes[i]][path_nodes[i + 1]]
        segment_sec = float(edge["travel_time_sec"])

        if cumulative + segment_sec >= elapsed_sec:
            return path_nodes[i]

        cumulative += segment_sec

    return path_nodes[-1]


def simulate_vehicle_from_snapshot(
    G,
    snapshot,
    elapsed_min,
    depot_node,
    delivery_nodes,
    service_time_min
):
    current_node = snapshot["current_node"]
    route = snapshot["remaining_route"].copy()
    service_remaining_sec = float(
        snapshot.get("service_remaining_sec", 0.0)
    )
    active = bool(snapshot.get("active", True))

    if not active:
        return {
            "current_node": current_node,
            "remaining_route": route,
            "service_remaining_sec": service_remaining_sec,
            "status": "운행 중단",
            "next_delivery": route[0] if route else "-",
            "completed_now": [],
            "active": False
        }

    remaining_sec = max(0.0, elapsed_min * 60)
    completed_now = []

    # epoch 시작 시 이미 배송지에서 하역 중이었다면 남은 서비스부터 처리
    if service_remaining_sec > 0 and route:
        if remaining_sec < service_remaining_sec:
            return {
                "current_node": current_node,
                "remaining_route": route,
                "service_remaining_sec": service_remaining_sec - remaining_sec,
                "status": f"{route[0]} 배송/하역 중",
                "next_delivery": route[0],
                "completed_now": completed_now,
                "active": True
            }

        remaining_sec -= service_remaining_sec
        completed_now.append(route[0])
        route = route[1:]
        service_remaining_sec = 0.0

    while route:
        delivery_name = route[0]
        target_node = delivery_nodes[delivery_name]

        path_result = get_path_metrics(
            G,
            current_node,
            target_node
        )

        travel_sec = path_result["travel_time_sec"]

        if remaining_sec < travel_sec:
            current_node = get_node_at_elapsed_seconds(
                G,
                path_result["path"],
                remaining_sec
            )

            return {
                "current_node": current_node,
                "remaining_route": route,
                "service_remaining_sec": 0.0,
                "status": f"{delivery_name}로 이동 중",
                "next_delivery": delivery_name,
                "completed_now": completed_now,
                "active": True
            }

        remaining_sec -= travel_sec
        current_node = target_node

        service_sec = float(
            service_time_min[delivery_name]
        ) * 60

        if remaining_sec < service_sec:
            return {
                "current_node": current_node,
                "remaining_route": route,
                "service_remaining_sec": service_sec - remaining_sec,
                "status": f"{delivery_name} 배송/하역 중",
                "next_delivery": delivery_name,
                "completed_now": completed_now,
                "active": True
            }

        remaining_sec -= service_sec
        completed_now.append(delivery_name)
        route = route[1:]

    # 배송 완료 후 물류센터 복귀
    if current_node != depot_node:
        path_result = get_path_metrics(
            G,
            current_node,
            depot_node
        )

        if remaining_sec < path_result["travel_time_sec"]:
            current_node = get_node_at_elapsed_seconds(
                G,
                path_result["path"],
                remaining_sec
            )

            return {
                "current_node": current_node,
                "remaining_route": [],
                "service_remaining_sec": 0.0,
                "status": "물류센터 복귀 중",
                "next_delivery": "-",
                "completed_now": completed_now,
                "active": True
            }

        current_node = depot_node

    return {
        "current_node": current_node,
        "remaining_route": [],
        "service_remaining_sec": 0.0,
        "status": "운행 완료",
        "next_delivery": "-",
        "completed_now": completed_now,
        "active": True
    }


def get_current_states(
    G,
    epoch_snapshots,
    epoch_start_min,
    sim_time_min,
    depot_node,
    delivery_nodes,
    service_time_min
):
    elapsed_min = sim_time_min - epoch_start_min

    return {
        vehicle_id: simulate_vehicle_from_snapshot(
            G=G,
            snapshot=snapshot,
            elapsed_min=elapsed_min,
            depot_node=depot_node,
            delivery_nodes=delivery_nodes,
            service_time_min=service_time_min
        )
        for vehicle_id, snapshot in epoch_snapshots.items()
    }


def calculate_remaining_route(
    G,
    state,
    route,
    depot_node,
    delivery_nodes,
    service_time_min
):
    current_node = state["current_node"]
    working_route = route.copy()

    total_distance_m = 0.0
    total_travel_time_sec = 0.0
    total_service_min = 0.0

    # 현재 배송지에서 이미 하역 중이면 그 배송지는 고정
    if state["service_remaining_sec"] > 0 and working_route:
        total_service_min += (
            state["service_remaining_sec"] / 60
        )
        current_node = delivery_nodes[working_route[0]]
        working_route = working_route[1:]

    for delivery_name in working_route:
        target_node = delivery_nodes[delivery_name]

        result = get_path_metrics(
            G,
            current_node,
            target_node
        )

        total_distance_m += result["distance_m"]
        total_travel_time_sec += result["travel_time_sec"]
        total_service_min += service_time_min[delivery_name]

        current_node = target_node

    result = get_path_metrics(
        G,
        current_node,
        depot_node
    )

    total_distance_m += result["distance_m"]
    total_travel_time_sec += result["travel_time_sec"]

    return {
        "distance_km": total_distance_m / 1000,
        "travel_time_min": total_travel_time_sec / 60,
        "service_time_min": total_service_min,
        "remaining_time_min": (
            total_travel_time_sec / 60
            + total_service_min
        )
    }


def evaluate_vehicle_candidate(
    G,
    delivery_name,
    vehicle_id,
    current_route,
    state,
    depot_node,
    delivery_nodes,
    service_time_min,
    max_remaining_time_min,
    deliveries_df=None,
    drivers_df=None
):
    """
    배송 1건을 특정 차량에 넣을 수 있는지 평가하고,
    가장 좋은 삽입 위치와 제외 사유를 함께 반환한다.
    """

    current_metrics = calculate_remaining_route(
        G,
        state,
        current_route,
        depot_node,
        delivery_nodes,
        service_time_min
    )

    capacity = None
    current_demand = None
    delivery_demand = None
    candidate_demand = None

    if (
        deliveries_df is not None
        and drivers_df is not None
    ):
        capacity = vehicle_capacity_value(
            vehicle_id,
            drivers_df
        )

        current_demand = route_total_demand(
            current_route,
            deliveries_df
        )

        delivery_demand = route_total_demand(
            [delivery_name],
            deliveries_df
        )

        candidate_demand = (
            current_demand
            + delivery_demand
        )

        if (
            capacity is not None
            and candidate_demand > capacity
        ):
            return None, {
                "delivery": delivery_name,
                "vehicle_id": vehicle_id,
                "capacity": capacity,
                "current_demand": current_demand,
                "delivery_demand": delivery_demand,
                "candidate_demand": candidate_demand,
                "added_time_min": np.nan,
                "added_distance_km": np.nan,
                "final_remaining_time_min": np.nan,
                "status": "제외",
                "reason": (
                    f"적재용량 초과 "
                    f"({current_demand}+{delivery_demand}>{capacity})"
                )
            }

    min_position = (
        1
        if (
            state["service_remaining_sec"] > 0
            and current_route
        )
        else 0
    )

    all_insertions = []

    for insert_position in range(
        min_position,
        len(current_route) + 1
    ):
        candidate_route = (
            current_route.copy()
        )
        candidate_route.insert(
            insert_position,
            delivery_name
        )

        metrics = calculate_remaining_route(
            G,
            state,
            candidate_route,
            depot_node,
            delivery_nodes,
            service_time_min
        )

        all_insertions.append({
            "insert_position": insert_position,
            "route": candidate_route,
            "metrics": metrics,
            "added_time_min": (
                metrics["remaining_time_min"]
                - current_metrics["remaining_time_min"]
            ),
            "added_distance_km": (
                metrics["distance_km"]
                - current_metrics["distance_km"]
            )
        })

    if not all_insertions:
        return None, {
            "delivery": delivery_name,
            "vehicle_id": vehicle_id,
            "capacity": capacity,
            "current_demand": current_demand,
            "delivery_demand": delivery_demand,
            "candidate_demand": candidate_demand,
            "added_time_min": np.nan,
            "added_distance_km": np.nan,
            "final_remaining_time_min": np.nan,
            "status": "제외",
            "reason": "삽입 가능한 위치 없음"
        }

    all_insertions.sort(
        key=lambda x: (
            x["added_time_min"],
            x["added_distance_km"],
            x["metrics"]["remaining_time_min"]
        )
    )

    best_any = all_insertions[0]

    feasible = [
        x
        for x in all_insertions
        if (
            x["metrics"]["remaining_time_min"]
            <= max_remaining_time_min
        )
    ]

    if not feasible:
        return None, {
            "delivery": delivery_name,
            "vehicle_id": vehicle_id,
            "capacity": capacity,
            "current_demand": current_demand,
            "delivery_demand": delivery_demand,
            "candidate_demand": candidate_demand,
            "added_time_min": best_any["added_time_min"],
            "added_distance_km": best_any["added_distance_km"],
            "final_remaining_time_min": (
                best_any["metrics"]["remaining_time_min"]
            ),
            "status": "제외",
            "reason": (
                f"잔여운행시간 초과 "
                f"({best_any['metrics']['remaining_time_min']:.2f}"
                f">{max_remaining_time_min:.2f}분)"
            )
        }

    feasible.sort(
        key=lambda x: (
            x["added_time_min"],
            x["added_distance_km"],
            x["metrics"]["remaining_time_min"]
        )
    )

    best = feasible[0]

    return best, {
        "delivery": delivery_name,
        "vehicle_id": vehicle_id,
        "capacity": capacity,
        "current_demand": current_demand,
        "delivery_demand": delivery_demand,
        "candidate_demand": candidate_demand,
        "added_time_min": best["added_time_min"],
        "added_distance_km": best["added_distance_km"],
        "final_remaining_time_min": (
            best["metrics"]["remaining_time_min"]
        ),
        "status": "가능",
        "reason": "제약 만족"
    }


def reassign_failed_vehicle_dynamic(
    G,
    failed_vehicle,
    current_states,
    depot_node,
    delivery_nodes,
    service_time_min,
    max_remaining_time_min,
    deliveries_df=None,
    drivers_df=None
):
    failed_state = current_states[
        failed_vehicle
    ]

    failed_deliveries = failed_state[
        "remaining_route"
    ].copy()

    working_routes = {
        vehicle_id: state[
            "remaining_route"
        ].copy()
        for vehicle_id, state in current_states.items()
        if (
            vehicle_id != failed_vehicle
            and state["active"]
        )
    }

    assignment_rows = []
    unassigned_rows = []
    candidate_rows = []

    for delivery_name in failed_deliveries:
        feasible_candidates = []
        delivery_candidate_rows = []

        for vehicle_id, current_route in (
            working_routes.items()
        ):
            state = current_states[
                vehicle_id
            ]

            best, candidate_info = (
                evaluate_vehicle_candidate(
                    G=G,
                    delivery_name=delivery_name,
                    vehicle_id=vehicle_id,
                    current_route=current_route,
                    state=state,
                    depot_node=depot_node,
                    delivery_nodes=delivery_nodes,
                    service_time_min=service_time_min,
                    max_remaining_time_min=max_remaining_time_min,
                    deliveries_df=deliveries_df,
                    drivers_df=drivers_df
                )
            )

            delivery_candidate_rows.append(
                candidate_info
            )

            if best is not None:
                feasible_candidates.append({
                    "vehicle_id": vehicle_id,
                    "insert_position": best[
                        "insert_position"
                    ],
                    "new_route": best[
                        "route"
                    ],
                    "added_time_min": best[
                        "added_time_min"
                    ],
                    "added_distance_km": best[
                        "added_distance_km"
                    ],
                    "final_remaining_time_min": (
                        best["metrics"][
                            "remaining_time_min"
                        ]
                    )
                })

        if feasible_candidates:
            feasible_candidates.sort(
                key=lambda x: (
                    x["added_time_min"],
                    x["added_distance_km"],
                    x[
                        "final_remaining_time_min"
                    ]
                )
            )

            best = feasible_candidates[0]

            working_routes[
                best["vehicle_id"]
            ] = best["new_route"]

            assignment_rows.append({
                "delivery": delivery_name,
                "assigned_vehicle": best[
                    "vehicle_id"
                ],
                "insert_position": (
                    best[
                        "insert_position"
                    ] + 1
                ),
                "added_distance_km": best[
                    "added_distance_km"
                ],
                "added_time_min": best[
                    "added_time_min"
                ],
                "final_remaining_time_min": (
                    best[
                        "final_remaining_time_min"
                    ]
                ),
                "new_route": " → ".join(
                    best["new_route"]
                )
            })

            for row in delivery_candidate_rows:
                if (
                    row["status"] == "가능"
                    and int(row["vehicle_id"])
                    == int(best["vehicle_id"])
                ):
                    row["status"] = "선택"
                    row["reason"] = (
                        "가능 후보 중 추가 운행시간이 가장 작음"
                    )
                elif row["status"] == "가능":
                    row["status"] = "미선택"
                    row["reason"] = (
                        "제약은 만족하지만 선택 차량보다 "
                        "추가 운행시간/거리가 불리함"
                    )

        else:
            # 어떤 차량도 현재 즉시 수용할 수 없음
            available_rows = [
                row
                for row in delivery_candidate_rows
                if pd.notna(
                    row.get(
                        "final_remaining_time_min"
                    )
                )
            ]

            if available_rows:
                best_infeasible = sorted(
                    available_rows,
                    key=lambda x: (
                        max(
                            0,
                            float(
                                x[
                                    "final_remaining_time_min"
                                ]
                            )
                            - max_remaining_time_min
                        ),
                        float(
                            x.get(
                                "added_time_min",
                                999999
                            )
                        )
                    )
                )[0]

                unassigned_rows.append({
                    "delivery": delivery_name,
                    "best_candidate_vehicle": (
                        best_infeasible[
                            "vehicle_id"
                        ]
                    ),
                    "best_candidate_final_time_min": (
                        best_infeasible[
                            "final_remaining_time_min"
                        ]
                    ),
                    "overload_min": max(
                        0,
                        float(
                            best_infeasible[
                                "final_remaining_time_min"
                            ]
                        )
                        - max_remaining_time_min
                    )
                })
            else:
                unassigned_rows.append({
                    "delivery": delivery_name,
                    "best_candidate_vehicle": np.nan,
                    "best_candidate_final_time_min": np.nan,
                    "overload_min": np.nan
                })

        candidate_rows.extend(
            delivery_candidate_rows
        )

    return {
        "failed_deliveries": failed_deliveries,
        "new_routes": working_routes,
        "assignment_df": pd.DataFrame(
            assignment_rows
        ),
        "unassigned_df": pd.DataFrame(
            unassigned_rows
        ),
        "candidate_comparison_df": pd.DataFrame(
            candidate_rows
        )
    }


# ============================================================
# Gemini 기반 복합 돌발상황 자연어 해석
# ============================================================

GEMINI_MODEL = "gemini-3.5-flash"


def normalize_delivery_name(value):
    """
    '7', '07', '배송지7', '배송지_7', '배송지_07' 등을
    앱 내부 형식인 '배송지_07'로 통일.
    """
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    match = re.search(r"(\d+)", value)

    if match:
        return f"배송지_{int(match.group(1)):02d}"

    return value


def get_gemini_api_key():
    """
    로컬에서는 환경변수 GEMINI_API_KEY,
    Streamlit 배포에서는 st.secrets["GEMINI_API_KEY"]도 지원.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return None


def current_operation_context_text(current_states):
    rows = []

    for vehicle_id, state in sorted(current_states.items()):
        rows.append(
            {
                "vehicle_id": int(vehicle_id),
                "active": bool(state["active"]),
                "status": state["status"],
                "remaining_route": state["remaining_route"],
                "next_delivery": state["next_delivery"],
            }
        )

    return json.dumps(
        rows,
        ensure_ascii=False,
        indent=2
    )


def interpret_incident_with_gemini(
    instruction_text,
    current_states
):
    """
    생성형 AI의 역할:
    자연어 운행지시를 '실행 가능한 구조화 명령'으로 변환.
    배차 최적화 자체는 AI가 하지 않음.
    """
    if genai is None:
        raise RuntimeError(
            "google-genai 패키지가 없습니다. "
            "`pip install -U google-genai` 후 다시 실행하세요."
        )

    api_key = get_gemini_api_key()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되어 있지 않습니다."
        )

    schema = {
        "type": "object",
        "properties": {
            "event_type": {
                "type": "string",
                "enum": [
                    "vehicle_failure",
                    "vehicle_delay",
                    "other"
                ],
                "description": (
                    "차량 운행 불가/사고/고장은 vehicle_failure, "
                    "단순 지연은 vehicle_delay"
                )
            },
            "vehicle_id": {
                "type": ["integer", "null"],
                "description": "돌발상황이 발생한 차량 번호"
            },
            "completed_through_delivery": {
                "type": ["string", "null"],
                "description": (
                    "'배송지 7까지 완료'처럼 해당 배송지까지 "
                    "원래 순서대로 완료했다는 의미. 예: 배송지_07"
                )
            },
            "completed_deliveries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "사용자가 개별적으로 완료했다고 명시한 배송지 목록"
                )
            },
            "fixed_vehicle_for_all_remaining": {
                "type": ["integer", "null"],
                "description": (
                    "'이후 배송은 차량3이 맡는다'처럼 "
                    "고장차량의 모든 남은 배송을 특정 차량에 "
                    "넘기라고 명시한 경우의 차량 번호"
                )
            },
            "fixed_reassignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "delivery": {
                            "type": "string"
                        },
                        "vehicle_id": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "delivery",
                        "vehicle_id"
                    ]
                },
                "description": (
                    "'배송지12는 차량3이 맡는다'처럼 "
                    "특정 배송지와 차량을 사용자가 지정한 경우"
                )
            },
            "auto_reassign_remaining": {
                "type": "boolean",
                "description": (
                    "사용자가 지정하지 않은 나머지 배송을 "
                    "시스템이 자동 재배차해도 되는지"
                )
            },
            "allowed_vehicles": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "'차량 2와 3으로만 보내', "
                    "'2, 3번 차량에 나눠서 보내'처럼 "
                    "남은 배송을 받을 수 있는 차량을 제한한 경우의 차량 목록"
                )
            },
            "force_assignment": {
                "type": "boolean",
                "description": (
                    "'무조건', '강제로', '제약 무시하고'처럼 "
                    "일반 최적화 제약보다 운영자 지시를 우선해 실행하라고 "
                    "명시한 경우 true"
                )
            },
            "ignore_capacity": {
                "type": "boolean",
                "description": (
                    "사용자가 적재용량/capacity/물량 제약을 "
                    "무시하라고 명시한 경우 true"
                )
            },
            "ignore_time_limit": {
                "type": "boolean",
                "description": (
                    "사용자가 최대 운행시간/잔여 운행시간/퇴근시간 제약을 "
                    "무시하라고 명시한 경우 true"
                )
            },
            "delay_min": {
                "type": ["number", "null"],
                "description": "단순 차량 지연인 경우 지연시간(분)"
            },
            "summary": {
                "type": "string",
                "description": "사용자 지시를 한 문장으로 요약"
            }
        },
        "required": [
            "event_type",
            "vehicle_id",
            "completed_through_delivery",
            "completed_deliveries",
            "fixed_vehicle_for_all_remaining",
            "fixed_reassignments",
            "auto_reassign_remaining",
            "allowed_vehicles",
            "force_assignment",
            "ignore_capacity",
            "ignore_time_limit",
            "delay_min",
            "summary"
        ]
    }

    prompt = f"""
너는 물류 운행 관제 시스템의 자연어 명령 해석기다.

사용자의 문장을 설명문으로 답하지 말고,
반드시 제공된 JSON 스키마에 맞는 구조화 명령으로 변환하라.

중요 규칙:
1. 사용자가 말하지 않은 배송 완료, 차량 번호, 강제 재배정을 추측하지 않는다.
2. "배송지 7까지 배송함"은 completed_through_delivery="배송지_07"이다.
3. "이후 배송은 차량 3이 받기로 했다"는
   fixed_vehicle_for_all_remaining=3 으로 해석한다.
4. "배송지 12는 차량 3이 맡는다"는 fixed_reassignments에 넣는다.
5. "나머지는 알아서 재배차"가 있으면 auto_reassign_remaining=true다.
6. 차량 사고, 고장, 운행불가, 더 이상 운행 못함은 vehicle_failure다.
7. 단순히 20분 늦는다와 같은 표현만 있으면 vehicle_delay다.
8. 현재 시스템 상태와 모순되는 내용을 임의로 수정하지 말고 그대로 추출한다.
   실제 유효성 검사는 이후 프로그램이 수행한다.
9. 배송지는 가능하면 '배송지_07' 형식으로 반환한다.
10. "차량 2와 3으로만 보내", "2번과 3번 차량에 나눠서 보내"는
    allowed_vehicles=[2,3] 으로 해석한다.
11. "무조건", "강제로", "제약 무시하고"라는 표현이 있으면
    force_assignment=true 로 해석한다.
12. "적재용량 무시", "capacity 무시", "물량 한도 무시"가 있으면
    ignore_capacity=true 로 해석한다.
13. "운행시간 무시", "시간 제한 무시", "퇴근시간 상관없이"가 있으면
    ignore_time_limit=true 로 해석한다.
14. 단순히 "차량 2와 3으로 보내"라고만 하고 제약 무시를 말하지 않았다면
    force_assignment=false이며 기존 제약은 유지한다.

현재 시스템 상태:
{current_operation_context_text(current_states)}

사용자 운행 지시:
{instruction_text}
"""

    client = genai.Client(
        api_key=api_key
    )

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": schema
        }
    )

    command = json.loads(
        interaction.output_text
    )

    command["completed_through_delivery"] = (
        normalize_delivery_name(
            command.get(
                "completed_through_delivery"
            )
        )
    )

    command["completed_deliveries"] = [
        normalize_delivery_name(x)
        for x in command.get(
            "completed_deliveries",
            []
        )
        if normalize_delivery_name(x)
    ]

    normalized_fixed = []

    for item in command.get(
        "fixed_reassignments",
        []
    ):
        delivery = normalize_delivery_name(
            item.get("delivery")
        )

        if delivery:
            normalized_fixed.append(
                {
                    "delivery": delivery,
                    "vehicle_id": int(
                        item["vehicle_id"]
                    )
                }
            )

    command["fixed_reassignments"] = (
        normalized_fixed
    )

    command["allowed_vehicles"] = [
        int(x)
        for x in command.get(
            "allowed_vehicles",
            []
        )
        if x is not None
    ]

    command["force_assignment"] = bool(
        command.get(
            "force_assignment",
            False
        )
    )

    command["ignore_capacity"] = bool(
        command.get(
            "ignore_capacity",
            False
        )
    )

    command["ignore_time_limit"] = bool(
        command.get(
            "ignore_time_limit",
            False
        )
    )

    return command


def validate_ai_incident_command(
    command,
    current_states,
    delivery_nodes
):
    """
    AI 출력은 그대로 실행하지 않고 현재 시스템 상태와 대조 검증한다.
    """
    errors = []
    warnings = []

    if command.get("event_type") != "vehicle_failure":
        errors.append(
            "현재 MVP에서 자연어 자동 실행은 차량 고장/사고 상황만 지원합니다."
        )

    vehicle_id = command.get("vehicle_id")

    if vehicle_id is None:
        errors.append(
            "고장 차량 번호를 확인할 수 없습니다."
        )
        return errors, warnings

    vehicle_id = int(vehicle_id)

    if vehicle_id not in current_states:
        errors.append(
            f"차량 {vehicle_id}가 현재 시스템에 없습니다."
        )
        return errors, warnings

    if not current_states[vehicle_id]["active"]:
        errors.append(
            f"차량 {vehicle_id}는 이미 운행 중단 상태입니다."
        )

    all_deliveries = set(
        delivery_nodes.keys()
    )

    through = command.get(
        "completed_through_delivery"
    )

    if (
        through is not None
        and through not in all_deliveries
    ):
        errors.append(
            f"{through}는 존재하지 않는 배송지입니다."
        )

    for delivery in command.get(
        "completed_deliveries",
        []
    ):
        if delivery not in all_deliveries:
            errors.append(
                f"{delivery}는 존재하지 않는 배송지입니다."
            )

    fixed_vehicle = command.get(
        "fixed_vehicle_for_all_remaining"
    )

    if fixed_vehicle is not None:
        fixed_vehicle = int(fixed_vehicle)

        if fixed_vehicle == vehicle_id:
            errors.append(
                "고장 차량 자신에게 남은 배송을 재배정할 수 없습니다."
            )
        elif fixed_vehicle not in current_states:
            errors.append(
                f"지정 차량 {fixed_vehicle}가 존재하지 않습니다."
            )
        elif not current_states[
            fixed_vehicle
        ]["active"]:
            errors.append(
                f"지정 차량 {fixed_vehicle}는 현재 운행할 수 없습니다."
            )

    for item in command.get(
        "fixed_reassignments",
        []
    ):
        delivery = item["delivery"]
        target_vehicle = int(
            item["vehicle_id"]
        )

        if delivery not in all_deliveries:
            errors.append(
                f"{delivery}는 존재하지 않는 배송지입니다."
            )

        if target_vehicle == vehicle_id:
            errors.append(
                f"{delivery}를 고장 차량 {vehicle_id}에 지정할 수 없습니다."
            )
        elif target_vehicle not in current_states:
            errors.append(
                f"지정 차량 {target_vehicle}가 존재하지 않습니다."
            )
        elif not current_states[
            target_vehicle
        ]["active"]:
            errors.append(
                f"지정 차량 {target_vehicle}는 현재 운행할 수 없습니다."
            )

    allowed_vehicles = [
        int(x)
        for x in command.get(
            "allowed_vehicles",
            []
        )
    ]

    for allowed_vehicle in allowed_vehicles:
        if allowed_vehicle == vehicle_id:
            errors.append(
                f"고장 차량 {vehicle_id}는 재배차 대상 차량으로 사용할 수 없습니다."
            )
        elif allowed_vehicle not in current_states:
            errors.append(
                f"허용 차량 {allowed_vehicle}가 현재 시스템에 없습니다."
            )
        elif not current_states[
            allowed_vehicle
        ]["active"]:
            errors.append(
                f"허용 차량 {allowed_vehicle}는 현재 운행할 수 없습니다."
            )

    if command.get(
        "force_assignment",
        False
    ):
        warnings.append(
            "운영자 강제 실행 모드입니다. 지정한 제약 무시 조건에 따라 "
            "차량 적재용량 또는 최대 운행시간을 초과할 수 있습니다."
        )

    if (
        command.get(
            "ignore_capacity",
            False
        )
        and not command.get(
            "force_assignment",
            False
        )
    ):
        warnings.append(
            "적재용량 무시 요청이 있으므로 강제 실행 모드로 처리합니다."
        )
        command[
            "force_assignment"
        ] = True

    if (
        command.get(
            "ignore_time_limit",
            False
        )
        and not command.get(
            "force_assignment",
            False
        )
    ):
        warnings.append(
            "운행시간 제한 무시 요청이 있으므로 강제 실행 모드로 처리합니다."
        )
        command[
            "force_assignment"
        ] = True

    return errors, warnings


def apply_completed_override(
    command,
    failed_state,
    delivery_nodes
):
    """
    '배송지 7까지 완료' 같은 현장 제보를 현재 상태에 반영.
    사용자가 더 최신 현장정보를 전달한 것으로 간주한다.
    """
    state = {
        "current_node": failed_state["current_node"],
        "remaining_route": failed_state[
            "remaining_route"
        ].copy(),
        "service_remaining_sec": failed_state[
            "service_remaining_sec"
        ],
        "status": failed_state["status"],
        "next_delivery": failed_state[
            "next_delivery"
        ],
        "active": failed_state["active"]
    }

    route = state["remaining_route"].copy()

    through = command.get(
        "completed_through_delivery"
    )

    if through and through in route:
        idx = route.index(through)
        route = route[idx + 1:]
        state["current_node"] = (
            delivery_nodes[through]
        )
        state["service_remaining_sec"] = 0.0

    individually_completed = set(
        command.get(
            "completed_deliveries",
            []
        )
    )

    route = [
        delivery
        for delivery in route
        if delivery not in individually_completed
    ]

    state["remaining_route"] = route
    state["next_delivery"] = (
        route[0]
        if route
        else "-"
    )

    return state


def route_total_demand(
    route,
    deliveries_df
):
    demand_map = dict(
        zip(
            deliveries_df["name"],
            deliveries_df["demand"]
        )
    )

    return sum(
        int(demand_map.get(name, 0))
        for name in route
    )


def vehicle_capacity_value(
    vehicle_id,
    drivers_df
):
    if "capacity" not in drivers_df.columns:
        return None

    row = drivers_df[
        drivers_df["vehicle_id"].astype(int)
        == int(vehicle_id)
    ]

    if row.empty:
        return None

    return int(
        row.iloc[0]["capacity"]
    )


def best_insertion_for_fixed_vehicle(
    G,
    vehicle_id,
    delivery_name,
    current_states,
    working_route,
    depot_node,
    delivery_nodes,
    service_time_min,
    max_remaining_time_min,
    deliveries_df,
    drivers_df
):
    """
    사용자가 '이 배송은 차량3'이라고 지정한 경우
    차량 선택은 고정하고, 그 차량 안에서 최적 삽입 위치만 계산.
    """
    state = current_states[
        vehicle_id
    ]

    min_position = (
        1
        if state["service_remaining_sec"] > 0
        and working_route
        else 0
    )

    candidates = []

    capacity = vehicle_capacity_value(
        vehicle_id,
        drivers_df
    )

    for insert_position in range(
        min_position,
        len(working_route) + 1
    ):
        candidate_route = (
            working_route.copy()
        )

        candidate_route.insert(
            insert_position,
            delivery_name
        )

        if capacity is not None:
            if (
                route_total_demand(
                    candidate_route,
                    deliveries_df
                )
                > capacity
            ):
                continue

        metrics = calculate_remaining_route(
            G,
            state,
            candidate_route,
            depot_node,
            delivery_nodes,
            service_time_min
        )

        if (
            metrics["remaining_time_min"]
            > max_remaining_time_min
        ):
            continue

        candidates.append(
            {
                "insert_position": insert_position,
                "route": candidate_route,
                "metrics": metrics
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["metrics"]["remaining_time_min"],
            x["metrics"]["distance_km"]
        )
    )

    return candidates[0]


def best_forced_insertion(
    G,
    vehicle_id,
    delivery_name,
    current_states,
    working_route,
    depot_node,
    delivery_nodes,
    service_time_min,
    max_remaining_time_min,
    deliveries_df,
    drivers_df,
    ignore_capacity=False,
    ignore_time_limit=False
):
    """
    운영자 강제지시용 삽입.
    지정한 제약은 위반을 허용하되 위반 정도를 결과에 기록한다.
    """
    state = current_states[
        vehicle_id
    ]

    min_position = (
        1
        if (
            state["service_remaining_sec"] > 0
            and working_route
        )
        else 0
    )

    capacity = vehicle_capacity_value(
        vehicle_id,
        drivers_df
    )

    current_demand = route_total_demand(
        working_route,
        deliveries_df
    )

    delivery_demand = route_total_demand(
        [delivery_name],
        deliveries_df
    )

    candidate_demand = (
        current_demand
        + delivery_demand
    )

    capacity_over = (
        max(
            0,
            candidate_demand - capacity
        )
        if capacity is not None
        else 0
    )

    candidates = []

    for insert_position in range(
        min_position,
        len(working_route) + 1
    ):
        candidate_route = (
            working_route.copy()
        )
        candidate_route.insert(
            insert_position,
            delivery_name
        )

        metrics = calculate_remaining_route(
            G,
            state,
            candidate_route,
            depot_node,
            delivery_nodes,
            service_time_min
        )

        time_over = max(
            0.0,
            metrics["remaining_time_min"]
            - max_remaining_time_min
        )

        # 무시하지 않은 제약은 여전히 지켜야 함
        if (
            capacity_over > 0
            and not ignore_capacity
        ):
            continue

        if (
            time_over > 0
            and not ignore_time_limit
        ):
            continue

        current_metrics = calculate_remaining_route(
            G,
            state,
            working_route,
            depot_node,
            delivery_nodes,
            service_time_min
        )

        candidates.append({
            "insert_position": insert_position,
            "route": candidate_route,
            "metrics": metrics,
            "added_time_min": (
                metrics["remaining_time_min"]
                - current_metrics[
                    "remaining_time_min"
                ]
            ),
            "added_distance_km": (
                metrics["distance_km"]
                - current_metrics[
                    "distance_km"
                ]
            ),
            "capacity": capacity,
            "current_demand": current_demand,
            "delivery_demand": delivery_demand,
            "candidate_demand": candidate_demand,
            "capacity_over": capacity_over,
            "time_over": time_over
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["added_time_min"],
            x["added_distance_km"],
            x["metrics"][
                "remaining_time_min"
            ]
        )
    )

    return candidates[0]


def force_assign_to_allowed_vehicles(
    command,
    failed_vehicle,
    failed_deliveries,
    current_states,
    working_routes,
    G,
    depot_node,
    delivery_nodes,
    service_time_min,
    max_remaining_time_min,
    deliveries_df,
    drivers_df
):
    """
    '제약 무시하고 차량 2,3으로만 보내' 같은 지시를 실행한다.
    차량 선택은 allowed_vehicles 내에서만 하며,
    사용자가 명시한 제약만 무시한다.
    """
    allowed_vehicles = [
        int(v)
        for v in command.get(
            "allowed_vehicles",
            []
        )
        if (
            int(v) != failed_vehicle
            and int(v) in working_routes
        )
    ]

    if not allowed_vehicles:
        return {
            "new_routes": working_routes,
            "assignment_df": pd.DataFrame(),
            "unassigned_df": pd.DataFrame(
                [
                    {
                        "delivery": delivery,
                        "reason": "강제 배정 가능한 허용 차량 없음"
                    }
                    for delivery in failed_deliveries
                ]
            ),
            "candidate_comparison_df": pd.DataFrame()
        }

    ignore_capacity = bool(
        command.get(
            "ignore_capacity",
            False
        )
    )
    ignore_time_limit = bool(
        command.get(
            "ignore_time_limit",
            False
        )
    )

    # "제약 무시"라고만 말한 경우에는 두 제약 모두 무시
    if (
        command.get(
            "force_assignment",
            False
        )
        and not ignore_capacity
        and not ignore_time_limit
    ):
        ignore_capacity = True
        ignore_time_limit = True

    assignment_rows = []
    unassigned_rows = []
    candidate_rows = []

    for delivery in failed_deliveries:
        feasible = []

        for vehicle_id in allowed_vehicles:
            route = working_routes[
                vehicle_id
            ]

            best = best_forced_insertion(
                G=G,
                vehicle_id=vehicle_id,
                delivery_name=delivery,
                current_states=current_states,
                working_route=route,
                depot_node=depot_node,
                delivery_nodes=delivery_nodes,
                service_time_min=service_time_min,
                max_remaining_time_min=max_remaining_time_min,
                deliveries_df=deliveries_df,
                drivers_df=drivers_df,
                ignore_capacity=ignore_capacity,
                ignore_time_limit=ignore_time_limit
            )

            if best is None:
                candidate_rows.append({
                    "delivery": delivery,
                    "vehicle_id": vehicle_id,
                    "status": "제외",
                    "reason": (
                        "운영자가 무시하도록 지정하지 않은 제약을 위반"
                    )
                })
                continue

            violations = []

            if best["capacity_over"] > 0:
                violations.append(
                    f"용량 {best['capacity_over']} 초과"
                )

            if best["time_over"] > 0:
                violations.append(
                    f"시간 {best['time_over']:.2f}분 초과"
                )

            candidate_rows.append({
                "delivery": delivery,
                "vehicle_id": vehicle_id,
                "capacity": best["capacity"],
                "current_demand": best[
                    "current_demand"
                ],
                "delivery_demand": best[
                    "delivery_demand"
                ],
                "candidate_demand": best[
                    "candidate_demand"
                ],
                "added_time_min": best[
                    "added_time_min"
                ],
                "added_distance_km": best[
                    "added_distance_km"
                ],
                "final_remaining_time_min": (
                    best["metrics"][
                        "remaining_time_min"
                    ]
                ),
                "status": "강제 후보",
                "reason": (
                    ", ".join(violations)
                    if violations
                    else "제약 위반 없음"
                )
            })

            feasible.append({
                "vehicle_id": vehicle_id,
                "best": best
            })

        if not feasible:
            unassigned_rows.append({
                "delivery": delivery,
                "reason": "허용 차량에 강제 배정할 수 없음"
            })
            continue

        feasible.sort(
            key=lambda x: (
                x["best"][
                    "added_time_min"
                ],
                x["best"][
                    "added_distance_km"
                ]
            )
        )

        chosen = feasible[0]
        vehicle_id = chosen[
            "vehicle_id"
        ]
        best = chosen[
            "best"
        ]

        working_routes[
            vehicle_id
        ] = best[
            "route"
        ]

        for row in candidate_rows:
            if (
                row.get("delivery")
                == delivery
                and row.get("vehicle_id")
                == vehicle_id
                and row.get("status")
                == "강제 후보"
            ):
                row["status"] = "강제 선택"
                row["reason"] = (
                    "운영자 강제 지시에 따라 허용 차량 중 선택 / "
                    + row["reason"]
                )

        violations = []

        if best["capacity_over"] > 0:
            violations.append(
                f"적재용량 {best['capacity_over']} 초과"
            )

        if best["time_over"] > 0:
            violations.append(
                f"운행시간 {best['time_over']:.2f}분 초과"
            )

        assignment_rows.append({
            "delivery": delivery,
            "requested_vehicle": (
                ",".join(
                    str(v)
                    for v in allowed_vehicles
                )
            ),
            "assigned_vehicle": vehicle_id,
            "insert_position": (
                best["insert_position"] + 1
            ),
            "added_distance_km": (
                best["added_distance_km"]
            ),
            "added_time_min": (
                best["added_time_min"]
            ),
            "final_remaining_time_min": (
                best["metrics"][
                    "remaining_time_min"
                ]
            ),
            "new_route": " → ".join(
                best["route"]
            ),
            "assignment_type": "운영자 강제 배정",
            "reason": (
                ", ".join(violations)
                if violations
                else "제약 위반 없음"
            )
        })

    return {
        "new_routes": working_routes,
        "assignment_df": pd.DataFrame(
            assignment_rows
        ),
        "unassigned_df": pd.DataFrame(
            unassigned_rows
        ),
        "candidate_comparison_df": pd.DataFrame(
            candidate_rows
        )
    }


def execute_ai_failure_command(
    command,
    current_states,
    G,
    depot_node,
    delivery_nodes,
    service_time_min,
    max_remaining_time_min,
    deliveries_df,
    drivers_df
):
    """
    자연어 지시 실행 로직

    핵심 원칙
    1. 운영자가 특정 차량을 지정한 경우 우선 반영한다.
    2. 적재용량/최대 운행시간을 초과하면 해당 지시를 전부 강제하지 않는다.
    3. 지정 차량에 실현 가능한 배송만 우선 배정한다.
    4. 수용하지 못한 나머지는 자동 재배차 대상으로 넘긴다.
    5. 화면에는 '운영자 지시 반영'과 '제약으로 인한 대체배정'을 구분해 보여준다.
    """

    failed_vehicle = int(
        command["vehicle_id"]
    )

    adjusted_states = {
        vehicle_id: {
            "current_node": state["current_node"],
            "remaining_route": state["remaining_route"].copy(),
            "service_remaining_sec": state["service_remaining_sec"],
            "status": state["status"],
            "next_delivery": state["next_delivery"],
            "active": state["active"]
        }
        for vehicle_id, state in current_states.items()
    }

    adjusted_failed_state = (
        apply_completed_override(
            command,
            adjusted_states[
                failed_vehicle
            ],
            delivery_nodes
        )
    )

    adjusted_states[
        failed_vehicle
    ] = adjusted_failed_state

    original_failed_remaining = (
        adjusted_failed_state[
            "remaining_route"
        ].copy()
    )

    # --------------------------------------------------------
    # 운영자 지정 조건 정리
    # --------------------------------------------------------
    preferred_map = {}

    fixed_vehicle_all = command.get(
        "fixed_vehicle_for_all_remaining"
    )

    if fixed_vehicle_all is not None:
        for delivery in original_failed_remaining:
            preferred_map[
                delivery
            ] = int(
                fixed_vehicle_all
            )

    for item in command.get(
        "fixed_reassignments",
        []
    ):
        delivery = item[
            "delivery"
        ]

        if delivery in original_failed_remaining:
            preferred_map[
                delivery
            ] = int(
                item[
                    "vehicle_id"
                ]
            )

    working_routes = {
        vehicle_id: state[
            "remaining_route"
        ].copy()
        for vehicle_id, state in adjusted_states.items()
        if (
            vehicle_id != failed_vehicle
            and state["active"]
        )
    }

    # --------------------------------------------------------
    # 운영자 강제 실행
    # --------------------------------------------------------
    if command.get(
        "force_assignment",
        False
    ):
        forced_deliveries = (
            original_failed_remaining.copy()
        )

        forced_result = (
            force_assign_to_allowed_vehicles(
                command=command,
                failed_vehicle=failed_vehicle,
                failed_deliveries=forced_deliveries,
                current_states=adjusted_states,
                working_routes=working_routes,
                G=G,
                depot_node=depot_node,
                delivery_nodes=delivery_nodes,
                service_time_min=service_time_min,
                max_remaining_time_min=max_remaining_time_min,
                deliveries_df=deliveries_df,
                drivers_df=drivers_df
            )
        )

        return {
            "failed_deliveries": original_failed_remaining,
            "new_routes": forced_result[
                "new_routes"
            ],
            "assignment_df": forced_result[
                "assignment_df"
            ],
            "unassigned_df": forced_result[
                "unassigned_df"
            ],
            "adjusted_failed_state": adjusted_failed_state,
            "deferred_preference_df": pd.DataFrame(),
            "candidate_comparison_df": (
                forced_result[
                    "candidate_comparison_df"
                ]
            ),
            "force_mode": True
        }

    preferred_rows = []
    deferred_deliveries = []

    # --------------------------------------------------------
    # 특정 차량 지정 배송을 우선 처리
    # --------------------------------------------------------
    preferred_deliveries = [
        delivery
        for delivery in original_failed_remaining
        if delivery in preferred_map
    ]

    # 큰 물량부터 넣으면 잔여용량 활용이 너무 불안정할 수 있어
    # MVP에서는 기존 배송 순서를 유지하면서 순차적으로 우선 반영
    for delivery in preferred_deliveries:

        target_vehicle = int(
            preferred_map[
                delivery
            ]
        )

        # 지정 차량 자체가 현재 운행 불가한 경우
        if target_vehicle not in working_routes:
            deferred_deliveries.append(
                {
                    "delivery": delivery,
                    "preferred_vehicle": target_vehicle,
                    "reason": "지정 차량 운행 불가"
                }
            )
            continue

        current_route = working_routes[
            target_vehicle
        ]

        before_metrics = calculate_remaining_route(
            G,
            adjusted_states[
                target_vehicle
            ],
            current_route,
            depot_node,
            delivery_nodes,
            service_time_min
        )

        best = best_insertion_for_fixed_vehicle(
            G=G,
            vehicle_id=target_vehicle,
            delivery_name=delivery,
            current_states=adjusted_states,
            working_route=current_route,
            depot_node=depot_node,
            delivery_nodes=delivery_nodes,
            service_time_min=service_time_min,
            max_remaining_time_min=max_remaining_time_min,
            deliveries_df=deliveries_df,
            drivers_df=drivers_df
        )

        if best is None:
            capacity = vehicle_capacity_value(
                target_vehicle,
                drivers_df
            )

            current_demand = route_total_demand(
                current_route,
                deliveries_df
            )

            delivery_demand = route_total_demand(
                [delivery],
                deliveries_df
            )

            if (
                capacity is not None
                and current_demand
                + delivery_demand
                > capacity
            ):
                reason = (
                    f"적재용량 초과 "
                    f"({current_demand}+{delivery_demand}"
                    f">{capacity})"
                )
            else:
                reason = (
                    "최대 잔여 운행시간 제약 초과"
                )

            deferred_deliveries.append(
                {
                    "delivery": delivery,
                    "preferred_vehicle": target_vehicle,
                    "reason": reason
                }
            )
            continue

        working_routes[
            target_vehicle
        ] = best[
            "route"
        ]

        preferred_rows.append(
            {
                "delivery": delivery,
                "requested_vehicle": target_vehicle,
                "assigned_vehicle": target_vehicle,
                "insert_position": (
                    best[
                        "insert_position"
                    ] + 1
                ),
                "added_distance_km": (
                    best[
                        "metrics"
                    ][
                        "distance_km"
                    ]
                    - before_metrics[
                        "distance_km"
                    ]
                ),
                "added_time_min": (
                    best[
                        "metrics"
                    ][
                        "remaining_time_min"
                    ]
                    - before_metrics[
                        "remaining_time_min"
                    ]
                ),
                "final_remaining_time_min": (
                    best[
                        "metrics"
                    ][
                        "remaining_time_min"
                    ]
                ),
                "new_route": " → ".join(
                    best[
                        "route"
                    ]
                ),
                "assignment_type": (
                    "운영자 지정 반영"
                ),
                "reason": "지정 차량 제약 만족"
            }
        )

    # --------------------------------------------------------
    # 자동 재배차 대상
    # - 애초에 지정되지 않은 배송
    # - 지정 차량이 수용하지 못한 배송
    # --------------------------------------------------------
    preferred_success = {
        row[
            "delivery"
        ]
        for row in preferred_rows
    }

    deferred_names = {
        row[
            "delivery"
        ]
        for row in deferred_deliveries
    }

    auto_remaining = [
        delivery
        for delivery in original_failed_remaining
        if (
            delivery not in preferred_success
        )
    ]

    auto_states = {
        vehicle_id: {
            "current_node": state[
                "current_node"
            ],
            "remaining_route": (
                working_routes.get(
                    vehicle_id,
                    state[
                        "remaining_route"
                    ]
                ).copy()
                if vehicle_id != failed_vehicle
                else auto_remaining.copy()
            ),
            "service_remaining_sec": state[
                "service_remaining_sec"
            ],
            "status": state[
                "status"
            ],
            "next_delivery": state[
                "next_delivery"
            ],
            "active": state[
                "active"
            ]
        }
        for vehicle_id, state in adjusted_states.items()
    }

    candidate_comparison_df = pd.DataFrame()

    if (
        auto_remaining
        and command.get(
            "auto_reassign_remaining",
            True
        )
    ):
        auto_result = (
            reassign_failed_vehicle_dynamic(
                G=G,
                failed_vehicle=failed_vehicle,
                current_states=auto_states,
                depot_node=depot_node,
                delivery_nodes=delivery_nodes,
                service_time_min=service_time_min,
                max_remaining_time_min=max_remaining_time_min,
                deliveries_df=deliveries_df,
                drivers_df=drivers_df
            )
        )

        final_routes = (
            auto_result[
                "new_routes"
            ]
        )

        candidate_comparison_df = (
            auto_result.get(
                "candidate_comparison_df",
                pd.DataFrame()
            ).copy()
        )

        auto_assignment_df = (
            auto_result[
                "assignment_df"
            ].copy()
        )

        auto_unassigned_df = (
            auto_result[
                "unassigned_df"
            ].copy()
        )

        if not auto_assignment_df.empty:
            auto_assignment_df[
                "requested_vehicle"
            ] = auto_assignment_df[
                "delivery"
            ].map(
                {
                    row[
                        "delivery"
                    ]: row[
                        "preferred_vehicle"
                    ]
                    for row in deferred_deliveries
                }
            )

            auto_assignment_df[
                "assignment_type"
            ] = np.where(
                auto_assignment_df[
                    "delivery"
                ].isin(
                    deferred_names
                ),
                "제약으로 대체배정",
                "자동 재배차"
            )

            defer_reason_map = {
                row[
                    "delivery"
                ]: row[
                    "reason"
                ]
                for row in deferred_deliveries
            }

            auto_assignment_df[
                "reason"
            ] = auto_assignment_df[
                "delivery"
            ].map(
                defer_reason_map
            ).fillna(
                "운영자 미지정 배송 자동 재배차"
            )

    else:
        final_routes = working_routes

        auto_assignment_df = pd.DataFrame()

        auto_unassigned_df = pd.DataFrame(
            [
                {
                    "delivery": delivery,
                    "best_candidate_vehicle": np.nan,
                    "best_candidate_final_time_min": np.nan,
                    "overload_min": np.nan,
                    "reason": (
                        "자동 재배차 미실행"
                    )
                }
                for delivery in auto_remaining
            ]
        )

    # --------------------------------------------------------
    # 결과 병합
    # --------------------------------------------------------
    preferred_df = pd.DataFrame(
        preferred_rows
    )

    assignment_frames = [
        df
        for df in [
            preferred_df,
            auto_assignment_df
        ]
        if not df.empty
    ]

    assignment_df = (
        pd.concat(
            assignment_frames,
            ignore_index=True,
            sort=False
        )
        if assignment_frames
        else pd.DataFrame()
    )

    unassigned_df = (
        auto_unassigned_df.copy()
        if not auto_unassigned_df.empty
        else pd.DataFrame()
    )

    # 대체배정까지 실패한 배송에는
    # 원래 운영자 지정 차량과 실패 사유를 남김
    if (
        not unassigned_df.empty
        and deferred_deliveries
    ):
        requested_map = {
            row[
                "delivery"
            ]: row[
                "preferred_vehicle"
            ]
            for row in deferred_deliveries
        }

        reason_map = {
            row[
                "delivery"
            ]: row[
                "reason"
            ]
            for row in deferred_deliveries
        }

        unassigned_df[
            "requested_vehicle"
        ] = unassigned_df[
            "delivery"
        ].map(
            requested_map
        )

        unassigned_df[
            "preferred_failure_reason"
        ] = unassigned_df[
            "delivery"
        ].map(
            reason_map
        )

    return {
        "failed_deliveries": original_failed_remaining,
        "new_routes": final_routes,
        "assignment_df": assignment_df,
        "unassigned_df": unassigned_df,
        "adjusted_failed_state": adjusted_failed_state,
        "deferred_preference_df": pd.DataFrame(
            deferred_deliveries
        ),
        "candidate_comparison_df": (
            candidate_comparison_df
        )
    }


# ============================================================
# 재배차 대기 큐
# - 고장 직후 수용 불가능한 배송을 '미배정'으로 확정하지 않음
# - 다른 차량이 배송을 완료해 적재 여유/운행시간 여유가 생기면 재검토
# ============================================================

def add_to_pending_queue(
    unassigned_df,
    source_vehicle,
    sim_time_min,
    requested_vehicle=None
):
    if unassigned_df is None or unassigned_df.empty:
        return

    existing = {
        item["delivery"]
        for item in st.session_state.pending_reassignments
    }

    for _, row in unassigned_df.iterrows():
        delivery = row["delivery"]

        if delivery in existing:
            continue

        requested = requested_vehicle

        if (
            "requested_vehicle" in unassigned_df.columns
            and pd.notna(row.get("requested_vehicle"))
        ):
            requested = int(
                row["requested_vehicle"]
            )

        reason = row.get(
            "preferred_failure_reason",
            row.get(
                "reason",
                "현재 차량 제약으로 즉시 재배차 불가"
            )
        )

        st.session_state.pending_reassignments.append({
            "delivery": delivery,
            "source_vehicle": int(source_vehicle),
            "requested_vehicle": requested,
            "created_min": float(sim_time_min),
            "reason": str(reason)
        })

        existing.add(delivery)


def update_failed_recovery_assignment(
    delivery,
    assigned_vehicle
):
    for failed_vehicle, records in (
        st.session_state.failed_recovery_records.items()
    ):
        for item in records:
            if (
                item.get("delivery") == delivery
                and item.get("assigned_vehicle") is None
            ):
                item["assigned_vehicle"] = int(
                    assigned_vehicle
                )
                return


def retry_pending_reassignments(
    G,
    current_states,
    depot_node,
    delivery_nodes,
    service_time_min,
    max_remaining_time_min,
    deliveries_df,
    drivers_df
):
    """
    현재 시점의 remaining_route를 기준으로
    재배차 대기 배송을 다시 검토한다.
    """

    pending = st.session_state.get(
        "pending_reassignments",
        []
    )

    if not pending:
        return None

    working_routes = {
        vehicle_id: state[
            "remaining_route"
        ].copy()
        for vehicle_id, state in current_states.items()
        if state["active"]
    }

    assigned_rows = []
    still_pending = []
    candidate_rows = []

    for item in pending:
        delivery = item[
            "delivery"
        ]

        source_vehicle = int(
            item["source_vehicle"]
        )

        requested_vehicle = item.get(
            "requested_vehicle"
        )

        feasible_candidates = []
        delivery_candidate_rows = []

        for vehicle_id, current_route in (
            working_routes.items()
        ):
            if vehicle_id == source_vehicle:
                continue

            state = current_states[
                vehicle_id
            ]

            best, candidate_info = (
                evaluate_vehicle_candidate(
                    G=G,
                    delivery_name=delivery,
                    vehicle_id=vehicle_id,
                    current_route=current_route,
                    state=state,
                    depot_node=depot_node,
                    delivery_nodes=delivery_nodes,
                    service_time_min=service_time_min,
                    max_remaining_time_min=max_remaining_time_min,
                    deliveries_df=deliveries_df,
                    drivers_df=drivers_df
                )
            )

            if (
                requested_vehicle is not None
                and int(vehicle_id)
                == int(requested_vehicle)
            ):
                candidate_info[
                    "operator_preferred"
                ] = "Y"
            else:
                candidate_info[
                    "operator_preferred"
                ] = "N"

            delivery_candidate_rows.append(
                candidate_info
            )

            if best is not None:
                feasible_candidates.append({
                    "vehicle_id": vehicle_id,
                    "preference_rank": (
                        0
                        if (
                            requested_vehicle is not None
                            and int(vehicle_id)
                            == int(requested_vehicle)
                        )
                        else 1
                    ),
                    "insert_position": best[
                        "insert_position"
                    ],
                    "route": best["route"],
                    "metrics": best[
                        "metrics"
                    ],
                    "added_time_min": best[
                        "added_time_min"
                    ],
                    "added_distance_km": best[
                        "added_distance_km"
                    ]
                })

        if not feasible_candidates:
            for row in delivery_candidate_rows:
                if row["status"] == "가능":
                    row["status"] = "미선택"

            candidate_rows.extend(
                delivery_candidate_rows
            )

            still_pending.append(
                item
            )
            continue

        feasible_candidates.sort(
            key=lambda x: (
                x["preference_rank"],
                x["added_time_min"],
                x["added_distance_km"],
                x["metrics"][
                    "remaining_time_min"
                ]
            )
        )

        best = feasible_candidates[0]

        working_routes[
            best["vehicle_id"]
        ] = best["route"]

        for row in delivery_candidate_rows:
            if (
                row["status"] == "가능"
                and int(row["vehicle_id"])
                == int(best["vehicle_id"])
            ):
                row["status"] = "선택"
                row["reason"] = (
                    "운영자 선호 또는 추가 운행시간 기준 최적 후보"
                )
            elif row["status"] == "가능":
                row["status"] = "미선택"
                row["reason"] = (
                    "제약은 만족하지만 선택 차량보다 "
                    "우선순위/추가 운행시간이 불리함"
                )

        candidate_rows.extend(
            delivery_candidate_rows
        )

        assigned_rows.append({
            "delivery": delivery,
            "assigned_vehicle": int(
                best["vehicle_id"]
            ),
            "insert_position": int(
                best["insert_position"] + 1
            ),
            "added_distance_km": float(
                best["added_distance_km"]
            ),
            "added_time_min": float(
                best["added_time_min"]
            ),
            "final_remaining_time_min": float(
                best["metrics"][
                    "remaining_time_min"
                ]
            ),
            "new_route": " → ".join(
                best["route"]
            ),
            "assignment_type": (
                "대기 후 자동 재배차"
            ),
            "requested_vehicle": (
                requested_vehicle
            )
        })

    st.session_state.pending_reassignments = (
        still_pending
    )

    if not assigned_rows:
        return {
            "new_routes": working_routes,
            "assignment_df": pd.DataFrame(),
            "candidate_comparison_df": pd.DataFrame(
                candidate_rows
            )
        }

    return {
        "new_routes": working_routes,
        "assignment_df": pd.DataFrame(
            assigned_rows
        ),
        "candidate_comparison_df": pd.DataFrame(
            candidate_rows
        )
    }


def node_to_lonlat(node, coord_transformer):
    lon, lat = coord_transformer.transform(
        float(node[0]),
        float(node[1])
    )
    return float(lon), float(lat)


VEHICLE_COLORS = {
    1: [230, 57, 70],     # 빨강
    2: [29, 78, 216],     # 파랑
    3: [46, 160, 67],     # 초록
    4: [245, 166, 35],    # 주황
    5: [132, 70, 180],    # 보라
    6: [0, 170, 170],     # 청록
    7: [255, 105, 180],   # 분홍
    8: [139, 90, 43],     # 갈색
    9: [100, 100, 100],   # 회색
    10: [0, 120, 160],    # 진한 청록
}


def build_future_path_nodes(
    G,
    state,
    depot_node,
    delivery_nodes
):
    """현재 위치부터 남은 배송지, 물류센터까지 실제 도로 링크 경로."""
    current_node = state["current_node"]
    route = state["remaining_route"].copy()
    full_path = [current_node]

    for delivery_name in route:
        target_node = delivery_nodes[delivery_name]

        result = get_path_metrics(
            G,
            current_node,
            target_node
        )

        if len(result["path"]) > 1:
            full_path.extend(
                result["path"][1:]
            )

        current_node = target_node

    if current_node != depot_node:
        result = get_path_metrics(
            G,
            current_node,
            depot_node
        )

        if len(result["path"]) > 1:
            full_path.extend(
                result["path"][1:]
            )

    return full_path


def build_live_deck(
    G,
    current_states,
    depot_node,
    delivery_nodes,
    coord_transformer,
    deliveries_df,
    incident_markers
):
    """
    차량별 색상 + 남은 예정 경로 + 배송지 + 고장 발생 위치를 지도에 표시.
    """
    path_rows = []
    vehicle_rows = []
    delivery_rows = []
    incident_rows = []

    # 차량과 남은 경로
    for vehicle_id in sorted(current_states):
        state = current_states[vehicle_id]

        color = VEHICLE_COLORS.get(
            vehicle_id,
            [90, 90, 90]
        )

        lon, lat = node_to_lonlat(
            state["current_node"],
            coord_transformer
        )

        vehicle_rows.append({
            "vehicle_id": vehicle_id,
            "label": f"차량 {vehicle_id}",
            "lon": float(lon),
            "lat": float(lat),
            "color": (
                color
                if state["active"]
                else [80, 80, 80]
            ),
        })

        # 고장 차량은 정지 상태이므로 앞으로의 경로를 표시하지 않음
        if not state["active"]:
            continue

        try:
            path_nodes = build_future_path_nodes(
                G,
                state,
                depot_node,
                delivery_nodes
            )

            coords = []

            for node in path_nodes:
                plon, plat = node_to_lonlat(
                    node,
                    coord_transformer
                )

                if (
                    np.isfinite(plon)
                    and np.isfinite(plat)
                    and -180 <= plon <= 180
                    and -90 <= plat <= 90
                ):
                    coords.append(
                        [float(plon), float(plat)]
                    )

            if len(coords) >= 2:
                path_rows.append({
                    "vehicle_id": vehicle_id,
                    "path": coords,
                    "color": color,
                })

        except Exception:
            pass

    # 배송지 위치
    for _, row in deliveries_df.iterrows():
        try:
            lon = float(row["lon"])
            lat = float(row["lat"])

            if (
                np.isfinite(lon)
                and np.isfinite(lat)
                and -180 <= lon <= 180
                and -90 <= lat <= 90
            ):
                delivery_rows.append({
                    "name": str(row["name"]),
                    "lon": lon,
                    "lat": lat,
                    "text": "●",
                })
        except Exception:
            pass

    # 돌발상황 발생 위치
    for item in incident_markers:
        try:
            lon = float(item["lon"])
            lat = float(item["lat"])

            if (
                np.isfinite(lon)
                and np.isfinite(lat)
                and -180 <= lon <= 180
                and -90 <= lat <= 90
            ):
                incident_rows.append({
                    "lon": lon,
                    "lat": lat,
                    "text": "⚠",
                    "label": (
                        f"{item['vehicle']} 고장 "
                        f"({item['time_min']:.0f}분)"
                    ),
                })
        except Exception:
            pass

    # 지도 중심 범위
    all_lons = [r["lon"] for r in vehicle_rows]
    all_lats = [r["lat"] for r in vehicle_rows]

    all_lons += [r["lon"] for r in delivery_rows]
    all_lats += [r["lat"] for r in delivery_rows]

    if incident_rows:
        all_lons += [r["lon"] for r in incident_rows]
        all_lats += [r["lat"] for r in incident_rows]

    center_lon = float(np.mean(all_lons))
    center_lat = float(np.mean(all_lats))

    span = max(
        max(all_lons) - min(all_lons),
        max(all_lats) - min(all_lats),
        0.005
    )

    if span < 0.02:
        zoom = 12.5
    elif span < 0.05:
        zoom = 11.5
    else:
        zoom = 10.5

    layers = []

    # 차량별 예정 경로
    if path_rows:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=path_rows,
                get_path="path",
                get_color="color",
                width_min_pixels=5,
                pickable=False,
            )
        )

    # 배송지
    if delivery_rows:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=delivery_rows,
                get_position="[lon, lat]",
                get_radius=70,
                get_fill_color=[20, 20, 20, 180],
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1,
                stroked=True,
                pickable=False,
            )
        )

        layers.append(
            pdk.Layer(
                "TextLayer",
                data=delivery_rows,
                get_position="[lon, lat]",
                get_text="name",
                get_size=12,
                get_color=[30, 30, 30],
                get_pixel_offset=[0, -18],
                pickable=False,
            )
        )

    # 차량 현재 위치
    if vehicle_rows:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=vehicle_rows,
                get_position="[lon, lat]",
                get_radius=120,
                get_fill_color="color",
                get_line_color=[255, 255, 255],
                line_width_min_pixels=2,
                stroked=True,
                pickable=False,
            )
        )

        layers.append(
            pdk.Layer(
                "TextLayer",
                data=vehicle_rows,
                get_position="[lon, lat]",
                get_text="label",
                get_size=13,
                get_color=[20, 20, 20],
                get_pixel_offset=[0, 18],
                pickable=False,
            )
        )

    # 돌발상황 위치 아이콘
    if incident_rows:
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=incident_rows,
                get_position="[lon, lat]",
                get_text="text",
                get_size=32,
                get_color=[220, 20, 60],
                get_pixel_offset=[0, 0],
                pickable=False,
            )
        )

        layers.append(
            pdk.Layer(
                "TextLayer",
                data=incident_rows,
                get_position="[lon, lat]",
                get_text="label",
                get_size=13,
                get_color=[160, 0, 0],
                get_pixel_offset=[0, -26],
                pickable=False,
            )
        )

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=zoom,
            pitch=0,
        ),
        map_provider="carto",
        map_style="light",
        tooltip=False,
    )


def make_operation_summary(
    G,
    states,
    depot_node,
    delivery_nodes,
    service_time_min
):
    rows = []

    for vehicle_id, state in states.items():
        if not state["active"]:
            continue

        metrics = calculate_remaining_route(
            G,
            state,
            state["remaining_route"],
            depot_node,
            delivery_nodes,
            service_time_min
        )

        rows.append({
            "vehicle_id": vehicle_id,
            "남은배송건수": len(
                state["remaining_route"]
            ),
            "잔여거리_km": metrics["distance_km"],
            "잔여운행시간_min": metrics[
                "remaining_time_min"
            ],
        })

    return pd.DataFrame(rows)


def make_before_after_comparison(
    before_states,
    after_states,
    G,
    depot_node,
    delivery_nodes,
    service_time_min,
    unassigned_count
):
    before_df = make_operation_summary(
        G,
        before_states,
        depot_node,
        delivery_nodes,
        service_time_min
    )

    after_df = make_operation_summary(
        G,
        after_states,
        depot_node,
        delivery_nodes,
        service_time_min
    )

    def safe_sum(df, col):
        return (
            float(df[col].sum())
            if not df.empty
            else 0.0
        )

    def safe_max(df, col):
        return (
            float(df[col].max())
            if not df.empty
            else 0.0
        )

    return pd.DataFrame({
        "항목": [
            "운행 가능 차량 수",
            "운행 차량의 남은 배송건수",
            "재배차 대기 배송건수",
            "총 잔여거리(km)",
            "전체 차량 잔여운행시간 합(min)",
            "최장 잔여운행시간(min)",
        ],
        "고장 발생 직전": [
            len(before_df),
            (
                int(before_df["남은배송건수"].sum())
                if not before_df.empty
                else 0
            ),
            0,
            safe_sum(
                before_df,
                "잔여거리_km"
            ),
            safe_sum(
                before_df,
                "잔여운행시간_min"
            ),
            safe_max(
                before_df,
                "잔여운행시간_min"
            ),
        ],
        "재배차 직후": [
            len(after_df),
            (
                int(after_df["남은배송건수"].sum())
                if not after_df.empty
                else 0
            ),
            int(unassigned_count),
            safe_sum(
                after_df,
                "잔여거리_km"
            ),
            safe_sum(
                after_df,
                "잔여운행시간_min"
            ),
            safe_max(
                after_df,
                "잔여운행시간_min"
            ),
        ],
    })


def get_failed_vehicle_unresolved_count(
    vehicle_id,
    current_states,
    recovery_records
):
    """
    고장 차량의 원래 남은 배송 중
    아직 다른 차량이 완료하지 못한 건수.
    """
    records = recovery_records.get(
        vehicle_id,
        []
    )

    unresolved = 0

    for item in records:
        delivery = item["delivery"]
        assigned_vehicle = item.get(
            "assigned_vehicle"
        )

        # 재배차 실패한 배송
        if assigned_vehicle is None:
            unresolved += 1
            continue

        assigned_state = current_states.get(
            assigned_vehicle
        )

        if assigned_state is None:
            unresolved += 1
            continue

        # 새 차량의 remaining_route에서 사라지면 완료된 것
        if delivery in assigned_state["remaining_route"]:
            unresolved += 1

    return unresolved


def initialize_dynamic_simulation(
    driver_nodes,
    original_routes
):
    st.session_state.sim_running = False
    st.session_state.sim_time_min = 0.0
    st.session_state.epoch_start_min = 0.0

    st.session_state.epoch_snapshots = {
        vehicle_id: {
            "current_node": driver_nodes[vehicle_id],
            "remaining_route": route.copy(),
            "service_remaining_sec": 0.0,
            "active": True
        }
        for vehicle_id, route in original_routes.items()
    }

    st.session_state.driver_notifications = []
    st.session_state.incident_history = []
    st.session_state.last_dynamic_result = None
    st.session_state.last_before_after_comparison = None
    st.session_state.failed_recovery_records = {}
    st.session_state.incident_markers = []
    st.session_state.pending_ai_command = None
    st.session_state.pending_ai_instruction = ""
    st.session_state.last_ai_deferred_preferences = pd.DataFrame()
    st.session_state.pending_reassignments = []
    st.session_state.pending_reassignment_history = []
    st.session_state.last_candidate_comparison = pd.DataFrame()
    st.session_state.candidate_comparison_history = []
    st.session_state.all_operations_finished_notified = False


try:
    G, depot_node, driver_nodes, delivery_nodes, coord_transformer = build_network()
except Exception as e:
    st.exception(e)
    st.stop()

original_routes = make_original_routes(
    routes_before_df
)

if "epoch_snapshots" not in st.session_state:
    initialize_dynamic_simulation(
        driver_nodes,
        original_routes
    )


# ============================================================
# 9. 실시간 운영 패널
# ============================================================

run_every = (
    AUTO_REFRESH_SEC
    if st.session_state.sim_running
    else None
)


@st.fragment(run_every=run_every)
def realtime_operation_panel():

    # 운행 중이면 화면 갱신 1회마다 시뮬레이션 1분 진행
    if st.session_state.sim_running:
        st.session_state.sim_time_min += SIM_MIN_PER_TICK

    service_time_min = make_service_times(
        deliveries_df,
        st.session_state.knowledge_df
    )

    current_states = get_current_states(
        G=G,
        epoch_snapshots=st.session_state.epoch_snapshots,
        epoch_start_min=st.session_state.epoch_start_min,
        sim_time_min=st.session_state.sim_time_min,
        depot_node=depot_node,
        delivery_nodes=delivery_nodes,
        service_time_min=service_time_min
    )

    # --------------------------------------------------------
    # 재배차 대기 배송 자동 재검토
    # --------------------------------------------------------
    if (
        st.session_state.sim_running
        and st.session_state.pending_reassignments
    ):
        retry_result = retry_pending_reassignments(
            G=G,
            current_states=current_states,
            depot_node=depot_node,
            delivery_nodes=delivery_nodes,
            service_time_min=service_time_min,
            max_remaining_time_min=float(
                st.session_state.get(
                    "dynamic_max_remaining",
                    DEFAULT_MAX_ROUTE_TIME_MIN
                )
            ),
            deliveries_df=deliveries_df,
            drivers_df=drivers_df
        )

        if retry_result is not None:
            retry_candidates = retry_result.get(
                "candidate_comparison_df",
                pd.DataFrame()
            )

            if not retry_candidates.empty:
                retry_candidates = (
                    retry_candidates.copy()
                )
                retry_candidates[
                    "평가시각_min"
                ] = st.session_state.sim_time_min

                st.session_state.last_candidate_comparison = (
                    retry_candidates
                )

                st.session_state.candidate_comparison_history.append(
                    retry_candidates
                )

            if retry_result["assignment_df"].empty:
                retry_result = None

        if retry_result is not None:
            new_snapshots = {}

            for vehicle_id, state in current_states.items():
                new_route = retry_result[
                    "new_routes"
                ].get(
                    vehicle_id,
                    state["remaining_route"].copy()
                )

                new_snapshots[vehicle_id] = {
                    "current_node": state["current_node"],
                    "remaining_route": new_route,
                    "service_remaining_sec": state[
                        "service_remaining_sec"
                    ],
                    "active": state["active"]
                }

            st.session_state.epoch_snapshots = (
                new_snapshots
            )
            st.session_state.epoch_start_min = (
                st.session_state.sim_time_min
            )

            for _, row in retry_result[
                "assignment_df"
            ].iterrows():
                delivery = row["delivery"]
                vehicle_id = int(
                    row["assigned_vehicle"]
                )

                update_failed_recovery_assignment(
                    delivery,
                    vehicle_id
                )

                history_row = {
                    "발생시각_min": (
                        st.session_state.sim_time_min
                    ),
                    "배송지": delivery,
                    "배정차량": f"차량 {vehicle_id}",
                    "추가시간_min": round(
                        float(
                            row["added_time_min"]
                        ),
                        2
                    ),
                    "사유": (
                        "배송 완료로 차량 여유가 생겨 "
                        "대기 배송 자동 재배차"
                    )
                }

                st.session_state.pending_reassignment_history.append(
                    history_row
                )

                st.session_state.driver_notifications.append({
                    "발생시각_min": st.session_state.sim_time_min,
                    "차량": f"차량 {vehicle_id}",
                    "사유": "재배차 대기 배송 자동 인계",
                    "추가배송": delivery,
                    "변경경로": row["new_route"]
                })

            st.rerun()

    # --------------------------------------------------------
    # 모든 차량 운행 종료 자동 감지
    # --------------------------------------------------------
    all_operations_finished = (
        bool(current_states)
        and all(
            (
                (not state["active"])
                or (
                    state["active"]
                    and state["status"] == "운행 완료"
                )
            )
            for state in current_states.values()
        )
        and not st.session_state.pending_reassignments
    )

    if all_operations_finished:
        if st.session_state.sim_running:
            st.session_state.sim_running = False

        if not st.session_state.get(
            "all_operations_finished_notified",
            False
        ):
            st.session_state.all_operations_finished_notified = True

        st.success(
            "✅ 모든 배송 차량의 운행이 종료되었습니다. "
            "시뮬레이션을 자동으로 종료했습니다."
        )

    # --------------------------------------------------------
    # 시뮬레이션 상태
    # --------------------------------------------------------
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "현재 시뮬레이션 시간",
        f"{st.session_state.sim_time_min:.0f}분"
    )

    c2.metric(
        "운행 상태",
        "🟢 운행 중"
        if st.session_state.sim_running
        else "⏸ 대기"
    )

    c3.metric(
        "운행 가능 차량",
        f"{sum(1 for s in current_states.values() if s['active'])}대"
    )

    b1, b2, b3 = st.columns(3)

    if b1.button(
        "▶ 시뮬레이션 시작",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.sim_running,
        key="start_dynamic_sim"
    ):
        st.session_state.sim_running = True
        st.session_state.all_operations_finished_notified = False
        st.rerun()

    if b2.button(
        "⏸ 일시정지",
        use_container_width=True,
        disabled=not st.session_state.sim_running,
        key="pause_dynamic_sim"
    ):
        st.session_state.sim_running = False
        st.rerun()

    if b3.button(
        "↺ 시뮬레이션 초기화",
        use_container_width=True,
        key="reset_dynamic_sim"
    ):
        initialize_dynamic_simulation(
            driver_nodes,
            original_routes
        )
        st.rerun()

    st.caption(
        "데모에서는 실제 1초마다 시뮬레이션 시간이 1분씩 진행됩니다."
    )

    # --------------------------------------------------------
    # 현재 차량 상태
    # --------------------------------------------------------
    state_rows = []
    current_map_rows = []

    for vehicle_id in sorted(current_states):
        state = current_states[vehicle_id]
        lon, lat = node_to_lonlat(
            state["current_node"],
            coord_transformer
        )

        if state["active"]:
            remaining_count = len(
                state["remaining_route"]
            )
        else:
            remaining_count = get_failed_vehicle_unresolved_count(
                vehicle_id,
                current_states,
                st.session_state.failed_recovery_records
            )

        state_rows.append({
            "vehicle_id": vehicle_id,
            "현재상태": state["status"],
            "다음배송": (
                state["next_delivery"]
                if state["active"]
                else "-"
            ),
            "남은배송건수": remaining_count,
            "현재_X": round(float(state["current_node"][0]), 1),
            "현재_Y": round(float(state["current_node"][1]), 1),
            "운행가능": "정상" if state["active"] else "고장"
        })

        current_map_rows.append({
            "lat": lat,
            "lon": lon,
            "차량": f"차량 {vehicle_id}"
        })

    st.markdown("#### 현재 차량 운행상태")
    st.dataframe(
        pd.DataFrame(state_rows),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### 차량별 현재 위치 · 배송지 · 예정 경로")

    # 차량별 범례: 1~10번 모두 실제 지도 색상과 동일한 원으로 표시
    st.markdown(
        """
        <div style="font-size:0.92rem; color:#555; margin-bottom:0.4rem;">
          <span style="color:rgb(230,57,70);">●</span> 차량 1 &nbsp;·&nbsp;
          <span style="color:rgb(29,78,216);">●</span> 차량 2 &nbsp;·&nbsp;
          <span style="color:rgb(46,160,67);">●</span> 차량 3 &nbsp;·&nbsp;
          <span style="color:rgb(245,166,35);">●</span> 차량 4 &nbsp;·&nbsp;
          <span style="color:rgb(132,70,180);">●</span> 차량 5 &nbsp;·&nbsp;
          <span style="color:rgb(0,170,170);">●</span> 차량 6 &nbsp;·&nbsp;
          <span style="color:rgb(255,105,180);">●</span> 차량 7 &nbsp;·&nbsp;
          <span style="color:rgb(139,90,43);">●</span> 차량 8 &nbsp;·&nbsp;
          <span style="color:rgb(100,100,100);">●</span> 차량 9 &nbsp;·&nbsp;
          <span style="color:rgb(0,120,160);">●</span> 차량 10 &nbsp;·&nbsp;
          <span style="color:rgb(20,20,20);">●</span> 배송지 &nbsp;·&nbsp;
          ⚠ 고장 발생 위치
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        live_deck = build_live_deck(
            G=G,
            current_states=current_states,
            depot_node=depot_node,
            delivery_nodes=delivery_nodes,
            coord_transformer=coord_transformer,
            deliveries_df=deliveries_df,
            incident_markers=st.session_state.incident_markers
        )

        st.pydeck_chart(
            live_deck,
            use_container_width=True,
            key="dynamic_route_map"
        )

    except Exception as e:
        st.warning(
            f"차량 경로 지도 표시 중 오류: {e}"
        )

    # --------------------------------------------------------
    # 돌발상황 입력
    # --------------------------------------------------------
    st.divider()
    st.subheader("③ 돌발상황 및 실시간 재배차")


    st.markdown("#### 🤖 생성형 AI 기반 복합 운행지시 입력")

    st.caption(
        "차량 고장, 배송 완료 여부, 특정 차량 인계, 사용 가능 차량 제한, "
        "적재용량·운행시간 제약 무시 등 여러 조건을 한 문장으로 입력할 수 있습니다."
    )

    ai_instruction = st.text_area(
        "현장/운영자 지시",
        placeholder=(
            "예시 1) 차량 2가 사고로 운행 불가능. 배송지 7까지 완료했고 나머지는 자동 재배차해줘.\n"
            "예시 2) 차량 1 고장. 배송지 12는 차량 3이 맡고 나머지는 알아서 배차해줘.\n"
            "예시 3) 차량 1 운행 불가. 남은 배송은 차량 2와 3으로만 나눠서 보내.\n"
            "예시 4) 차량 1 고장. 적재용량 제약은 무시하고 차량 2와 3으로만 보내.\n"
            "예시 5) 차량 1 고장. 차량 2와 3으로만 보내고 운행시간 제한은 무시해.\n"
            "예시 6) 차량 1 운행 불가. 제약을 무시하고 차량 2와 3으로만 강제 배정해."
        ),
        height=90,
        key="ai_incident_instruction"
    )

    ai_col1, ai_col2 = st.columns(
        [1, 1]
    )

    if ai_col1.button(
        "✨ Gemini로 운행지시 해석",
        use_container_width=True,
        key="interpret_ai_incident"
    ):
        if not ai_instruction.strip():
            st.warning(
                "자연어 지시를 입력해주세요."
            )
        else:
            try:
                with st.spinner(
                    "Gemini가 운행 지시를 구조화하고 있습니다..."
                ):
                    command = (
                        interpret_incident_with_gemini(
                            instruction_text=ai_instruction,
                            current_states=current_states
                        )
                    )

                errors, warnings = (
                    validate_ai_incident_command(
                        command=command,
                        current_states=current_states,
                        delivery_nodes=delivery_nodes
                    )
                )

                st.session_state.pending_ai_command = (
                    command
                )
                st.session_state.pending_ai_instruction = (
                    ai_instruction
                )

                if errors:
                    st.session_state.pending_ai_command[
                        "_validation_errors"
                    ] = errors

                if warnings:
                    st.session_state.pending_ai_command[
                        "_validation_warnings"
                    ] = warnings

            except Exception as e:
                st.error(
                    f"Gemini 지시 해석 실패: {e}"
                )

    pending_command = st.session_state.get(
        "pending_ai_command"
    )

    if pending_command:
        st.markdown("##### AI 해석 결과")

        command_errors = pending_command.get(
            "_validation_errors",
            []
        )

        command_warnings = pending_command.get(
            "_validation_warnings",
            []
        )

        failed_id = pending_command.get(
            "vehicle_id"
        )

        through = pending_command.get(
            "completed_through_delivery"
        )

        fixed_all = pending_command.get(
            "fixed_vehicle_for_all_remaining"
        )

        fixed_items = pending_command.get(
            "fixed_reassignments",
            []
        )

        allowed_vehicles = pending_command.get(
            "allowed_vehicles",
            []
        )

        force_assignment = pending_command.get(
            "force_assignment",
            False
        )

        ignore_capacity = pending_command.get(
            "ignore_capacity",
            False
        )

        ignore_time_limit = pending_command.get(
            "ignore_time_limit",
            False
        )

        summary_rows = [
            {
                "항목": "돌발상황 유형",
                "AI 해석": (
                    "차량 고장·사고로 운행 불가"
                    if pending_command.get(
                        "event_type"
                    ) == "vehicle_failure"
                    else pending_command.get(
                        "event_type"
                    )
                )
            },
            {
                "항목": "운행 중단 차량",
                "AI 해석": (
                    f"차량 {failed_id}"
                    if failed_id is not None
                    else "-"
                )
            },
            {
                "항목": "배송 완료 기준",
                "AI 해석": through or "-"
            },
            {
                "항목": "남은 배송 우선 인계 차량",
                "AI 해석": (
                    f"차량 {fixed_all}"
                    if fixed_all is not None
                    else "-"
                )
            },
            {
                "항목": "특정 배송 지정 차량",
                "AI 해석": (
                    ", ".join(
                        f"{x['delivery']} → 차량 {x['vehicle_id']}"
                        for x in fixed_items
                    )
                    if fixed_items
                    else "-"
                )
            },
            {
                "항목": "재배차 허용 차량",
                "AI 해석": (
                    ", ".join(
                        f"차량 {v}"
                        for v in allowed_vehicles
                    )
                    if allowed_vehicles
                    else "모든 운행 가능 차량"
                )
            },
            {
                "항목": "재배차 실행 방식",
                "AI 해석": (
                    "⚠ 운영자 지시 우선(제약 예외 허용)"
                    if force_assignment
                    else "기본 최적화 제약 준수"
                )
            },
            {
                "항목": "예외 적용 제약",
                "AI 해석": (
                    ", ".join(
                        [
                            name
                            for flag, name in [
                                (
                                    ignore_capacity,
                                    "적재용량"
                                ),
                                (
                                    ignore_time_limit,
                                    "운행시간"
                                )
                            ]
                            if flag
                        ]
                    )
                    if (
                        ignore_capacity
                        or ignore_time_limit
                    )
                    else (
                        "적재용량·최대 운행시간 모두"
                        if force_assignment
                        else "-"
                    )
                )
            },
            {
                "항목": "기타 남은 배송 처리",
                "AI 해석": (
                    "자동 재배차"
                    if pending_command.get(
                        "auto_reassign_remaining"
                    )
                    else "자동 재배차 미실행"
                )
            }
        ]

        st.dataframe(
            pd.DataFrame(
                summary_rows
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Gemini 요약: "
            f"{pending_command.get('summary', '-')}"
        )

        if force_assignment:
            st.error(
                "⚠ 운영자 지시 우선 모드가 요청되었습니다. "
                "지정 차량으로 인계하기 위해 적재용량 또는 "
                "최대 운행시간을 초과할 수 있습니다. "
                "아래 AI 해석 결과와 제약 초과 여부를 확인한 뒤 실행하세요."
            )

        for warning in command_warnings:
            st.warning(warning)

        for error in command_errors:
            st.error(error)

        if ai_col2.button(
            "✅ 해석 결과 확인 후 재배차 실행",
            type="primary",
            use_container_width=True,
            disabled=bool(command_errors),
            key="execute_ai_incident"
        ):
            failed_vehicle_ai = int(
                pending_command[
                    "vehicle_id"
                ]
            )

            max_remaining_time_ai = float(
                st.session_state.get(
                    "dynamic_max_remaining",
                    DEFAULT_MAX_ROUTE_TIME_MIN
                )
            )

            before_states = {
                vehicle_id: {
                    "current_node": state["current_node"],
                    "remaining_route": state["remaining_route"].copy(),
                    "service_remaining_sec": state["service_remaining_sec"],
                    "status": state["status"],
                    "next_delivery": state["next_delivery"],
                    "active": state["active"]
                }
                for vehicle_id, state in current_states.items()
            }

            result = execute_ai_failure_command(
                command=pending_command,
                current_states=current_states,
                G=G,
                depot_node=depot_node,
                delivery_nodes=delivery_nodes,
                service_time_min=service_time_min,
                max_remaining_time_min=max_remaining_time_ai,
                deliveries_df=deliveries_df,
                drivers_df=drivers_df
            )

            adjusted_failed_state = result[
                "adjusted_failed_state"
            ]

            incident_lon, incident_lat = node_to_lonlat(
                adjusted_failed_state[
                    "current_node"
                ],
                coord_transformer
            )

            st.session_state.incident_markers.append({
                "vehicle": f"차량 {failed_vehicle_ai}",
                "time_min": st.session_state.sim_time_min,
                "lon": incident_lon,
                "lat": incident_lat,
            })

            new_snapshots = {}

            for vehicle_id, state in current_states.items():
                if vehicle_id == failed_vehicle_ai:
                    new_snapshots[
                        vehicle_id
                    ] = {
                        "current_node": adjusted_failed_state[
                            "current_node"
                        ],
                        "remaining_route": adjusted_failed_state[
                            "remaining_route"
                        ].copy(),
                        "service_remaining_sec": adjusted_failed_state[
                            "service_remaining_sec"
                        ],
                        "active": False
                    }
                    continue

                new_route = result[
                    "new_routes"
                ].get(
                    vehicle_id,
                    state[
                        "remaining_route"
                    ].copy()
                )

                new_snapshots[
                    vehicle_id
                ] = {
                    "current_node": state["current_node"],
                    "remaining_route": new_route,
                    "service_remaining_sec": state[
                        "service_remaining_sec"
                    ],
                    "active": state["active"]
                }

            st.session_state.epoch_snapshots = (
                new_snapshots
            )

            st.session_state.epoch_start_min = (
                st.session_state.sim_time_min
            )

            st.session_state.last_dynamic_result = (
                result
            )

            if "candidate_comparison_df" in result:
                st.session_state.last_candidate_comparison = (
                    result[
                        "candidate_comparison_df"
                    ].copy()
                )

            # 운영자 지정 차량이 capacity/시간 제약을 만족하지 못한 경우
            # 자동 대체배정 대상으로 넘긴 내역을 별도로 저장
            st.session_state.last_ai_deferred_preferences = (
                result.get(
                    "deferred_preference_df",
                    pd.DataFrame()
                )
            )

            after_states = {
                vehicle_id: {
                    "current_node": snapshot[
                        "current_node"
                    ],
                    "remaining_route": snapshot[
                        "remaining_route"
                    ].copy(),
                    "service_remaining_sec": snapshot[
                        "service_remaining_sec"
                    ],
                    "status": (
                        "운행 중단"
                        if not snapshot[
                            "active"
                        ]
                        else current_states[
                            vehicle_id
                        ]["status"]
                    ),
                    "next_delivery": (
                        "-"
                        if not snapshot[
                            "active"
                        ]
                        else (
                            snapshot[
                                "remaining_route"
                            ][0]
                            if snapshot[
                                "remaining_route"
                            ]
                            else "-"
                        )
                    ),
                    "active": snapshot[
                        "active"
                    ]
                }
                for vehicle_id, snapshot in new_snapshots.items()
            }

            st.session_state.last_before_after_comparison = (
                make_before_after_comparison(
                    before_states=before_states,
                    after_states=after_states,
                    G=G,
                    depot_node=depot_node,
                    delivery_nodes=delivery_nodes,
                    service_time_min=service_time_min,
                    unassigned_count=len(
                        result[
                            "unassigned_df"
                        ]
                    )
                )
            )

            recovery_records = []

            if not result[
                "assignment_df"
            ].empty:
                for _, row in result[
                    "assignment_df"
                ].iterrows():
                    recovery_records.append(
                        {
                            "delivery": row[
                                "delivery"
                            ],
                            "assigned_vehicle": int(
                                row[
                                    "assigned_vehicle"
                                ]
                            )
                        }
                    )

                    st.session_state.driver_notifications.append({
                        "발생시각_min": st.session_state.sim_time_min,
                        "차량": (
                            f"차량 {int(row['assigned_vehicle'])}"
                        ),
                        "선택·제외 사유": (
                            (
                                "AI 복합지시 / 운영자 강제 배정 / "
                                if result.get(
                                    "force_mode",
                                    False
                                )
                                else "AI 복합지시 / "
                            )
                            + f"차량 {failed_vehicle_ai} 운행 중단"
                        ),
                        "추가배송": row[
                            "delivery"
                        ],
                        "변경경로": row[
                            "new_route"
                        ]
                    })

            if not result[
                "unassigned_df"
            ].empty:
                for _, row in result[
                    "unassigned_df"
                ].iterrows():
                    recovery_records.append(
                        {
                            "delivery": row[
                                "delivery"
                            ],
                            "assigned_vehicle": None
                        }
                    )

            st.session_state.failed_recovery_records[
                failed_vehicle_ai
            ] = recovery_records

            # 즉시 배차하지 못한 배송은 최종 미배정이 아니라 대기 큐에 저장
            add_to_pending_queue(
                unassigned_df=result["unassigned_df"],
                source_vehicle=failed_vehicle_ai,
                sim_time_min=st.session_state.sim_time_min
            )

            st.session_state.incident_history.append({
                "발생시각_min": st.session_state.sim_time_min,
                "고장차량": f"차량 {failed_vehicle_ai}",
                "재배차대상": (
                    ", ".join(
                        result[
                            "failed_deliveries"
                        ]
                    )
                    if result[
                        "failed_deliveries"
                    ]
                    else "-"
                ),
                "입력방식": "Gemini 자연어 지시"
            })

            st.session_state.pending_ai_command = None
            st.session_state.pending_ai_instruction = ""

            st.rerun()

    st.markdown("#### ⚡ 빠른 돌발상황 등록")

    active_vehicle_ids = [
        vehicle_id
        for vehicle_id, state in current_states.items()
        if state["active"]
    ]

    if active_vehicle_ids:
        c1, c2 = st.columns(2)

        failed_vehicle = c1.selectbox(
            "고장 차량",
            active_vehicle_ids,
            format_func=lambda x: f"차량 {x}",
            key="dynamic_failed_vehicle"
        )

        max_remaining_time_min = c2.number_input(
            "재배차 후 최대 허용 남은 운행시간(분)",
            min_value=15,
            max_value=240,
            value=int(DEFAULT_MAX_ROUTE_TIME_MIN),
            step=5,
            key="dynamic_max_remaining"
        )

        failed_state = current_states[failed_vehicle]

        st.info(
            f"현재 차량 {failed_vehicle}: "
            f"{failed_state['status']} / "
            f"남은 배송 {len(failed_state['remaining_route'])}건"
        )

        if st.button(
            "🚨 현재 시점에 차량 고장 등록",
            type="primary",
            use_container_width=True,
            key="dynamic_failure_button"
        ):
            before_states = {
                vehicle_id: {
                    "current_node": state["current_node"],
                    "remaining_route": state["remaining_route"].copy(),
                    "service_remaining_sec": state["service_remaining_sec"],
                    "status": state["status"],
                    "next_delivery": state["next_delivery"],
                    "active": state["active"]
                }
                for vehicle_id, state in current_states.items()
            }

            # 발생 위치를 지도에 기록
            incident_lon, incident_lat = node_to_lonlat(
                current_states[failed_vehicle]["current_node"],
                coord_transformer
            )

            st.session_state.incident_markers.append({
                "vehicle": f"차량 {failed_vehicle}",
                "time_min": st.session_state.sim_time_min,
                "lon": incident_lon,
                "lat": incident_lat,
            })

            result = reassign_failed_vehicle_dynamic(
                G=G,
                failed_vehicle=failed_vehicle,
                current_states=current_states,
                depot_node=depot_node,
                delivery_nodes=delivery_nodes,
                service_time_min=service_time_min,
                max_remaining_time_min=float(max_remaining_time_min),
                deliveries_df=deliveries_df,
                drivers_df=drivers_df
            )

            # 현재 위치를 새로운 출발점으로 고정
            new_snapshots = {}

            for vehicle_id, state in current_states.items():

                if vehicle_id == failed_vehicle:
                    new_snapshots[vehicle_id] = {
                        "current_node": state["current_node"],
                        "remaining_route": state["remaining_route"].copy(),
                        "service_remaining_sec": state["service_remaining_sec"],
                        "active": False
                    }
                    continue

                new_route = result["new_routes"].get(
                    vehicle_id,
                    state["remaining_route"].copy()
                )

                new_snapshots[vehicle_id] = {
                    "current_node": state["current_node"],
                    "remaining_route": new_route,
                    "service_remaining_sec": state["service_remaining_sec"],
                    "active": state["active"]
                }

            st.session_state.epoch_snapshots = new_snapshots
            st.session_state.epoch_start_min = (
                st.session_state.sim_time_min
            )

            st.session_state.last_dynamic_result = result

            if "candidate_comparison_df" in result:
                st.session_state.last_candidate_comparison = (
                    result[
                        "candidate_comparison_df"
                    ].copy()
                )

            after_states = {
                vehicle_id: {
                    "current_node": snapshot["current_node"],
                    "remaining_route": snapshot["remaining_route"].copy(),
                    "service_remaining_sec": snapshot["service_remaining_sec"],
                    "status": (
                        "운행 중단"
                        if not snapshot["active"]
                        else current_states[vehicle_id]["status"]
                    ),
                    "next_delivery": (
                        "-"
                        if not snapshot["active"]
                        else (
                            snapshot["remaining_route"][0]
                            if snapshot["remaining_route"]
                            else "-"
                        )
                    ),
                    "active": snapshot["active"]
                }
                for vehicle_id, snapshot in new_snapshots.items()
            }

            st.session_state.last_before_after_comparison = (
                make_before_after_comparison(
                    before_states=before_states,
                    after_states=after_states,
                    G=G,
                    depot_node=depot_node,
                    delivery_nodes=delivery_nodes,
                    service_time_min=service_time_min,
                    unassigned_count=len(
                        result["unassigned_df"]
                    )
                )
            )

            recovery_records = []

            if not result["assignment_df"].empty:
                for _, row in result["assignment_df"].iterrows():
                    recovery_records.append({
                        "delivery": row["delivery"],
                        "assigned_vehicle": int(
                            row["assigned_vehicle"]
                        )
                    })

            if not result["unassigned_df"].empty:
                for _, row in result["unassigned_df"].iterrows():
                    recovery_records.append({
                        "delivery": row["delivery"],
                        "assigned_vehicle": None
                    })

            st.session_state.failed_recovery_records[
                failed_vehicle
            ] = recovery_records

            # 즉시 배차하지 못한 배송은 최종 미배정이 아니라 대기 큐에 저장
            add_to_pending_queue(
                unassigned_df=result["unassigned_df"],
                source_vehicle=failed_vehicle,
                sim_time_min=st.session_state.sim_time_min
            )

            # 기사 전달 메시지 생성
            if not result["assignment_df"].empty:
                for _, row in result["assignment_df"].iterrows():
                    vehicle_id = int(row["assigned_vehicle"])

                    st.session_state.driver_notifications.append({
                        "발생시각_min": st.session_state.sim_time_min,
                        "차량": f"차량 {vehicle_id}",
                        "사유": f"차량 {failed_vehicle} 고장",
                        "추가배송": row["delivery"],
                        "변경경로": row["new_route"]
                    })

            st.session_state.incident_history.append({
                "발생시각_min": st.session_state.sim_time_min,
                "고장차량": f"차량 {failed_vehicle}",
                "재배차대상": ", ".join(result["failed_deliveries"])
                if result["failed_deliveries"]
                else "-"
            })

            # 중요: 시뮬레이션은 멈추지 않음
            st.rerun()

    else:
        st.warning("현재 운행 가능한 차량이 없습니다.")

    # --------------------------------------------------------
    # 최근 재배차 결과
    # --------------------------------------------------------
    result = st.session_state.last_dynamic_result

    if result is not None:
        st.markdown("#### 최근 재배차 결과")

        assignment_df = result["assignment_df"]
        unassigned_df = result["unassigned_df"]

        if not assignment_df.empty:
            show_df = assignment_df.copy()

            for col in [
                "added_distance_km",
                "added_time_min",
                "final_remaining_time_min"
            ]:
                show_df[col] = show_df[col].round(2)

            st.success(
                f"{len(assignment_df)}건의 배송이 현재 운행 중인 차량에 재배차되었습니다."
            )

            st.dataframe(
                show_df,
                use_container_width=True,
                hide_index=True
            )

        if not unassigned_df.empty:
            st.warning(
                f"{len(unassigned_df)}건은 현재 즉시 재배차가 어려워 "
                "재배차 대기 상태로 전환되었습니다."
            )

            st.dataframe(
                unassigned_df.round(2),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # 재배차 후보 비교
    # --------------------------------------------------------
    candidate_df = st.session_state.get(
        "last_candidate_comparison",
        pd.DataFrame()
    )

    if (
        isinstance(
            candidate_df,
            pd.DataFrame
        )
        and not candidate_df.empty
    ):
        st.markdown(
            "#### 🔎 실시간 재배차 후보 비교"
        )

        st.caption(
            "각 배송을 어느 차량에 넣을 수 있었는지와 "
            "선택·제외 사유를 비교합니다."
        )

        show_candidate = (
            candidate_df.copy()
        )

        rename_map = {
            "delivery": "배송지",
            "vehicle_id": "후보 차량",
            "capacity": "최대 적재용량",
            "current_demand": "현재 잔여 배송물량",
            "delivery_demand": "추가 인계 물량",
            "candidate_demand": "인계 후 총 물량",
            "added_time_min": "추가 운행시간(분)",
            "added_distance_km": "추가 이동거리(km)",
            "final_remaining_time_min": "인계 후 잔여운행시간(분)",
            "status": "검토 결과",
            "reason": "선택·제외 사유",
            "operator_preferred": "운영자 지정 여부"
        }

        show_candidate = (
            show_candidate.rename(
                columns=rename_map
            )
        )

        if "후보 차량" in show_candidate.columns:
            show_candidate[
                "후보 차량"
            ] = show_candidate[
                "후보 차량"
            ].apply(
                lambda x: (
                    f"차량 {int(x)}"
                    if pd.notna(x)
                    else "-"
                )
            )

        for col in [
            "추가 운행시간(분)",
            "추가 이동거리(km)",
            "인계 후 잔여운행시간(분)"
        ]:
            if col in show_candidate.columns:
                show_candidate[col] = (
                    pd.to_numeric(
                        show_candidate[col],
                        errors="coerce"
                    ).round(2)
                )

        preferred_columns = [
            "배송지",
            "후보 차량",
            "최대 적재용량",
            "현재 잔여 배송물량",
            "추가 인계 물량",
            "인계 후 총 물량",
            "추가 운행시간(분)",
            "추가 이동거리(km)",
            "인계 후 잔여운행시간(분)",
            "운영자 지정 여부",
            "검토 결과",
            "선택·제외 사유"
        ]

        visible_columns = [
            col
            for col in preferred_columns
            if col in show_candidate.columns
        ]

        st.dataframe(
            show_candidate[
                visible_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # 재배차 대기 현황
    # --------------------------------------------------------
    if st.session_state.pending_reassignments:
        st.markdown("#### ⏳ 재배차 대기 배송 현황")

        pending_df = pd.DataFrame(
            st.session_state.pending_reassignments
        ).rename(
            columns={
                "delivery": "배송지",
                "source_vehicle": "고장 차량",
                "requested_vehicle": "운영자 요청 차량",
                "created_min": "대기 시작(min)",
                "reason": "대기 사유"
            }
        )

        if "고장 차량" in pending_df.columns:
            pending_df["고장 차량"] = pending_df[
                "고장 차량"
            ].apply(
                lambda x: f"차량 {int(x)}"
            )

        if "운영자 요청 차량" in pending_df.columns:
            pending_df["운영자 요청 차량"] = pending_df[
                "운영자 요청 차량"
            ].apply(
                lambda x: (
                    f"차량 {int(x)}"
                    if pd.notna(x)
                    else "-"
                )
            )

        st.info(
            "현재는 적재용량 또는 잔여 운행시간 때문에 바로 인계할 수 없지만, "
            "다른 차량이 배송을 완료해 여유가 생기면 매 시뮬레이션 시점마다 "
            "자동으로 다시 재배차를 시도합니다."
        )

        st.dataframe(
            pending_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        if st.session_state.pending_reassignment_history:
            st.success(
                "현재 재배차 대기 중인 배송이 없습니다."
            )

    if st.session_state.pending_reassignment_history:
        with st.expander(
            "대기 후 자동 재배차 이력"
        ):
            st.dataframe(
                pd.DataFrame(
                    st.session_state.pending_reassignment_history
                ),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # 고장 발생 전후 비교
    # --------------------------------------------------------
    if st.session_state.last_before_after_comparison is not None:
        st.markdown("#### 고장 발생 전후 운영 비교")

        comparison_show = (
            st.session_state.last_before_after_comparison.copy()
        )

        for col in [
            "고장 발생 직전",
            "재배차 직후"
        ]:
            comparison_show[col] = comparison_show[col].apply(
                lambda x: round(x, 2)
                if isinstance(x, (float, np.floating))
                else x
            )

        st.dataframe(
            comparison_show,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # 기사 전달 알림
    # --------------------------------------------------------

    if (
        "last_ai_deferred_preferences" in st.session_state
        and isinstance(
            st.session_state.last_ai_deferred_preferences,
            pd.DataFrame
        )
        and not st.session_state.last_ai_deferred_preferences.empty
    ):
        st.markdown(
            "#### ⚠ 운영자 지정 조건 대체 처리"
        )

        deferred_view = (
            st.session_state.last_ai_deferred_preferences
            .rename(
                columns={
                    "delivery": "배송지",
                    "preferred_vehicle": "요청 차량",
                    "reason": "원래 지시 미반영 사유",
                }
            )
            .copy()
        )

        deferred_view[
            "요청 차량"
        ] = deferred_view[
            "요청 차량"
        ].apply(
            lambda x: (
                f"차량 {int(x)}"
                if pd.notna(x)
                else "-"
            )
        )

        st.caption(
            "지정 차량의 적재용량 또는 최대 운행시간을 초과한 배송은 "
            "즉시 배차가 어렵다면 재배차 대기로 전환하고, 이후 차량 여유가 생기면 다시 자동 재배차합니다."
        )

        st.dataframe(
            deferred_view,
            use_container_width=True,
            hide_index=True
        )

    if st.session_state.driver_notifications:
        st.markdown("#### 📨 배송 차량 경로변경 알림")

        notifications_df = pd.DataFrame(
            st.session_state.driver_notifications
        )

        st.dataframe(
            notifications_df,
            use_container_width=True,
            hide_index=True
        )

        latest = st.session_state.driver_notifications[-1]

        latest_reason = latest.get(
            "사유",
            latest.get(
                "선택·제외 사유",
                "-"
            )
        )

        st.success(
            f"**{latest['차량']} 기사 알림**  \n"
            f"사유: {latest_reason}  \n"
            f"추가 배송: {latest['추가배송']}  \n"
            f"변경 경로: {latest['변경경로']}"
        )


realtime_operation_panel()


# ============================================================
# 10. 돌발상황 이력
# ============================================================

if st.session_state.incident_history:
    st.divider()
    st.subheader("④ 돌발상황 이력")

    st.dataframe(
        pd.DataFrame(st.session_state.incident_history),
        use_container_width=True,
        hide_index=True
    )


st.caption(
    "※ 차량 위치와 색상 경로는 실제 GPS가 아니라 SHP 도로 네트워크, "
    "계획 경로 및 링크 이동시간을 이용해 계산한 MVP 시뮬레이션 값입니다. "
    "고장 차량의 남은 배송건수는 재배차된 배송이 실제 완료될 때마다 감소합니다. "
    "즉시 수용할 차량이 없는 배송은 재배차 대기 상태로 두고, 이후 다른 차량의 "
    "배송 완료로 적재·운행 여유가 생기면 자동으로 다시 배차를 시도합니다."
)

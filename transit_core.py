from __future__ import annotations

import heapq
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


BASE_DIR = Path(__file__).resolve().parent
AVG_DISTANCE_PER_SEGMENT = 1.3


FareStrategy = Callable[[float, str], int]


def krt_fare_strategy(distance_km: float, line_type: str) -> int:
    fare = 20 + (math.ceil((distance_km - 5) / 2) * 5) if distance_km > 5 else 20
    max_fare = 35 if "C" in line_type or "LRT" in line_type else 60
    return min(fare, max_fare)


def tpi_fare_strategy(distance_km: float, line_type: str) -> int:
    fare = 20 + (math.ceil((distance_km - 5) / 3) * 5) if distance_km > 5 else 20
    return min(fare, 65)


@dataclass(frozen=True)
class MapConfig:
    key: str
    display_name: str
    data_file: str
    image_file: str
    fare_file: str
    fare_strategy: FareStrategy


MAP_DATABASE: dict[str, MapConfig] = {
    "高雄捷運": MapConfig(
        key="krt",
        display_name="高雄捷運",
        data_file="krt_data.json",
        image_file="krt_map.jpg",
        fare_file="krt_real_fare.json",
        fare_strategy=krt_fare_strategy,
    ),
    "台北捷運": MapConfig(
        key="tpi",
        display_name="台北捷運",
        data_file="tpi_data.json",
        image_file="TaipeiMetroStamp.png",
        fare_file="tpi_real_fare.json",
        fare_strategy=tpi_fare_strategy,
    ),
}


@dataclass(frozen=True)
class Station:
    sid: str
    name: str
    coords: tuple[float, float]
    line_type: str
    neighbors: tuple[str, ...]

    @property
    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class FareResult:
    amount: int
    details: str
    source: str


@dataclass(frozen=True)
class StationParseResult:
    start_id: str | None
    end_id: str | None
    message: str
    source: str

    @property
    def ok(self) -> bool:
        return bool(self.start_id and self.end_id)


class TransitSystem:
    def __init__(
        self,
        stations: dict[str, Station],
        fare_matrix: dict[str, dict[str, int]],
        fare_strategy: FareStrategy,
    ) -> None:
        self.stations = stations
        self.fare_matrix = fare_matrix
        self.fare_strategy = fare_strategy
        self._name_to_sid = self._build_name_index()
        self._alias_index = self._build_alias_index()

    @classmethod
    def from_config(cls, config: MapConfig) -> "TransitSystem":
        data = load_json_data(config.data_file)
        fare_matrix = load_json_data(config.fare_file)
        stations = {
            sid: Station(
                sid=sid,
                name=str(info["name"]),
                coords=tuple(info.get("coords", (0, 0))),
                line_type=str(info.get("line_type", "")),
                neighbors=tuple(info.get("neighbors", ())),
            )
            for sid, info in data.items()
        }
        return cls(stations, fare_matrix, config.fare_strategy)

    def get_station(self, sid: str | None) -> Station | None:
        if sid is None:
            return None
        return self.stations.get(sid)

    def get_all_display_names(self) -> list[str]:
        return sorted({station.display_name for station in self.stations.values()})

    def get_sid_by_name(self, display_name: str) -> str | None:
        return self._name_to_sid.get(display_name)

    def station_catalog_for_prompt(self) -> str:
        return "、".join(f"{s.name}({s.sid})" for s in self.stations.values())

    def nearest_station(
        self,
        click_x: float,
        click_y: float,
        rendered_width: float,
        original_width: float,
        threshold_px: float = 130,
    ) -> Station | None:
        if not self.stations or original_width <= 0:
            return None

        scale_ratio = rendered_width / original_width
        closest: Station | None = None
        min_distance = float("inf")

        for station in self.stations.values():
            sx = station.coords[0] * scale_ratio
            sy = station.coords[1] * scale_ratio
            distance = math.hypot(click_x - sx, click_y - sy)
            if distance < min_distance:
                closest = station
                min_distance = distance

        return closest if min_distance < threshold_px * scale_ratio else None

    def data_health(self) -> dict[str, object]:
        missing_neighbors: list[tuple[str, str]] = []
        edge_count = 0
        transfer_like_nodes = 0

        for sid, station in self.stations.items():
            edge_count += len(station.neighbors)
            if len(station.neighbors) >= 3:
                transfer_like_nodes += 1
            for neighbor in station.neighbors:
                if neighbor not in self.stations:
                    missing_neighbors.append((sid, neighbor))

        fare_pairs = sum(len(row) for row in self.fare_matrix.values() if isinstance(row, dict))
        fare_values = [
            int(value)
            for row in self.fare_matrix.values()
            if isinstance(row, dict)
            for value in row.values()
            if isinstance(value, int | float) or str(value).isdigit()
        ]

        return {
            "station_count": len(self.stations),
            "edge_count": edge_count // 2,
            "fare_pairs": fare_pairs,
            "transfer_like_nodes": transfer_like_nodes,
            "missing_neighbors": missing_neighbors,
            "min_fare": min(fare_values) if fare_values else None,
            "max_fare": max(fare_values) if fare_values else None,
        }

    def _build_name_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for sid, station in self.stations.items():
            index.setdefault(station.display_name, sid)
        return index

    def _build_alias_index(self) -> list[tuple[str, str]]:
        aliases: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for station in self.stations.values():
            candidates = station_aliases(station.name)
            for alias in candidates:
                normalized = normalize_text(alias)
                if len(normalized) < 2:
                    continue
                key = (normalized, station.sid)
                if key not in seen:
                    seen.add(key)
                    aliases.append((normalized, station.sid))

        aliases.sort(key=lambda item: len(item[0]), reverse=True)
        return aliases

    def find_station_mentions(self, user_text: str) -> list[tuple[int, str]]:
        text = normalize_text(user_text)
        mentions: list[tuple[int, str]] = []
        matched_sids: set[str] = set()

        for alias, sid in self._alias_index:
            position = text.find(alias)
            if position >= 0 and sid not in matched_sids:
                mentions.append((position, sid))
                matched_sids.add(sid)

        mentions.sort(key=lambda item: item[0])
        return mentions


def load_json_data(filepath: str | Path) -> dict:
    path = Path(filepath)
    if not path.is_absolute():
        path = BASE_DIR / path
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_map_config(display_name: str) -> MapConfig:
    return MAP_DATABASE[display_name]


def normalize_text(value: str) -> str:
    text = value.lower().replace("臺", "台")
    text = re.sub(r"[\s　,，。.!！?？:：;；/／\\\-_()（）\[\]【】「」『』]", "", text)
    return text


def station_aliases(station_name: str) -> set[str]:
    aliases = {station_name}
    aliases.add(station_name.replace("(", "").replace(")", ""))
    aliases.add(station_name.replace("（", "").replace("）", ""))

    main_name = re.sub(r"[（(].*?[）)]", "", station_name).strip()
    if main_name:
        aliases.add(main_name)

    for content in re.findall(r"[（(](.*?)[）)]", station_name):
        content = content.strip()
        if content:
            aliases.add(content)
            aliases.add(f"{content}站")
            aliases.add(f"{main_name}{content}")
            aliases.add(f"{main_name}{content}站")

    if "高鐵" in station_name:
        aliases.update({"高鐵", "高鐵站", "左營高鐵", "左營高鐵站"})

    return aliases


def parse_ai_json(text: str) -> tuple[str | None, str | None]:
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not match:
        return None, None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None
    return data.get("start_id"), data.get("end_id")


def parse_stations_locally(user_text: str, system: TransitSystem) -> StationParseResult:
    if not user_text.strip():
        return StationParseResult(None, None, "尚未輸入查詢文字。", "local")

    mentions = system.find_station_mentions(user_text)
    if len(mentions) < 2:
        return StationParseResult(None, None, "本地文字比對找不到兩個有效站點。", "local")

    start_id = mentions[0][1]
    end_id = mentions[-1][1]
    return StationParseResult(start_id, end_id, "已由本地站名/別名比對解析。", "local")


def find_shortest_path(system: TransitSystem, start_id: str | None, end_id: str | None) -> list[str]:
    if not start_id or not end_id or start_id not in system.stations or end_id not in system.stations:
        return []

    queue: deque[list[str]] = deque([[start_id]])
    visited = {start_id}

    while queue:
        path = queue.popleft()
        current_id = path[-1]
        if current_id == end_id:
            return path

        current_station = system.get_station(current_id)
        if current_station is None:
            continue

        for neighbor_id in current_station.neighbors:
            if neighbor_id not in visited and neighbor_id in system.stations:
                visited.add(neighbor_id)
                queue.append([*path, neighbor_id])

    return []


def calculate_fare_fallback(system: TransitSystem, path_ids: Iterable[str]) -> int:
    path = list(path_ids)
    if len(path) < 2:
        return 0

    total_fare = 0
    current_line = system.get_station(path[0]).line_type
    segment_edges = 0

    for previous_id, current_id in zip(path, path[1:]):
        previous_station = system.get_station(previous_id)
        current_station = system.get_station(current_id)
        if previous_station is None or current_station is None:
            continue

        if current_station.line_type != current_line:
            total_fare += _fare_for_segment(system, segment_edges, current_line)
            current_line = current_station.line_type
            segment_edges = 0
        else:
            segment_edges += 1

    total_fare += _fare_for_segment(system, segment_edges, current_line)
    return total_fare


def find_cheapest_path(system: TransitSystem, start_id: str | None, end_id: str | None) -> list[str]:
    if not start_id or not end_id or start_id not in system.stations or end_id not in system.stations:
        return []

    start_line = system.get_station(start_id).line_type
    priority_queue: list[tuple[int, int, str, str, list[str]]] = [(0, 1, start_id, start_line, [start_id])]
    best_cost: dict[tuple[str, str], tuple[int, int]] = {(start_id, start_line): (0, 1)}

    while priority_queue:
        current_fare, station_count, current_id, current_line, path = heapq.heappop(priority_queue)
        if current_id == end_id:
            return path

        best_fare, best_count = best_cost.get((current_id, current_line), (float("inf"), float("inf")))
        if current_fare > best_fare or (current_fare == best_fare and station_count > best_count):
            continue

        current_station = system.get_station(current_id)
        if current_station is None:
            continue

        for neighbor_id in current_station.neighbors:
            if neighbor_id in path or neighbor_id not in system.stations:
                continue

            new_path = [*path, neighbor_id]
            neighbor_line = system.get_station(neighbor_id).line_type
            new_fare = calculate_fare_fallback(system, new_path)
            new_count = len(new_path)
            key = (neighbor_id, neighbor_line)
            best_neighbor = best_cost.get(key, (float("inf"), float("inf")))

            if new_fare < best_neighbor[0] or (new_fare == best_neighbor[0] and new_count < best_neighbor[1]):
                best_cost[key] = (new_fare, new_count)
                heapq.heappush(priority_queue, (new_fare, new_count, neighbor_id, neighbor_line, new_path))

    return []


def get_fare_and_details(system: TransitSystem, path_ids: list[str]) -> FareResult:
    if len(path_ids) < 2:
        return FareResult(0, "無需搭乘", "none")

    start_id = path_ids[0]
    end_id = path_ids[-1]
    matrix_value = system.fare_matrix.get(start_id, {}).get(end_id)
    fare_from_matrix = _safe_int(matrix_value)

    if fare_from_matrix and fare_from_matrix > 0:
        amount = fare_from_matrix
        source = "官方票價矩陣"
    else:
        amount = calculate_fare_fallback(system, path_ids)
        source = "系統公式估算"

    details = summarize_path_segments(system, path_ids)
    details.append(f"\n總金額：{amount} 元（{source}）")
    return FareResult(amount, "\n".join(details), source)


def summarize_path_segments(system: TransitSystem, path_ids: list[str]) -> list[str]:
    if not path_ids:
        return []

    segments: list[str] = []
    current_line = system.get_station(path_ids[0]).line_type
    segment_start = system.get_station(path_ids[0]).name

    for index in range(1, len(path_ids)):
        current_station = system.get_station(path_ids[index])
        previous_station = system.get_station(path_ids[index - 1])
        if current_station is None or previous_station is None:
            continue

        if current_station.line_type != current_line:
            segments.append(f"- {current_line} 線：{segment_start} -> {previous_station.name}")
            segment_start = previous_station.name
            current_line = current_station.line_type

    end_station = system.get_station(path_ids[-1])
    if end_station is not None:
        segments.append(f"- {current_line} 線：{segment_start} -> {end_station.name}")
    return segments


def _fare_for_segment(system: TransitSystem, segment_edges: int, line_type: str) -> int:
    if segment_edges <= 0:
        return 0
    distance = segment_edges * AVG_DISTANCE_PER_SEGMENT
    return system.fare_strategy(distance, line_type)


def _safe_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None

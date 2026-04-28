from __future__ import annotations

from transit_core import (
    MAP_DATABASE,
    TransitSystem,
    find_shortest_path,
    get_fare_and_details,
    parse_stations_locally,
)


def main() -> None:
    for name, config in MAP_DATABASE.items():
        system = TransitSystem.from_config(config)
        health = system.data_health()
        assert health["station_count"] > 0, f"{name} has no stations"
        assert not health["missing_neighbors"], f"{name} has missing neighbors"
        print(
            f"{name}: {health['station_count']} stations, "
            f"{health['edge_count']} edges, {health['fare_pairs']} fare pairs"
        )

    krt = TransitSystem.from_config(MAP_DATABASE["高雄捷運"])
    parsed = parse_stations_locally("從高鐵站搭到愛河之心", krt)
    assert parsed.start_id == "R16" and parsed.end_id == "C24", parsed

    path = find_shortest_path(krt, parsed.start_id, parsed.end_id)
    assert path, "demo path should exist"
    fare = get_fare_and_details(krt, path)
    assert fare.amount > 0, fare
    print(f"Demo query: {parsed.start_id} -> {parsed.end_id}, {len(path)} stations, {fare.amount} dollars")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

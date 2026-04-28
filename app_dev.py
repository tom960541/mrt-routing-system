from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from transit_core import MAP_DATABASE, TransitSystem, load_json_data


BASE_DIR = Path(__file__).resolve().parent


def run() -> None:
    st.title("系統後台管理與資料庫監控")
    st.caption("用來向教授展示資料不是寫死在畫面中，而是由 JSON 路網與票價矩陣驅動。")

    selected_map = st.selectbox("選擇要檢查的路網", list(MAP_DATABASE.keys()))
    config = MAP_DATABASE[selected_map]

    try:
        system = TransitSystem.from_config(config)
        health = system.data_health()
    except Exception as exc:
        st.error(f"資料載入失敗：{exc}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("站點數", health["station_count"])
    col2.metric("路網邊數", health["edge_count"])
    col3.metric("票價組合", health["fare_pairs"])
    col4.metric("可能轉乘節點", health["transfer_like_nodes"])

    min_fare = health["min_fare"]
    max_fare = health["max_fare"]
    st.info(f"票價範圍：{min_fare if min_fare is not None else '-'} 到 {max_fare if max_fare is not None else '-'} 元")

    missing_neighbors = health["missing_neighbors"]
    if missing_neighbors:
        st.warning(f"發現 {len(missing_neighbors)} 筆 neighbors 指向不存在站點，建議修正 JSON。")
        st.dataframe(missing_neighbors, use_container_width=True)
    else:
        st.success("路網 neighbors 檢查通過：沒有指向不存在站點的連線。")

    st.divider()
    target_file = st.selectbox(
        "檢視原始資料",
        [config.data_file, config.fare_file, "krt_data.json", "tpi_data.json", "krt_real_fare.json", "tpi_real_fare.json"],
    )

    try:
        data = load_json_data(target_file)
        st.success(f"成功讀取 {target_file}，共包含 {len(data)} 筆主鍵資料。")
        with st.expander("展開 JSON 節點資料"):
            st.json(data)
    except FileNotFoundError:
        st.error(f"找不到檔案 {target_file}。")
    except json.JSONDecodeError as exc:
        st.error(f"{target_file} 不是有效 JSON：{exc}")

    st.divider()
    st.code(
        f"資料目錄：{BASE_DIR}\n"
        "核心模組：transit_core.py\n"
        "前台介面：app_user.py\n"
        "後台介面：app_dev.py",
        language="text",
    )

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import streamlit as st
from gtts import gTTS
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from transit_core import (
    MAP_DATABASE,
    StationParseResult,
    TransitSystem,
    find_cheapest_path,
    find_shortest_path,
    get_fare_and_details,
    parse_ai_json,
    parse_stations_locally,
)


DEFAULT_MAP_WIDTH = 520
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def load_transit_system(map_name: str) -> TransitSystem:
    return TransitSystem.from_config(MAP_DATABASE[map_name])


def get_optional_secret(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_model_candidates() -> list[str]:
    configured = get_optional_secret("GEMINI_MODEL")
    candidates = [configured] if configured else []
    candidates.extend(["gemini-2.0-flash", "gemini-1.5-flash"])
    return [model for index, model in enumerate(candidates) if model and model not in candidates[:index]]


def get_stations_from_ai(user_text: str, system: TransitSystem) -> StationParseResult:
    if not user_text.strip():
        return StationParseResult(None, None, "尚未輸入查詢文字。", "input")

    api_key = get_optional_secret("GEMINI_API_KEY")
    if not api_key:
        local_result = parse_stations_locally(user_text, system)
        if local_result.ok:
            return StationParseResult(local_result.start_id, local_result.end_id, "未設定 API Key，已改用本地站名/別名解析。", "local")
        return StationParseResult(None, None, "未設定 API Key，且本地解析找不到兩個站點。", "local")

    prompt = (
        "你是一個捷運站點解析器。"
        "請只輸出 JSON，不要輸出解釋文字。"
        "格式必須是 {\"start_id\":\"...\",\"end_id\":\"...\"}。"
        f"可用站點清單：{system.station_catalog_for_prompt()}。"
        f"使用者輸入：{user_text}"
    )

    last_error = ""
    for model_name in get_model_candidates():
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=model_name, contents=prompt)
            start_id, end_id = parse_ai_json(response.text or "")
            if start_id in system.stations and end_id in system.stations:
                return StationParseResult(start_id, end_id, f"Gemini 解析成功（{model_name}）。", "ai")
            last_error = f"{model_name} 回傳站碼不在資料庫中"
        except Exception as exc:
            last_error = f"{model_name}: {exc}"

    local_result = parse_stations_locally(user_text, system)
    if local_result.ok:
        return StationParseResult(
            local_result.start_id,
            local_result.end_id,
            f"AI 暫時不可用，已改用本地站名/別名解析。AI 訊息：{last_error}",
            "local",
        )

    return StationParseResult(None, None, f"AI 與本地解析皆失敗。AI 訊息：{last_error}", "failed")


def generate_speech_audio(start_name: str, end_name: str, fare: int) -> BytesIO | None:
    try:
        text = f"已為您規劃從{start_name}到{end_name}的路徑。總票價{fare}元。"
        tts = gTTS(text=text, lang="zh-tw")
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception:
        return None


def render_path(path_ids: list[str], system: TransitSystem) -> str:
    return " -> ".join(f"[{system.get_station(sid).line_type}] {system.get_station(sid).name}" for sid in path_ids)


def sync_station_state(config_key: str, names: list[str]) -> tuple[str, str, str, str]:
    start_key = f"{config_key}_start_st"
    end_key = f"{config_key}_end_st"
    click_key = f"{config_key}_next_click_is_start"
    last_click_key = f"{config_key}_last_click"

    if start_key not in st.session_state or st.session_state[start_key] not in names:
        st.session_state[start_key] = names[0]
    if end_key not in st.session_state or st.session_state[end_key] not in names:
        st.session_state[end_key] = names[min(1, len(names) - 1)]
    if click_key not in st.session_state:
        st.session_state[click_key] = True
    if last_click_key not in st.session_state:
        st.session_state[last_click_key] = None

    return start_key, end_key, click_key, last_click_key


def run() -> None:
    try:
        st.set_page_config(page_title="AI 智慧捷運路徑規劃系統", layout="wide")
    except Exception:
        pass

    st.title("AI 智慧捷運路徑規劃系統")
    st.caption("自然語言站點解析、Graph 路徑規劃、票價矩陣與地圖點選整合展示")

    selected_map = st.selectbox("選擇路網", list(MAP_DATABASE.keys()))
    config = MAP_DATABASE[selected_map]

    try:
        system = load_transit_system(selected_map)
    except Exception as exc:
        st.error(f"資料載入失敗：{exc}")
        return

    names = system.get_all_display_names()
    if not names:
        st.error("目前路網沒有站點資料。")
        return

    start_key, end_key, click_key, last_click_key = sync_station_state(config.key, names)
    col_ui, col_map = st.columns([1, 2.2])

    with col_ui:
        st.subheader("AI 文字助理")
        if not get_optional_secret("GEMINI_API_KEY"):
            st.info("目前未設定 GEMINI_API_KEY，系統會自動使用本地站名/別名解析，仍可現場 Demo。")

        with st.form(key=f"{config.key}_ai_form"):
            user_input = st.text_input("你想去哪？", placeholder="例如：從高鐵站搭到愛河之心")
            submit_btn = st.form_submit_button("AI 解析起終點", use_container_width=True)

        if submit_btn:
            with st.spinner("正在解析自然語言..."):
                result = get_stations_from_ai(user_input, system)

            if result.ok:
                st.session_state[start_key] = system.get_station(result.start_id).display_name
                st.session_state[end_key] = system.get_station(result.end_id).display_name
                st.success(result.message)
                st.rerun()
            else:
                st.error(result.message)

        st.divider()
        start_index = names.index(st.session_state[start_key])
        end_index = names.index(st.session_state[end_key])
        selected_start = st.selectbox("出發站", names, index=start_index)
        selected_end = st.selectbox("終點站", names, index=end_index)
        st.session_state[start_key] = selected_start
        st.session_state[end_key] = selected_end

        search_mode = st.radio(
            "路徑規劃策略",
            ["最少站數（BFS）", "票價估算成本（Dijkstra）"],
            horizontal=True,
        )

        if st.button("查詢路徑", type="primary", use_container_width=True):
            if selected_start == selected_end:
                st.warning("起點與終點相同，無需搭乘。")
            else:
                start_id = system.get_sid_by_name(selected_start)
                end_id = system.get_sid_by_name(selected_end)
                path_ids = (
                    find_shortest_path(system, start_id, end_id)
                    if "BFS" in search_mode
                    else find_cheapest_path(system, start_id, end_id)
                )

                if not path_ids:
                    st.error("找不到相連路徑。")
                    return

                fare = get_fare_and_details(system, path_ids)
                st.success(f"系統報價：{fare.amount} 元｜總站數：{len(path_ids)} 站｜來源：{fare.source}")
                st.text_area("路徑詳情", fare.details, height=140)
                st.info(f"建議路徑：\n{render_path(path_ids, system)}")

                audio = generate_speech_audio(selected_start, selected_end, fare.amount)
                if audio:
                    st.audio(audio, format="audio/mp3", autoplay=True)
                else:
                    st.caption("語音播報服務目前無法連線，文字結果已完整產生。")

    with col_map:
        st.subheader("互動地圖")
        st.info("請點擊出發站" if st.session_state[click_key] else "請點擊終點站")

        try:
            image_path = BASE_DIR / config.image_file
            img = Image.open(image_path)
            original_width, _ = img.size
            click = streamlit_image_coordinates(img, width=DEFAULT_MAP_WIDTH, key=f"{config.key}_map_click")

            if click:
                click_point = (click["x"], click["y"])
                if st.session_state[last_click_key] != click_point:
                    st.session_state[last_click_key] = click_point
                    station = system.nearest_station(click["x"], click["y"], DEFAULT_MAP_WIDTH, original_width)
                    if station:
                        if st.session_state[click_key]:
                            st.session_state[start_key] = station.display_name
                            st.session_state[click_key] = False
                        else:
                            st.session_state[end_key] = station.display_name
                            st.session_state[click_key] = True
                        st.rerun()
        except Exception as exc:
            st.error(f"地圖元件錯誤：{exc}")


if __name__ == "__main__":
    run()

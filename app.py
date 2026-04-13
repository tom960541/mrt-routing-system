import streamlit as st
import json
import math
import re
from collections import deque
from PIL import Image
from io import BytesIO
from gtts import gTTS
from google import genai
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 系統設定
# ==========================================
# 建議將 API_KEY 放在 Streamlit Secrets 中
API_KEY = st.secrets.get("GEMINI_API_KEY", "你的_API_KEY") 
client = genai.Client(api_key=API_KEY)

MAP_DATABASE = {
    "高雄捷運": {
        "json": "krt_data.json", 
        "img": "krt_map.jpg", 
        "fare_json": "krt_real_fare.json"
    },
    "台北捷運": {
        "json": "tpi_data.json", 
        "img": "TaipeiMetroStamp.png", 
        "fare_json": "tpi_real_fare.json"
    }
}

# ==========================================
# 核心資料結構
# ==========================================
@st.cache_data
def load_json_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

class Station:
    def __init__(self, sid, name, coords, line_type, neighbors):
        self.sid = sid
        self.name = name
        self.display_name = name # 顯示純中文名
        self.coords = coords     # 座標：用於地圖點擊判定
        self.line_type = line_type
        self.neighbors = neighbors

class TransitSystem:
    def __init__(self, data, fare_matrix):
        self.stations = {}
        self.fare_matrix = fare_matrix # 真實票價矩陣
        if not data: return 
        for sid, info in data.items():
            self.stations[sid] = Station(
                sid=sid, name=info["name"], coords=info["coords"],
                line_type=info["line_type"], neighbors=info.get("neighbors", [])
            )

    def get_station(self, sid):
        return self.stations.get(sid)

    def get_all_display_names(self):
        unique_names = set(s.display_name for s in self.stations.values())
        return sorted(list(unique_names))

    def get_sid_by_name(self, display_name):
        for sid, s in self.stations.items():
            if s.display_name == display_name:
                return sid
        return None

# ==========================================
# 邏輯函式
# ==========================================
def find_shortest_path(system, start_id, end_id):
    """BFS 尋找最少站數路徑"""
    if not start_id or not end_id: return []
    queue = deque([[start_id]])
    visited = {start_id}
    while queue:
        path = queue.popleft()
        if path[-1] == end_id: return path
        curr_st = system.get_station(path[-1])
        for neighbor in curr_st.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []

def get_fare_and_details(system, path_ids):
    """從矩陣查票價並組合路徑明細"""
    if not path_ids or len(path_ids) < 2: return 0, "無需搭乘"
    start_id, end_id = path_ids[0], path_ids[-1]

    # 查表獲取真實總票價
    try:
        total_fare = system.fare_matrix.get(start_id, {}).get(end_id, 0)
    except:
        total_fare = "查無資料"

    details = []
    curr_line = system.get_station(path_ids[0]).line_type
    seg_start_name = system.get_station(path_ids[0]).name
    for i in range(1, len(path_ids)):
        st_info = system.get_station(path_ids[i])
        if st_info.line_type != curr_line:
            details.append(f"- {curr_line} 線: {seg_start_name} ➔ {system.get_station(path_ids[i-1]).name}")
            seg_start_name = system.get_station(path_ids[i-1]).name
            curr_line = st_info.line_type
    details.append(f"- {curr_line} 線: {seg_start_name} ➔ {system.get_station(path_ids[-1]).name}")
    details.append(f"\n💰 官方真實票價：{total_fare} 元")
    return total_fare, "\n".join(details)

def get_stations_from_ai(user_text, system):
    try:
        station_info = ", ".join([f"{s.name}({s.sid})" for s in system.stations.values()])
        prompt = f"你是一個捷運解析器。請嚴格輸出JSON: {{\"start_id\":\"...\",\"end_id\":\"...\"}}。站點列表:[{station_info}]。輸入：「{user_text}」"
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("start_id"), data.get("end_id"), "成功"
        return None, None, "格式錯誤"
    except Exception as e: return None, None, str(e)

def generate_speech_audio(start_name, end_name, fare):
    text = f"已為您規劃從{start_name}到{end_name}的路徑。總票價{fare}元。"
    tts = gTTS(text=text, lang='zh-tw')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# ==========================================
# UI 介面
# ==========================================
def run():
    st.title("🚆 智慧捷運路徑規劃系統 (點擊地圖+真實票價版)")

    selected_map = st.selectbox("🗺️ 選擇路網", list(MAP_DATABASE.keys()))
    config = MAP_DATABASE[selected_map]

    sys_data = load_json_data(config["json"])
    fare_data = load_json_data(config["fare_json"])
    mrt = TransitSystem(sys_data, fare_data)

    if not sys_data:
        st.error("地圖資料載入失敗")
        return

    # 初始化 Session State
    names = mrt.get_all_display_names()
    if 'start_st' not in st.session_state: st.session_state.start_st = names[0]
    if 'end_st' not in st.session_state: st.session_state.end_st = names[0]
    if 'next_click_is_start' not in st.session_state: st.session_state.next_click_is_start = True
    if 'last_click' not in st.session_state: st.session_state.last_click = None

    col_ui, col_map = st.columns([1, 2])

    with col_ui:
        st.subheader("✨ AI 語音/文字助理")
        user_input = st.text_input("你想去哪？", placeholder="例如：從高鐵站搭到大東")
        if st.button("🤖 AI 規劃"):
            sid_s, sid_e, msg = get_stations_from_ai(user_input, mrt)
            if sid_s and sid_e:
                st.session_state.start_st = mrt.get_station(sid_s).display_name
                st.session_state.end_st = mrt.get_station(sid_e).display_name
                st.rerun()

        st.divider()
        idx_s = names.index(st.session_state.start_st) if st.session_state.start_st in names else 0
        idx_e = names.index(st.session_state.end_st) if st.session_state.end_st in names else 0

        sel_start = st.selectbox("出發站", names, index=idx_s)
        sel_end = st.selectbox("終點站", names, index=idx_e)

        # 同步手動選單回 Session State
        st.session_state.start_st = sel_start
        st.session_state.end_st = sel_end

        if st.button("🔍 查詢路徑", type="primary"):
            if sel_start == sel_end:
                st.warning("起終點相同")
            else:
                id_s, id_e = mrt.get_sid_by_name(sel_start), mrt.get_sid_by_name(sel_end)
                path = find_shortest_path(mrt, id_s, id_e)
                if path:
                    fare, details = get_fare_and_details(mrt, path)
                    st.success(f"**真實總票價：{fare} 元**")
                    st.text_area("路徑詳情", details, height=150)
                    audio = generate_speech_audio(sel_start, sel_end, fare)
                    st.audio(audio, format="audio/mp3", autoplay=True)

    with col_map:
        st.subheader("🗺️ 互動地圖")
        if st.session_state.next_click_is_start: st.info("👆 請在地圖點擊 **出發站**")
        else: st.warning("👆 請在地圖點擊 **終點站**")

        try:
            # 1. 讀取原始圖片
            img = Image.open(config["img"])
            w_orig, h_orig = img.size

            # 2. 設定我們想要的固定網頁顯示寬度 (例如 800)
        
            TARGET_WIDTH = 450
            scale_ratio = TARGET_WIDTH / w_orig  # 計算縮放比例
            TARGET_HEIGHT = int(h_orig * scale_ratio)

            # 3. 使用 Python 強制縮放圖片
            img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT))

            # 4. 顯示縮放後的圖片 (移除 use_column_width，保持 1:1)
            click = streamlit_image_coordinates(img_resized, key="map_click")

            if click:
                cx, cy = click["x"], click["y"]
                if st.session_state.last_click != (cx, cy):
                    st.session_state.last_click = (cx, cy)
                    closest, min_dist = None, float('inf')

                    for s in mrt.stations.values():
                        # ✨ 重點：將 JSON 的原始座標也乘上縮放比例，對齊目前的畫面！
                        scaled_x = s.coords[0] * scale_ratio
                        scaled_y = s.coords[1] * scale_ratio

                        dist = math.sqrt((cx - scaled_x)**2 + (cy - scaled_y)**2)
                        if dist < min_dist: 
                            min_dist, closest = dist, s

                    # ✨ 容錯距離也要跟著縮放 (假設原本容許誤差 130px)
                    threshold = 130 * scale_ratio

                    if closest and min_dist < threshold:
                        if st.session_state.next_click_is_start:
                            st.session_state.start_st = closest.display_name
                            st.session_state.next_click_is_start = False
                        else:
                            st.session_state.end_st = closest.display_name
                            st.session_state.next_click_is_start = True
                        st.rerun()
        except Exception as e:
            st.error(f"地圖元件錯誤: {e}")

if __name__ == "__main__":
    run()

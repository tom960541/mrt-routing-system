import streamlit as st
import json
import math
import heapq
import re
import os
from collections import deque
from PIL import Image
from io import BytesIO
from gtts import gTTS
from google import genai

# ==========================================
# 系統設定與地圖預設資料庫
# ==========================================
# 建議將 API_KEY 放在 Streamlit 的 Secrets 中
API_KEY = st.secrets.get("GEMINI_API_KEY", "你的_API_KEY") 
client = genai.Client(api_key=API_KEY)

# 預先設定好支援的地圖，包含路網 JSON、地圖圖片與真實票價矩陣 JSON
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
        self.display_name = name # 前端顯示純站名，過濾重複
        self.coords = coords
        self.line_type = line_type
        self.neighbors = neighbors

class TransitSystem:
    def __init__(self, data, fare_matrix):
        self.stations = {}
        self.fare_matrix = fare_matrix # 儲存 N x N 真實票價表
        if not data: return 
        for sid, info in data.items():
            self.stations[sid] = Station(
                sid=sid, name=info["name"], coords=info["coords"],
                line_type=info["line_type"], neighbors=info.get("neighbors", [])
            )
            
    def get_station(self, sid):
        return self.stations.get(sid)

    def get_all_display_names(self):
        # 利用 set 去除轉乘站造成的重複站名
        unique_names = set(s.display_name for s in self.stations.values())
        return sorted(list(unique_names))
        
    def get_sid_by_name(self, display_name):
        for sid, s in self.stations.items():
            if s.display_name == display_name:
                return sid
        return None

# ==========================================
# 演算法與邏輯 (查表化)
# ==========================================

def get_stations_from_ai(user_text, system):
    try:
        station_info = ", ".join([f"{s.name}({s.sid})" for s in system.stations.values()])
        prompt = f"你是一個捷運站點解析器。請從句子推斷起點與終點站代號。嚴格輸出 JSON。列表：[{station_info}]。輸入：「{user_text}」"
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("start_id"), data.get("end_id"), "成功"
        return None, None, "AI 解析失敗"
    except Exception as e:
        return None, None, str(e)

def find_path(system, start_id, end_id):
    """使用 BFS 尋找站數最少的路徑"""
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
    """
    從 fare_matrix 真實票價表查出總價
    並根據路徑 path_ids 組合文字說明
    """
    if not path_ids or len(path_ids) < 2: return 0, "無需搭乘"
    
    start_id = path_ids[0]
    end_id = path_ids[-1]
    
    # 1. 直接查表獲取真實總票價 (O(1))
    # 假設 JSON 結構為 fare_matrix[start_id][end_id]
    try:
        total_fare = system.fare_matrix.get(start_id, {}).get(end_id, 0)
    except Exception:
        total_fare = "查無資料"

    # 2. 組合路徑描述 (顯示轉乘細節)
    details = []
    curr_line = system.get_station(path_ids[0]).line_type
    seg_start_name = system.get_station(path_ids[0]).name
    
    for i in range(1, len(path_ids)):
        st_info = system.get_station(path_ids[i])
        if st_info.line_type != curr_line:
            details.append(f"- {curr_line} 線: {seg_start_name} ➔ {system.get_station(path_ids[i-1]).name}")
            seg_start_name = system.get_station(path_ids[i-1]).name # 轉乘點
            curr_line = st_info.line_type
            
    details.append(f"- {curr_line} 線: {seg_start_name} ➔ {system.get_station(path_ids[-1]).name}")
    details.append(f"\n💰 官方真實票價：{total_fare} 元")
    
    return total_fare, "\n".join(details)

def generate_speech_audio(start_name, end_name, fare):
    text = f"已為您規劃從{start_name}到{end_name}的路徑。總票價{fare}元。"
    tts = gTTS(text=text, lang='zh-tw')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="捷運智慧路徑規劃", layout="wide")
st.title("🚆 智慧捷運查表路徑規劃系統 (TDX 資料版)")

# 1. 載入資料
selected_map = st.selectbox("🗺️ 選擇路網", list(MAP_DATABASE.keys()))
config = MAP_DATABASE[selected_map]

sys_data = load_json_data(config["json"])
fare_data = load_json_data(config["fare_json"])
mrt = TransitSystem(sys_data, fare_data)

col_ui, col_map = st.columns([1, 2])

with col_ui:
    st.subheader("✨ AI 語音/文字助理")
    user_input = st.text_input("你想去哪？", placeholder="例如：從台北車站去淡水")
    
    if 'start_st' not in st.session_state: st.session_state.start_st = None
    if 'end_st' not in st.session_state: st.session_state.end_st = None

    if st.button("🤖 AI 規劃"):
        with st.spinner('解析中...'):
            sid_s, sid_e, msg = get_stations_from_ai(user_input, mrt)
            if sid_s and sid_e:
                st.session_state.start_st = mrt.get_station(sid_s).display_name
                st.session_state.end_st = mrt.get_station(sid_e).display_name
                st.success("解析成功！")
            else:
                st.error(f"解析失敗: {msg}")

    st.divider()
    
    names = mrt.get_all_display_names()
    idx_s = names.index(st.session_state.start_st) if st.session_state.start_st in names else 0
    idx_e = names.index(st.session_state.end_st) if st.session_state.end_st in names else 0
    
    sel_start = st.selectbox("出發站", names, index=idx_s)
    sel_end = st.selectbox("終點站", names, index=idx_e)

    if st.button("🔍 查詢路徑與真實票價", type="primary"):
        if sel_start == sel_end:
            st.warning("起終點相同。")
        else:
            id_s, id_e = mrt.get_sid_by_name(sel_start), mrt.get_sid_by_name(sel_end)
            path = find_path(mrt, id_s, id_e)
            
            if path:
                fare, details = get_fare_and_details(mrt, path)
                
                # 顯示結果
                st.success(f"**真實總票價：{fare} 元**")
                st.text_area("路徑詳情", details, height=150)
                
                # 建議路徑字串
                path_display = " ➔ ".join([mrt.get_station(i).name for i in path])
                st.info(f"建議路徑：{path_display}")

                # 播放語音
                audio = generate_speech_audio(sel_start, sel_end, fare)
                st.audio(audio, format="audio/mp3", autoplay=True)
            else:
                st.error("找不到連通路徑。")

with col_map:
    st.subheader("🗺️ 路網圖預覽")
    try:
        img = Image.open(config["img"])
        st.image(img, use_container_width=True)
    except:
        st.warning("圖檔載入失敗")

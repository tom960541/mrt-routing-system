import streamlit as st
import json
import math
import heapq
import re
from collections import deque
from PIL import Image
from io import BytesIO
from gtts import gTTS
from google import genai
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 策略模式：定義不同捷運的計價邏輯
# ==========================================
def krt_fare_strategy(dist, line_type):
    """高雄捷運計價公式：起跳20，超過5km每2km加5元，上限60(輕軌35)"""
    fare = 20 + (math.ceil((dist - 5) / 2) * 5) if dist > 5 else 20
    max_fare = 35 if "C" in line_type or "LRT" in line_type else 60
    return min(fare, max_fare)

def tpi_fare_strategy(dist, line_type):
    """台北捷運計價公式 (簡化版)：起跳20，超過5km每3km加5元，上限65"""
    fare = 20 + (math.ceil((dist - 5) / 3) * 5) if dist > 5 else 20
    return min(fare, 65)

# ==========================================
# 系統設定與地圖資料庫 (綁定計價策略)
# ==========================================
# 部署前記得改用 st.secrets!
API_KEY = "AIzaSyAxDkIOX4d6Ve3pXPGAQtfT33NsKo4Gg7w" 
client = genai.Client(api_key=API_KEY)
AVG_DISTANCE_PER_SEGMENT = 1.3

MAP_DATABASE = {
    "高雄捷運": {
        "json": "krt_data.json", 
        "img": "krt_map.jpg",
        "fare_func": krt_fare_strategy
    },
    "台北捷運": {
        "json": "tpi_data.json", 
        "img": "TaipeiMetroStamp.png",
        "fare_func": tpi_fare_strategy
    }
}

# ==========================================
# 核心資料結構與演算法
# ==========================================
@st.cache_data
def load_system_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

class Station:
    def __init__(self, sid, name, coords, line_type, neighbors):
        self.sid = sid
        self.name = name
        # 修改點 1：將顯示名稱改為「純站名」，捨棄代號，方便前端過濾重複
        self.display_name = name 
        self.coords = coords
        self.line_type = line_type
        self.neighbors = neighbors

class TransitSystem:
    def __init__(self, data, fare_func):
        self.stations = {}
        self.fare_func = fare_func 
        if not data: return 
        for sid, info in data.items():
            self.stations[sid] = Station(
                sid=sid, name=info["name"], coords=info["coords"],
                line_type=info["line_type"], neighbors=info.get("neighbors", [])
            )

    def get_station(self, sid): 
        return self.stations.get(sid)
        
    def get_all_display_names(self): 
        # 修改點 2：利用 set (集合) 自動濾除同名站點，再轉回 list 進行排序
        unique_names = set(s.display_name for s in self.stations.values())
        return sorted(list(unique_names))
        
    def get_sid_by_name(self, display_name):
        # 只要純中文站名對得上，就回傳找到的第一個代碼 (演算法會處理跨線問題)
        for sid, s in self.stations.items():
            if s.display_name == display_name: return sid
        return None

def get_stations_from_ai(user_text, system):
    try:
        station_info = ", ".join([f"名稱:{s.name}(代號:{s.sid})" for s in system.stations.values()])
        prompt = f"""
        你是一個智慧捷運站點解析器。請從使用者的句子推斷起點與終點捷運站。
        可用站點：[{station_info}]
        請「絕對嚴格」只輸出 JSON。範例：{{"start_id": "R16", "end_id": "O10"}}
        使用者輸入：「{user_text}」
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("start_id"), data.get("end_id"), "成功"
        return None, None, "AI 回傳格式錯誤"
    except Exception as e:
        return None, None, str(e)

def find_shortest_path(system, start_id, end_id):
    if not system.get_station(start_id) or not system.get_station(end_id): return []
    queue = deque([[start_id]])
    visited = {start_id}
    while queue:
        path = queue.popleft()
        if path[-1] == end_id: return path
        for neighbor in system.get_station(path[-1]).neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []

def find_cheapest_path(system, start_id, end_id):
    start_station = system.get_station(start_id)
    if not start_station: return []
    pq = [(0, 1, [start_id], start_station.line_type)]
    min_costs = {(start_id, start_station.line_type): (0, 1)}

    while pq:
        curr_fare, num_stat, path, curr_line = heapq.heappop(pq)
        curr_id = path[-1]
        best = min_costs.get((curr_id, curr_line), (float('inf'), float('inf')))
        if curr_fare > best[0] or (curr_fare == best[0] and num_stat > best[1]): continue
        if curr_id == end_id: return path

        for neighbor_id in system.get_station(curr_id).neighbors:
            new_path = path + [neighbor_id]
            new_fare, _ = calculate_fare_details(system, new_path)
            neighbor_line = system.get_station(neighbor_id).line_type

            best_neigh = min_costs.get((neighbor_id, neighbor_line), (float('inf'), float('inf')))
            if new_fare < best_neigh[0] or (new_fare == best_neigh[0] and len(new_path) < best_neigh[1]):
                min_costs[(neighbor_id, neighbor_line)] = (new_fare, len(new_path))
                heapq.heappush(pq, (new_fare, len(new_path), new_path, neighbor_line))
    return []

def calculate_fare_details(system, path_ids):
    if not path_ids or len(path_ids) < 2: return 0, "無須搭乘"
    total_fare, details = 0, []
    segment = [path_ids[0]]
    curr_line = system.get_station(path_ids[0]).line_type

    def process_segment(seg, line):
        dist = (len(seg) - 1) * AVG_DISTANCE_PER_SEGMENT
        fare = system.fare_func(dist, line)
        details.append(f"- {line}: {system.get_station(seg[0]).name} ➔ {system.get_station(seg[-1]).name} (距離 {dist:.1f} km, 票價 {fare} 元)")
        return fare

    for i in range(1, len(path_ids)):
        sid = path_ids[i]
        next_line = system.get_station(sid).line_type
        if next_line != curr_line:
            segment.append(sid)
            total_fare += process_segment(segment, curr_line)
            segment, curr_line = [sid], next_line
        else:
            segment.append(sid)

    if len(segment) > 1:
        total_fare += process_segment(segment, curr_line)

    return total_fare, "\n".join(details)

def generate_speech_audio(start_name, end_name, fare, path_str):
    # 因為前面已經拿掉代號了，直接用純中文名即可
    text = f"已為您規劃從{start_name}到{end_name}的路徑。總票價{fare}元。"
    if "轉乘" in path_str: text += "此路徑包含轉乘，請留意車廂廣播。"
    tts = gTTS(text=text, lang='zh-tw')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# ==========================================
# Streamlit 網頁介面
# ==========================================
st.set_page_config(page_title="智慧捷運路徑規劃", layout="wide")
st.title("🚆 通用捷運 AI 語音路徑規劃系統")

selected_map_name = st.selectbox("🗺️ 請選擇要使用的捷運路網", list(MAP_DATABASE.keys()))
map_config = MAP_DATABASE[selected_map_name]

# 初始化帶有專屬策略的 TransitSystem
system_data = load_system_data(map_config["json"])
krt = TransitSystem(system_data, map_config["fare_func"])

if not system_data:
    st.error(f"找不到地圖資料檔：{map_config['json']}，請確定檔案存在。")
    st.stop()

display_names = krt.get_all_display_names()

# 狀態初始化
if 'start_station' not in st.session_state: st.session_state.start_station = display_names[0] if display_names else ""
if 'end_station' not in st.session_state: st.session_state.end_station = display_names[0] if display_names else ""
if 'next_click_is_start' not in st.session_state: st.session_state.next_click_is_start = True
if 'last_click' not in st.session_state: st.session_state.last_click = None

col_ui, col_map = st.columns([1, 2])

with col_ui:
    st.subheader("✨ 告訴 AI 你想去哪裡")
    user_ai_input = st.text_input("輸入需求：", placeholder="例如：從高鐵站搭到駁二怎麼走？")

    if st.button("🤖 AI 幫我找"):
        with st.spinner('AI 正在努力分析您的需求...'):
            s_id, e_id, status = get_stations_from_ai(user_ai_input, krt)
            if s_id and e_id and krt.get_station(s_id) and krt.get_station(e_id):
                st.session_state.start_station = krt.get_station(s_id).display_name
                st.session_state.end_station = krt.get_station(e_id).display_name
                st.success("✅ AI 解析成功！已自動填入下方站點。")
            else:
                st.error(f"❌ AI 分析失敗：{status}")

    st.divider()

    # 修改點 3：清理重複渲染的 UI 區塊，統一使用 Session State 索引連動
    start_idx = display_names.index(st.session_state.start_station) if st.session_state.start_station in display_names else 0
    end_idx = display_names.index(st.session_state.end_station) if st.session_state.end_station in display_names else 0

    selected_start = st.selectbox("出發站", display_names, index=start_idx)
    selected_end = st.selectbox("目的站", display_names, index=end_idx)

    # 捕捉使用者手動切換選單的操作，並更新到 Session State 中
    if selected_start != st.session_state.start_station or selected_end != st.session_state.end_station:
        st.session_state.start_station = selected_start
        st.session_state.end_station = selected_end

    # 加入搜尋策略選項
    st.subheader("⚙️ 路徑規劃策略")
    search_mode = st.radio(
        "請選擇優先考量：", 
        ["最省票價 (最佳權重 Dijkstra)", "最少站數 (最短路徑 BFS)"],
        horizontal=True
    )

    if st.button("🔍 開始規劃路徑", type="primary"):
        start_station = st.session_state.start_station
        end_station = st.session_state.end_station

        if start_station == end_station:
            st.warning("您已經在目的地囉！")
        else:
            s_id = krt.get_sid_by_name(start_station)
            e_id = krt.get_sid_by_name(end_station)

            # 動態切換尋路演算法
            if "最少站數" in search_mode:
                path_ids = find_shortest_path(krt, s_id, e_id)
            else:
                path_ids = find_cheapest_path(krt, s_id, e_id)

            if path_ids:
                total_fare, fare_details = calculate_fare_details(krt, path_ids)
                display_path = []
                for i, node_id in enumerate(path_ids):
                    st_info = krt.get_station(node_id)
                    if i > 0 and st_info.name == krt.get_station(path_ids[i-1]).name:
                        display_path[-1] += f" (轉乘 {st_info.line_type})"
                    else:
                        display_path.append(f"[{st_info.line_type}] {st_info.name}")
                path_str = " ➔ ".join(display_path)

                st.success(f"**總票價**：{total_fare} 元 | **總站數**：{len(path_ids)} 站")
                st.text_area("票價詳情", fare_details, height=100)
                st.info(f"**建議路徑**：\n{path_str}")

                audio_data = generate_speech_audio(start_station, end_station, total_fare, path_str)
                st.audio(audio_data, format="audio/mp3", autoplay=True)
            else:
                st.error("找不到相連的路徑，請確認地圖資料設定 (Neighbors)。")

with col_map:
    st.subheader("🗺️ 捷運路網圖")
    if st.session_state.next_click_is_start:
        st.info("👆 請在地圖上點擊您的 **出發站**")
    else:
        st.warning("👆 請在地圖上點擊您的 **目的站**")

    try:
        img = Image.open(map_config["img"])
        click_coords = streamlit_image_coordinates(img, key="map_click")

        if click_coords is not None:
            cx, cy = click_coords["x"], click_coords["y"]
            click_point = (cx, cy)

            if st.session_state.last_click != click_point:
                st.session_state.last_click = click_point
                closest_station, min_dist = None, float('inf')
                for station in krt.stations.values():
                    sx, sy = station.coords
                    dist = math.sqrt((cx-sx)**2 + (cy-sy)**2)
                    if dist < min_dist:
                        min_dist, closest_station = dist, station

                if closest_station and min_dist < 130:
                    if st.session_state.next_click_is_start:
                        st.session_state.start_station = closest_station.display_name
                        st.session_state.next_click_is_start = False
                    else:
                        st.session_state.end_station = closest_station.display_name
                        st.session_state.next_click_is_start = True
                    st.rerun()
                else:
                    st.error("點擊位置離站點太遠囉！請點擊黑點或站名附近。")

            st.caption(f"🔧 點擊座標：(X: {cx}, Y: {cy}) | 最近站點：{closest_station.name if closest_station else '無'} | 距離：{min_dist:.1f} 像素")
    except Exception as e:
        st.error(f"無法載入地圖元件：{e}")

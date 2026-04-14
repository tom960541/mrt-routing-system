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
# 系統設定與策略模式
# ==========================================
# 建議將 API_KEY 放在 Streamlit Secrets 中
import streamlit as st
from google import genai

# 直接從 Streamlit Secrets 讀取
# 這樣如果找不到 Key 會直接報錯，避免你忘記設定
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("找不到 API Key！請檢查 .streamlit/secrets.toml 是否已正確設定。")
    st.stop() # 停止執行程式

client = genai.Client(api_key=API_KEY)

# 接下來就可以繼續寫你的 app 邏輯了...
AVG_DISTANCE_PER_SEGMENT = 1.3

# ✨ 備用計價公式 (當 TDX 查無資料時的 Fallback)
def krt_fare_strategy(dist, line_type):
    fare = 20 + (math.ceil((dist - 5) / 2) * 5) if dist > 5 else 20
    max_fare = 35 if "C" in line_type or "LRT" in line_type else 60
    return min(fare, max_fare)

def tpi_fare_strategy(dist, line_type):
    fare = 20 + (math.ceil((dist - 5) / 3) * 5) if dist > 5 else 20
    return min(fare, 65)

MAP_DATABASE = {
    "高雄捷運": {
        "json": "krt_data.json", 
        "img": "krt_map.jpg", 
        "fare_json": "krt_real_fare.json",
        "fare_func": krt_fare_strategy  # 加回計費公式
    },
    "台北捷運": {
        "json": "tpi_data.json", 
        "img": "TaipeiMetroStamp.png", 
        "fare_json": "tpi_real_fare.json",
        "fare_func": tpi_fare_strategy  # 加回計費公式
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
    def __init__(self, data, fare_matrix, fare_func):
        self.stations = {}
        self.fare_matrix = fare_matrix # 真實票價矩陣
        self.fare_func = fare_func     # ✨ 儲存備用計價公式
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
# 邏輯函式 (尋路與雙引擎計價)
# ==========================================
def find_shortest_path(system, start_id, end_id):
    """【策略 A】BFS：尋找最少站數路徑"""
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

# ✨ 內部函式：純數學公式估算 (供 Dijkstra 與輕軌備用)
def calculate_fare_fallback(system, path_ids):
    if not path_ids or len(path_ids) < 2: return 0
    total_fare = 0
    segment, curr_line = [path_ids[0]], system.get_station(path_ids[0]).line_type
    
    for i in range(1, len(path_ids)):
        sid, next_line = path_ids[i], system.get_station(path_ids[i]).line_type
        if next_line != curr_line:
            dist = (len(segment) - 1) * AVG_DISTANCE_PER_SEGMENT
            total_fare += system.fare_func(dist, curr_line)
            segment, curr_line = [sid], next_line
        else: segment.append(sid)
        
    if len(segment) > 1:
        dist = (len(segment) - 1) * AVG_DISTANCE_PER_SEGMENT
        total_fare += system.fare_func(dist, curr_line)
    return total_fare

def find_cheapest_path(system, start_id, end_id):
    """【策略 B】Dijkstra：尋找最省票價路徑"""
    start_st = system.get_station(start_id)
    if not start_st: return []
    pq = [(0, 1, [start_id], start_st.line_type)]
    min_costs = {(start_id, start_st.line_type): (0, 1)}

    while pq:
        curr_fare, num_stat, path, curr_line = heapq.heappop(pq)
        curr_id = path[-1]
        best = min_costs.get((curr_id, curr_line), (float('inf'), float('inf')))
        if curr_fare > best[0] or (curr_fare == best[0] and num_stat > best[1]): continue
        if curr_id == end_id: return path

        for n_id in system.get_station(curr_id).neighbors:
            new_path = path + [n_id]
            # 用數學公式算出這個新路線的成本，作為權重
            new_fare = calculate_fare_fallback(system, new_path)
            n_line = system.get_station(n_id).line_type
            
            best_neigh = min_costs.get((n_id, n_line), (float('inf'), float('inf')))
            if new_fare < best_neigh[0] or (new_fare == best_neigh[0] and len(new_path) < best_neigh[1]):
                min_costs[(n_id, n_line)] = (new_fare, len(new_path))
                heapq.heappush(pq, (new_fare, len(new_path), new_path, n_line))
    return []

def get_fare_and_details(system, path_ids):
    """雙引擎計價器：優先查矩陣，查無資料就優雅降級為數學公式"""
    if not path_ids or len(path_ids) < 2: return 0, "無需搭乘"
    start_id, end_id = path_ids[0], path_ids[-1]

    # 1. 嘗試查表獲取真實總票價
    tdx_fare = system.fare_matrix.get(start_id, {}).get(end_id)

    # 2. ✨ 如果查得到就用官方的；查不到(如輕軌)就切換公式估算
    if tdx_fare is not None and tdx_fare > 0:
        total_fare = tdx_fare
        source_tag = "✅ 官方 TDX 真實票價"
    else:
        total_fare = calculate_fare_fallback(system, path_ids)
        source_tag = "⚠️ 系統公式估算 (涵蓋輕軌/特殊路線)"

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
    details.append(f"\n💰 總金額：{total_fare} 元 ({source_tag})")
    
    return total_fare, "\n".join(details)

def get_stations_from_ai(user_text, system):
    try:
        if not user_text.strip():
            return None, None, "您沒有輸入任何文字喔！"

        station_info = ", ".join([f"{s.name}({s.sid})" for s in system.stations.values()])
        prompt = f"你是一個捷運解析器。請嚴格輸出JSON: {{\"start_id\":\"...\",\"end_id\":\"...\"}}。站點列表:[{station_info}]。輸入：「{user_text}」"
        
        # 🚀 重大升級：建立「模型備援清單 (Fallback List)」
        # 系統會依序嘗試，直到找到你的 API Key 有權限使用的模型為止！
        model_candidates = [
            'gemini-2.0-flash', 
            'gemini-1.5-flash-latest', 
            'gemini-1.5-flash', 
            'gemini-pro'
        ]
        
        response = None
        last_error = ""
        
        for model_name in model_candidates:
            try:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=prompt
                )
                break # ✨ 只要有一個模型成功，就立刻跳出迴圈！
            except Exception as e:
                last_error = str(e)
                continue # ❌ 失敗了沒關係，默默嘗試名單上的下一個
                
        # 如果整份名單都試過了還是失敗
        if not response:
            return None, None, f"您的金鑰無法存取已知模型。最後錯誤：{last_error}"
        
        match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            
            s_id = data.get("start_id")
            e_id = data.get("end_id")
            
            # 防呆：檢查 AI 抓出的代號是不是真的存在於地圖中
            if s_id not in system.stations or e_id not in system.stations:
                return None, None, f"AI 產生了不存在的站點代號 ({s_id}, {e_id})"
                
            return s_id, e_id, "成功"
            
        return None, None, f"AI 格式錯誤，原始回答：{response.text}"
    except Exception as e: 
        return None, None, f"系統連線錯誤：{str(e)}"

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
    # 避免直接執行此檔案時報錯
    try: st.set_page_config(page_title="智慧捷運路徑規劃系統", layout="wide")
    except: pass

    st.title("🚆 智慧捷運路徑規劃系統 (高畫質+雙引擎版)")

    selected_map = st.selectbox("🗺️ 選擇路網", list(MAP_DATABASE.keys()))
    config = MAP_DATABASE[selected_map]

    sys_data = load_json_data(config["json"])
    fare_data = load_json_data(config["fare_json"])
    # ✨ 將 fare_func 傳入系統
    mrt = TransitSystem(sys_data, fare_data, config["fare_func"])

    if not sys_data:
        st.error("地圖資料載入失敗")
        return

    # 初始化 Session State
    names = mrt.get_all_display_names()
    if 'start_st' not in st.session_state: st.session_state.start_st = names[0]
    if 'end_st' not in st.session_state: st.session_state.end_st = names[0]
    if 'next_click_is_start' not in st.session_state: st.session_state.next_click_is_start = True
    if 'last_click' not in st.session_state: st.session_state.last_click = None

    # 將地圖區塊比例稍微調大 (1 : 2.5) 以配合高畫質縮放
    col_ui, col_map = st.columns([1, 2.5])

    with col_ui:
        st.subheader("✨ AI 語音/文字助理")
        
        # 🚀 重大修正：使用 st.form 包裝，防止 Streamlit 吃掉按鈕點擊事件
        with st.form(key="ai_form"):
            user_input = st.text_input("你想去哪？", placeholder="例如：從高鐵站搭到愛河之心")
            # form 裡面的按鈕必須使用 st.form_submit_button
            submit_btn = st.form_submit_button("🤖 AI 規劃", use_container_width=True)
            
        if submit_btn:
            with st.spinner("AI 腦力激盪中..."):
                sid_s, sid_e, msg = get_stations_from_ai(user_input, mrt)
                
                if sid_s and sid_e:
                    st.session_state.start_st = mrt.get_station(sid_s).display_name
                    st.session_state.end_st = mrt.get_station(sid_e).display_name
                    st.success("✅ AI 解析成功！")
                    st.rerun() # 畫面更新
                else:
                    # ✨ 如果失敗，絕對會在這裡印出大大的紅色錯誤訊息
                    st.error(f"❌ 解析失敗：{msg}")

        st.divider()
        # ...(下方原本的 idx_s = names.index... 程式碼保持不變)

        st.divider()
        idx_s = names.index(st.session_state.start_st) if st.session_state.start_st in names else 0
        idx_e = names.index(st.session_state.end_st) if st.session_state.end_st in names else 0

        sel_start = st.selectbox("出發站", names, index=idx_s)
        sel_end = st.selectbox("終點站", names, index=idx_e)

        st.session_state.start_st = sel_start
        st.session_state.end_st = sel_end

        # ✨ 加回策略選擇選單
        search_mode = st.radio("⚙️ 選擇路徑規劃策略", ["最少站數 (BFS)", "最省票價 (Dijkstra)"], horizontal=True)

        if st.button("🔍 查詢路徑", type="primary", use_container_width=True):
            if sel_start == sel_end:
                st.warning("起終點相同")
            else:
                id_s, id_e = mrt.get_sid_by_name(sel_start), mrt.get_sid_by_name(sel_end)
                
                # ✨ 根據選擇呼叫不同演算法
                if "最少站數" in search_mode:
                    path = find_shortest_path(mrt, id_s, id_e)
                else:
                    path = find_cheapest_path(mrt, id_s, id_e)
                    
                if path:
                    fare, details = get_fare_and_details(mrt, path)
                    st.success(f"**系統報價：{fare} 元** | **總站數：{len(path)} 站**")
                    st.text_area("路徑詳情", details, height=130)
                    
                    # 顯示連串的路徑建議
                    path_display = " ➔ ".join([f"[{mrt.get_station(i).line_type}] {mrt.get_station(i).name}" for i in path])
                    st.info(f"建議路徑：\n{path_display}")
                    
                    audio = generate_speech_audio(sel_start, sel_end, fare)
                    st.audio(audio, format="audio/mp3", autoplay=True)
                else:
                    st.error("找不到相連路徑")

    with col_map:
        st.subheader("🗺️ 互動地圖")
        if st.session_state.next_click_is_start: st.info("👆 請在地圖點擊 **出發站**")
        else: st.warning("👆 請在地圖點擊 **終點站**")

        try:
            # 1. 讀取原始高畫質圖片
            img = Image.open(config["img"])
            w_orig, h_orig = img.size

            # 2. 設定我們想要的固定網頁顯示寬度
            TARGET_WIDTH = 450
            scale_ratio = TARGET_WIDTH / w_orig  # 依然保留縮放比例，用來校正座標

            # 3. 直接把原圖丟進去，用內建的 width 參數讓網頁前端自動高畫質縮放
            click = streamlit_image_coordinates(
                img, 
                width=TARGET_WIDTH, 
                key="map_click"
            )

            if click:
                cx, cy = click["x"], click["y"]
                if st.session_state.last_click != (cx, cy):
                    st.session_state.last_click = (cx, cy)
                    closest, min_dist = None, float('inf')

                    for s in mrt.stations.values():
                        # 將 JSON 的原始座標乘上縮放比例，對齊目前的畫面
                        scaled_x = s.coords[0] * scale_ratio
                        scaled_y = s.coords[1] * scale_ratio

                        dist = math.sqrt((cx - scaled_x)**2 + (cy - scaled_y)**2)
                        if dist < min_dist: 
                            min_dist, closest = dist, s

                    # 容錯距離也跟著縮放
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

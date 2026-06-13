import sys
import streamlit as st
import cv2
import numpy as np
from collections import deque
import tempfile
import os
import uuid
import time

# ======== 全域定時自動清理機制 (15分鐘 = 900秒) ========
def safe_auto_cleanup(max_age_seconds=900):
    """
    定期檢查並清除超過指定存活時間的暫存檔案。
    透過比對檔案最後修改時間與當前時間，避免誤刪正在處理中的檔案。
    """
    temp_dir = tempfile.gettempdir()
    now = time.time()
    existing_files = []
    try:
        for f in os.listdir(temp_dir):
            # 僅針對此專案產生的特定前綴檔案進行清理
            if f.startswith("bmt_in_") or f.startswith("bmt_out_"):
                f_path = os.path.join(temp_dir, f)
                file_age = now - os.path.getmtime(f_path)
                print(f"[FILE TIME] Found project file: {f} | Age: {int(file_age)} seconds")
                existing_files.append(f"File: {f} (Age: {file_age}s)")
                
                # 超過指定秒數未更新則執行刪除
                if now - os.path.getmtime(f_path) > max_age_seconds:
                    os.remove(f_path)
                    print(f"[CACHE CLEANUP] Successfully auto-cleaned expired file: {f}")
    except Exception as global_err:
        print(f"[CACHE GLOBAL ERROR] Failed to read temporary directory: {global_err}")

# 每次調用此腳本時觸發全域快取檢查
safe_auto_cleanup()


# ======== MediaPipe 初始化 (含錯誤捕捉) ========
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except (ImportError, AttributeError) as e:
    try:
        import mediapipe.python.solutions.pose as mp_pose
        import mediapipe.python.solutions.drawing_utils as mp_drawing
        import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
    except Exception as inner_e:
        st.error("MediaPipe 載入失敗！")
        st.warning(f"初步錯誤: {e}")
        st.warning(f"深層錯誤: {inner_e}")
        
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        with st.expander("查看目前安裝套件 (Debug Use)"):
            st.code(result.stdout)
        st.stop()


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calc_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def calculate_speed(prev, curr, fps):
    if prev is None or curr is None:
        return 0
    distance = calc_distance(prev, curr)
    return distance * fps


# ======== Streamlit 網頁介面 ========
st.set_page_config(page_title="羽球姿勢分析", page_icon="🏸")
st.title("羽球動作技術AI 分析系統 öㅅö")

# 初始化 Session 狀態變數
if 'analyzed_path' not in st.session_state:
    st.session_state.analyzed_path = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'current_input_path' not in st.session_state:
    st.session_state.current_input_path = None
# 控制當前選擇的分析模式（None: 等待選擇, 'angle': 角度分析, 'gravity': 重心分析）
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = None

uploaded_file = st.file_uploader("選擇影片檔案...", type=["mp4", "mov", "avi"])

# ======== 使用者更換上傳檔案時，主動清除該會話之前的舊暫存檔 ========
if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
    # 移除舊的輸入來源暫存檔
    if st.session_state.current_input_path and os.path.exists(st.session_state.current_input_path):
        try: os.remove(st.session_state.current_input_path)
        except: pass
    # 移除舊的輸出分析結果檔
    if st.session_state.analyzed_path and os.path.exists(st.session_state.analyzed_path):
        try: os.remove(st.session_state.analyzed_path)
        except: pass
            
    st.session_state.analyzed_path = None
    st.session_state.current_input_path = None
    st.session_state.analysis_mode = None  # 切換檔案時重置分析模式
    st.session_state.last_uploaded_file = uploaded_file.name


if uploaded_file is not None:
    if st.session_state.analyzed_path is None:
        
        st.info("影片上傳成功，請選擇您要執行的 AI 分析項目：")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("啟動角度分析", use_container_width=True):
                st.session_state.analysis_mode = "angle"
        with col2:
            if st.button("啟動重心分析", use_container_width=True):
                st.session_state.analysis_mode = "gravity"
        
        # 當使用者點擊按鈕變更模式後，才執行影像處理迴圈
        if st.session_state.analysis_mode is not None:
            # 使用 UUID 產生唯一識別碼，避免多使用者並行時發生檔名衝突
            user_uuid = uuid.uuid4().hex
            
            with tempfile.NamedTemporaryFile(delete=False, prefix=f"bmt_in_{user_uuid}_", suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                input_path = tfile.name
            
            st.session_state.current_input_path = input_path

            cap = cv2.VideoCapture(input_path)

            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 根據影片原始長寬比，動態調整輸出尺寸，防止直向或非標準比例影片變形失真
            max_dim = 1080
            if orig_width > orig_height: # 橫向
                target_w = max_dim
                target_h = int(max_dim * (orig_height / orig_width))
            else: # 直向
                target_h = max_dim
                target_w = int(max_dim * (orig_width / orig_height))

            # 輸出路徑同樣綁定該使用者的 UUID
            output_path = os.path.join(tempfile.gettempdir(), f"bmt_out_{user_uuid}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

            progress_bar = st.progress(0)
            status_text = st.empty()

            trajectory = deque(maxlen=30)
            angle_history = deque(maxlen=30)
            # 專門用於儲存與顯示重心位移軌跡線的容器
            gravity_trajectory = deque(maxlen=60)
            prev_wrist = None

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (target_w, target_h))
                    h, w = frame.shape[:2]
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark

                        # ------------------ 模式一：角度分析 ------------------
                        if st.session_state.analysis_mode == "angle":
                            shoulder = [
                                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
                            ]
                            elbow = [
                                lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                                lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h
                            ]
                            wrist = [
                                lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                                lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h
                            ]

                            # 關節角度計算與視覺化標記
                            angle = calculate_angle(shoulder, elbow, wrist)
                            angle_history.append(angle)
                            color = (0, 255, 0)
                            text = f"Elbow angle: {int(angle)} deg"
                            if angle < 100:
                                color = (0, 0, 255)
                                text += " (Too bent)"
                            elif angle > 165:
                                color = (0, 165, 255)
                                text += " (Too straight)"
                            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                            # 動作一致性評估
                            if len(angle_history) >= 5:
                                std_angle = np.std(angle_history)
                                consistency_score = max(0, 100 - std_angle)
                            else:
                                consistency_score = 100
                            cv2.putText(frame, f"Consistency: {int(consistency_score)}%", (30, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                            # 繪製骨架節點與軌跡
                            mp_drawing.draw_landmarks(
                                frame,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )

                            trajectory.append(tuple(map(int, wrist)))
                            for i in range(1, len(trajectory)):
                                if trajectory[i - 1] is None or trajectory[i] is None:
                                    continue
                                speed = calculate_speed(trajectory[i - 1], trajectory[i], fps)
                                speed_norm = np.clip(speed / 50.0, 0, 1)
                                line_color = (int(255 * speed_norm), int(255 * (1 - speed_norm)), 0)
                                cv2.line(frame, trajectory[i - 1], trajectory[i], line_color, 4)

                            # 揮拍速度估算
                            current_speed = calculate_speed(prev_wrist, wrist, fps)
                            cv2.putText(frame, f"Swing speed: {current_speed:.1f} px/s", (30, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            prev_wrist = wrist

                        # ------------------ 模式二：重心分析 (核心四點) ------------------
                        elif st.session_state.analysis_mode == "gravity":
                            # 讀取左肩、右肩、左髖、右髖的二維像素座標
                            ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                            rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                            lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
                            rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

                            # 運用幾何中心公式計算核心重心點 (cx, cy)
                            cx = int((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
                            cy = int((ls[1] + rs[1] + lh[1] + rh[1]) / 4)
                            gravity_trajectory.append((cx, cy))

                            # 繪製全身體重幾何重心點 (紅色半徑7實心圓)
                            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)

                            # 繪製步伐位移之運動軌跡連續線段 (綠色粗度3)
                            for i in range(1, len(gravity_trajectory)):
                                if gravity_trajectory[i - 1] is None or gravity_trajectory[i] is None:
                                    continue
                                cv2.line(frame, gravity_trajectory[i - 1], gravity_trajectory[i], (0, 255, 0), 3)

                            # 於影像左上方渲染功能模式提示文字
                            cv2.putText(frame, "Body Gravity Center Tracking", (30, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    out.write(frame)
                    count += 1
                    if frame_count > 0:
                        progress_bar.progress(min(count / frame_count, 1.0))
                    status_text.text(f"正在分析... {count}/{frame_count} 幀")

            cap.release()
            out.release()

            st.session_state.analyzed_path = output_path
            st.rerun()

    else:
        if st.session_state.analyzed_path and os.path.exists(st.session_state.analyzed_path):
            st.success("分析完成！可以點擊下方按鈕下載囉 (σ′▽‵)′▽‵)σ")
            
            with open(st.session_state.analyzed_path, "rb") as file:
                st.download_button(
                    label="下載分析結果影片",
                    data=file,
                    file_name="badminton_analysis.mp4",
                    mime="video/mp4"
                )

            if st.button("重新分析該影片"):
                if st.session_state.current_input_path and os.path.exists(st.session_state.current_input_path):
                    try: 
                        os.remove(st.session_state.current_input_path)
                        print(f"[USER ACTION] Successfully deleted input file: {os.path.basename(st.session_state.current_input_path)}")
                    except Exception as e: 
                        print(f"[USER ACTION ERROR] Input file locked: {e}")
                if st.session_state.analyzed_path and os.path.exists(st.session_state.analyzed_path):
                    try: 
                        os.remove(st.session_state.analyzed_path)
                        print(f"[USER ACTION] Successfully deleted output file: {os.path.basename(st.session_state.analyzed_path)}")
                    except Exception as e: 
                        print(f"[USER ACTION ERROR] Output file locked: {e}")
                    
                st.session_state.analyzed_path = None
                st.session_state.current_input_path = None
                st.session_state.analysis_mode = None  # 重置分析模式
                st.rerun()
                
        else:
            st.error("該影片已超過快取存活時間，暫存檔案已被系統回收，請重新分析。")
            
            st.session_state.analyzed_path = None
            st.session_state.current_input_path = None
            st.session_state.analysis_mode = None
            
            if st.button("返回上傳介面重新上傳"):
                st.rerun()
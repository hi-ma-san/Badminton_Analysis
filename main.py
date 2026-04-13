import traceback

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tempfile
import os

# ======== MediaPipe 初始化 (增加錯誤捕捉) ========
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
except AttributeError:
    # 如果 mp 沒有 solutions 屬性，通常是因為安裝不完全或版本衝突
    st.error("MediaPipe 結構異常")
    st.stop()
except Exception as e:
    st.error("MediaPipe 載入失敗！")
    st.warning(f"具體錯誤: {type(e).__name__}: {e}")
    with st.expander("查看詳細錯誤追蹤 (Traceback)"):
        st.code(traceback.format_exc())
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

if 'analyzed_path' not in st.session_state:
    st.session_state.analyzed_path = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

uploaded_file = st.file_uploader("選擇影片檔案...", type=["mp4", "mov", "avi"])

if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
    st.session_state.analyzed_path = None
    st.session_state.last_uploaded_file = uploaded_file.name

if uploaded_file is not None:
    if st.session_state.analyzed_path is None:
        # 使用更安全的方式處理暫存檔
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            input_path = tfile.name

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(tempfile.gettempdir(), "analyzed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 640))

        progress_bar = st.progress(0)
        status_text = st.empty()

        # ======== 與 Mediapipe w.py 相同的分析狀態 ========
        trajectory = deque(maxlen=30)
        angle_history = deque(maxlen=30)
        prev_wrist = None

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (1080, 640))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    h, w = frame.shape[:2]

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

                    # Elbow angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    angle_history.append(angle)
                    color = (0, 255, 0)
                    text = f"Elbow angle: {int(angle)}°"
                    if angle < 100:
                        color = (0, 0, 255)
                        text += " (Too bent)"
                    elif angle > 165:
                        color = (0, 165, 255)
                        text += " (Too straight)"
                    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Consistency
                    if len(angle_history) >= 5:
                        std_angle = np.std(angle_history)
                        consistency_score = max(0, 100 - std_angle)
                    else:
                        consistency_score = 100
                    cv2.putText(frame, f"Consistency: {int(consistency_score)}%", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    # Draw landmarks & trajectory
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

                    # Swing speed
                    current_speed = calculate_speed(prev_wrist, wrist, fps)
                    cv2.putText(frame, f"Swing speed: {current_speed:.1f} px/s", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    prev_wrist = wrist

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
        st.success("分析完成！可以點擊下方按鈕下載囉 (σ′▽‵)′▽‵)σ")
        with open(st.session_state.analyzed_path, "rb") as file:
            st.download_button(
                label="下載分析結果影片",
                data=file,
                file_name="badminton_analysis.mp4",
                mime="video/mp4"
            )

        if st.button("重新分析該影片"):
            st.session_state.analyzed_path = None
            st.rerun()
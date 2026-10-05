import streamlit as st
import pandas as pd
import plotly.express as px
import time

from src.vision_tracker import get_visual_features, cap
from src.system_tracker import get_system_features
from src.fuzzy_engine import evaluate_behavior

st.set_page_config(page_title="FocusGuard Live Monitor", layout="wide")
st.title("🛡️ FocusGuard: Real-Time Focus & Distraction Monitor")
st.markdown("Live multimodal session tracking (Webcam + Telemetry + Fuzzy Inference)")

# Initialize session state memory for live observations
if "session_running" not in st.session_state:
    st.session_state["session_running"] = False
if "session_data" not in st.session_state:
    st.session_state["session_data"] = []

# Sidebar Controls
st.sidebar.header("Session Settings")
session_goal = st.sidebar.text_input("Enter Your Goal:", value="DSA Practice")

col_btn1, col_btn2 = st.sidebar.columns(2)
start_clicked = col_btn1.button("▶️ Start Live", type="primary", use_container_width=True)
stop_clicked = col_btn2.button("⏹️ Stop & Analyze", use_container_width=True)

if start_clicked:
    st.session_state["session_running"] = True
    st.session_state["session_data"] = []  # Clear previous run
    st.rerun()

if stop_clicked:
    st.session_state["session_running"] = False
    st.rerun()

# ----------------------------------------------------
# 1. LIVE MONITORING VIEW (While Active)
# ----------------------------------------------------
if st.session_state["session_running"]:
    st.info(f"🟢 **Monitoring Active** for Goal: `{session_goal}`. Press **'Stop & Analyze'** in the sidebar when done.")
    
    col_cam, col_live = st.columns([1.2, 1])
    cam_box = col_cam.empty()
    stats_box = col_live.empty()

    # Continuous live capture loop
    while st.session_state["session_running"]:
        start_time = time.time()
        
        # 1. Capture real-time visual and system signals
        vis, frame = get_visual_features()
        sys = get_system_features()
        timestamp = time.strftime("%H:%M:%S")

        obs = {
            "timestamp": timestamp,
            "session_goal": session_goal,
            "head_deviated": vis["head_deviated"],
            "phone_detected": vis["phone_detected"],
            "keystrokes": sys["keystroke_count"],
            "mouse_clicks": sys["mouse_click_count"],
            "idle_seconds": sys["idle_seconds"],
            "active_window": str(sys["active_window"])
        }

        # 2. Evaluate with Fuzzy Inference
        scores = evaluate_behavior(obs)
        obs["focus_score"] = scores["focus_score"]
        obs["distraction_score"] = scores["distraction_score"]
        obs["uncertain_score"] = scores["uncertain_score"]

        # Determine dominant state
        state_map = {
            "Focus-Compatible": scores["focus_score"],
            "Distraction-Like": scores["distraction_score"],
            "Uncertain": scores["uncertain_score"]
        }
        obs["State"] = max(state_map, key=state_map.get)

        # Save observation to in-memory session list
        st.session_state["session_data"].append(obs)

        # 3. Stream camera feed
        if frame is not None:
            frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            cam_box.image(frame_rgb, caption="Live Camera Stream", use_column_width=True)

        # 4. Display live diagnostic cards
        with stats_box.container():
            st.markdown(f"**Active Window:** `{obs['active_window'][:35]}`")
            st.markdown(f"**Current State:** `{obs['State']}`")
            st.markdown(f"**Phone Detected:** `{'Yes' if obs['phone_detected'] else 'No'}` | **Head Away:** `{'Yes' if obs['head_deviated'] else 'No'}`")
            st.markdown(f"**Idle Time:** `{obs['idle_seconds']}s`")
            st.markdown("---")
            st.progress(scores["focus_score"], text=f"Focus: {int(scores['focus_score'] * 100)}%")
            st.progress(scores["distraction_score"], text=f"Distraction: {int(scores['distraction_score'] * 100)}%")
            st.progress(scores["uncertain_score"], text=f"Uncertain (Reading): {int(scores['uncertain_score'] * 100)}%")

        # Observation interval (e.g. every 2 seconds for responsive live feedback)
        elapsed = time.time() - start_time
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)

# ----------------------------------------------------
# 2. RESULTS & ANALYTICS DASHBOARD (After Stop Clicked)
# ----------------------------------------------------
else:
    data = st.session_state.get("session_data", [])
    
    if len(data) == 0:
        st.info("👈 Set your goal in the sidebar and click **'Start Live'** to begin monitoring your work.")
    else:
        df = pd.DataFrame(data)
        st.success("🎉 **Session Completed!** Here is your analyzed focus summary:")

        total_steps = len(df)
        focus_pct = round((df["State"] == "Focus-Compatible").sum() / total_steps * 100, 1)
        distract_pct = round((df["State"] == "Distraction-Like").sum() / total_steps * 100, 1)
        uncertain_pct = round((df["State"] == "Uncertain").sum() / total_steps * 100, 1)

        total_seconds = total_steps * 2
        mins = total_seconds // 60
        secs = total_seconds % 60

        # Summary Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Task-Compatible", f"{focus_pct}%")
        col2.metric("Distraction-Like", f"{distract_pct}%")
        col3.metric("Uncertain (Reading/Thinking)", f"{uncertain_pct}%")
        col4.metric("Session Duration", f"{mins}m {secs}s")

        st.markdown("---")
        distracted_rows = df[df["State"] == "Distraction-Like"]
        top_distraction = distracted_rows["active_window"].value_counts().index[0] if not distracted_rows.empty else "None"
        st.info(f"**Primary Distracting Window/Application:** `{top_distraction}`")

        # Interactive Timeline Graph
        st.subheader("Your Session Behavioral Timeline")
        fig = px.scatter(
            df,
            x="timestamp",
            y="State",
            color="State",
            color_discrete_map={
                "Focus-Compatible": "#2ecc71",
                "Distraction-Like": "#e74c3c",
                "Uncertain": "#f1c40f"
            },
            hover_data=["active_window", "idle_seconds", "phone_detected"]
        )
        fig.update_traces(marker=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Full Recorded Intervals"):
            st.dataframe(df)
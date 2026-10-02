# Merges vision + system logs into one CSV


import time
import os
import pandas as pd
from vision_tracker import get_visual_features, cap
from system_tracker import get_system_features

# 1. User sets the task context before starting the session
print("=== FocusGuard Data Collector ===")
session_goal = input("Enter your session goal (e.g., DSA Practice, Web Dev, Reading): ").strip()
if not session_goal:
    session_goal = "DSA Practice"

print(f"\n[INFO] Starting recording session for: '{session_goal}'")
print("[INFO] Focus on your work. Press 'Ctrl + C' in this terminal when finished.\n")

records = []
interval = 5  # Capture observation every 5 seconds

try:
    while True:
        start_time = time.time()
        
        # Pull visual metrics (from camera/YOLO)[cite: 2, 4]
        vis_features, _ = get_visual_features()
        
        # Pull system metrics (keyboard, mouse, active window)[cite: 2, 4]
        sys_features = get_system_features()
        
        # Merge all into one record row[cite: 4, 10]
        timestamp = time.strftime("%H:%M:%S")
        row = {
            "timestamp": timestamp,
            "session_goal": session_goal,
            "head_deviated": vis_features["head_deviated"],
            "phone_detected": vis_features["phone_detected"],
            "keystroke_count": sys_features["keystroke_count"],
            "mouse_click_count": sys_features["mouse_click_count"],
            "idle_seconds": sys_features["idle_seconds"],
            "active_window": sys_features["active_window"]
        }
        
        records.append(row)
        print(f"[{timestamp}] Logged: Phone={row['phone_detected']} | Away={row['head_deviated']} | Idle={row['idle_seconds']}s | Window='{row['active_window'][:25]}...'")
        
        # Wait remainder of the 5-second interval
        elapsed = time.time() - start_time
        if elapsed < interval:
            time.sleep(interval - elapsed)

except KeyboardInterrupt:
    print("\n[INFO] Stopping session and saving data...")
    
    # Release camera
    cap.release()
    
    # Save to CSV
    os.makedirs("data/processed", exist_ok=True)
    file_name = f"data/processed/session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    
    df = pd.DataFrame(records)
    df.to_csv(file_name, index=False)
    
    print(f"[SUCCESS] Saved {len(df)} records to '{file_name}'!")
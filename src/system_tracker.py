# Member 2: Keystroke, mouse, & window logger

import time
from pynput import mouse, keyboard
import pygetwindow as gw

keystrokes = 0
mouse_clicks = 0
last_active_time = time.time()

def on_press(key):
    global keystrokes, last_active_time
    keystrokes += 1
    last_active_time = time.time()

def on_click(x, y, button, pressed):
    global mouse_clicks, last_active_time
    if pressed:
        mouse_clicks += 1
        last_active_time = time.time()

# Start background listeners for keyboard and mouse
keyboard_listener = keyboard.Listener(on_press=on_press)
mouse_listener = mouse.Listener(on_click=on_click)

keyboard_listener.daemon = True
mouse_listener.daemon = True

keyboard_listener.start()
mouse_listener.start()

def get_system_features():
    global keystrokes, mouse_clicks
    
    # Calculate idle time in seconds
    idle_time = round(time.time() - last_active_time, 2)
    
    # Safely get current active window title on Windows
    try:
        active_window_obj = gw.getActiveWindow()
        active_title = active_window_obj.title if active_window_obj else "Unknown"
    except Exception:
        active_title = "Unknown"
        
    data = {
        "keystroke_count": keystrokes,
        "mouse_click_count": mouse_clicks,
        "idle_seconds": idle_time,
        "active_window": active_title
    }
    
    # Reset interaction counts for the next observation window
    keystrokes = 0
    mouse_clicks = 0
    return data

# Standalone test loop to verify background tracking
if __name__ == "__main__":
    print("Testing system activity tracker. Press Ctrl+C in terminal to stop.")
    try:
        while True:
            time.sleep(3)  # Print stats every 3 seconds
            stats = get_system_features()
            print(f"Stats: {stats}")
    except KeyboardInterrupt:
        print("\nTracker stopped.")
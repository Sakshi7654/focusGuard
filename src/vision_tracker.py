# Member 1: MediaPipe & Phone detection

import cv2
from ultralytics import YOLO

# Standard official YOLO model (downloads yolov8n.pt automatically ~6MB)[cite: 5]
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

def get_visual_features():
    ret, frame = cap.read()
    if not ret:
        return {"head_deviated": 0, "phone_detected": 0}, None

    phone_detected = 0
    person_detected = 0
    head_deviated = 0

    # Run inference on current frame
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        
        # Class 67: Cell phone[cite: 2, 4]
        if cls_id == 67:
            phone_detected = 1
            
        # Class 0: Person (User sitting in front of camera)[cite: 2, 4]
        elif cls_id == 0:
            person_detected = 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            person_center_x = (x1 + x2) / 2
            frame_width = frame.shape[1]
            
            # If the user shifts far off to either screen boundary[cite: 14]
            if person_center_x < (0.20 * frame_width) or person_center_x > (0.80 * frame_width):
                head_deviated = 1

    # If the user steps away or leaves camera view completely[cite: 5, 14]
    if person_detected == 0:
        head_deviated = 1

    features = {"head_deviated": head_deviated, "phone_detected": phone_detected}
    return features, frame

if __name__ == "__main__":
    print("Testing vision tracking. Press 'q' on the video window to quit.")
    while cap.isOpened():
        features, frame = get_visual_features()
        if frame is None:
            break

        status = f"Away/Deviated: {features['head_deviated']} | Phone: {features['phone_detected']}"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("FocusGuard - Vision Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
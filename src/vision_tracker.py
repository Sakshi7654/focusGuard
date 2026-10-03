# Member 1: MediaPipe & Phone detection

import cv2
import mediapipe as mp
from ultralytics import YOLO

# Upgraded to yolov8s.pt (Small) for much higher phone detection accuracy
model = YOLO("yolov8s.pt")

# Setup MediaPipe Face Mesh for head deviation detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def get_visual_features():
    ret, frame = cap.read()
    if not ret:
        return {"head_deviated": 0, "phone_detected": 0}, None

    phone_detected = 0
    head_deviated = 0

    # 1. Run YOLO inference with lowered confidence threshold and 640px image size
    results = model(frame, conf=0.10, imgsz=640, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Draw blue bounding box and class name for all detected objects
        label = f"{model.names[cls_id]}: {conf_score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Class 67: Cell phone
        if cls_id == 67:
            phone_detected = 1
            # Highlight detected phone with a thick green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 2. Run MediaPipe Face Mesh to detect head angle/looking away
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            # Key landmarks: Nose tip (1), Left edge of face (234), Right edge of face (454)
            nose = face_landmarks.landmark[1]
            left_face = face_landmarks.landmark[234]
            right_face = face_landmarks.landmark[454]

            nose_x = nose.x * w
            left_x = left_face.x * w
            right_x = right_face.x * w

            # Relative position of nose across the face width
            face_width = right_x - left_x
            if face_width > 0:
                relative_nose_pos = (nose_x - left_x) / face_width

                # If the nose moves too far left (< 0.30) or right (> 0.70), head is turned
                if relative_nose_pos < 0.30 or relative_nose_pos > 0.70:
                    head_deviated = 1
    else:
        # No face detected in frame (user looked away completely or stepped away)
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
        color = (0, 0, 255) if (features['head_deviated'] or features['phone_detected']) else (0, 255, 0)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("FocusGuard - Vision Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# 1. Initialize YOLO
model = YOLO("yolov8s.pt")

# 2. Setup MediaPipe Tasks Face Landmarker
model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def get_visual_features():
    ret, frame = cap.read()
    if not ret:
        return {"head_deviated": 0, "phone_detected": 0}, None

    phone_detected = 0
    head_deviated = 0

    # 1. YOLO inference
    results = model(frame, conf=0.10, imgsz=640, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        label = f"{model.names[cls_id]}: {conf_score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Class 67: Cell phone
        if cls_id == 67:
            phone_detected = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 2. Face Landmarker inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape

        nose = face_landmarks[1]
        left_face = face_landmarks[234]
        right_face = face_landmarks[454]

        nose_x = nose.x * w
        left_x = left_face.x * w
        right_x = right_face.x * w

        face_width = right_x - left_x
        if face_width > 0:
            relative_nose_pos = (nose_x - left_x) / face_width
            if relative_nose_pos < 0.30 or relative_nose_pos > 0.70:
                head_deviated = 1
    else:
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
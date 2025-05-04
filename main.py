import cv2
import face_recognition
import pygame
import torch
from ultralytics import YOLO
import time
import torch.nn as nn
import os
from dotenv import load_dotenv

load_dotenv()  # read .env into os.environ

# read settings
YOLO_MODEL_PATH   = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
ADMIN_IMAGE_DIR   = os.getenv("ADMIN_IMAGE_DIR", "admin_images")
BUZZER_AUDIO      = os.getenv("BUZZER_AUDIO", "buzzer.wav")
MOTION_THRESHOLD  = int(os.getenv("MOTION_THRESHOLD", 25))
MOTION_PIXELS     = int(os.getenv("MOTION_PIXELS", 5000))
VIDEO_SOURCE      = int(os.getenv("VIDEO_SOURCE", 0))
COOLDOWN_SECONDS  = float(os.getenv("COOLDOWN_SECONDS", 3))
LAST_ADMIN_TIME   = 0

pygame.mixer.init()
pygame.mixer.music.load(BUZZER_AUDIO)

def play_buzzer():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def hook_fn(module, input, output):
    features.append(output.detach().cpu())

# Function to call when unknown human is detected
def handle_unknown_person():
    global LAST_ADMIN_TIME
    if time.time() - LAST_ADMIN_TIME > COOLDOWN_SECONDS:
        print("⚠️ Unknown person detected!")
        play_buzzer()
    else:
        print("⏱️ Buzzer skipped due to recent admin detection.")

# Motion detector using frame differencing
def detect_motion(prev_frame, curr_frame, threshold=25):
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh) > MOTION_PIXELS

# Load your human detection model (YOLOv8 preferably)
model = YOLO(YOLO_MODEL_PATH)

# Prepare storage for hooked features
features = []
# Select a convolutional feature layer
conv_layers = [m for m in model.model.modules() if isinstance(m, nn.Conv2d)]
feature_layer = conv_layers[-2]  # Second to last convolutional layer
hook_handle = feature_layer.register_forward_hook(hook_fn)
print(f"✅ Hooked layer: {feature_layer}")

# Load all images in ADMIN_IMAGE_DIR:
admin_encodings = []
for fname in os.listdir(ADMIN_IMAGE_DIR):
    if fname.lower().endswith((".jpg",".jpeg",".png")):
        img = face_recognition.load_image_file(os.path.join(ADMIN_IMAGE_DIR, fname))
        encs = face_recognition.face_encodings(img)
        if encs:
            admin_encodings.append(encs[0])
if not admin_encodings:
    raise RuntimeError("No admin face encodings found in " + ADMIN_IMAGE_DIR)


# Main loop
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, prev_frame = cap.read()

while True:
    ret, frame = cap.read()
    annotated_frame = frame.copy()
    if not ret:
        break

    if detect_motion(prev_frame, frame, MOTION_THRESHOLD):
        print("📹 Motion detected!")
        # Detect objects
        results = model(frame)

        # Draw YOLO detections
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle & label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for general objects
            cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        boxes = results[0].boxes
        humans = [box for box in boxes if model.names[int(box.cls[0])] == 'person']

        # Visualize feature heatmap
        if features:
            feat_map = features[0][0]
            embedding = torch.mean(feat_map.view(feat_map.size(0), -1), dim=1)
            activation = torch.mean(feat_map, dim=0).numpy()
            act_norm = cv2.normalize(activation, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            act_blur = cv2.GaussianBlur(act_norm, (7, 7), 0)
            act_resized = cv2.resize(act_blur, (frame.shape[1], frame.shape[0]))
            heatmap = cv2.applyColorMap(act_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            cv2.imshow('Feature Heatmap Overlay', overlay)

        if humans:
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                match_found = False
                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    match = face_recognition.compare_faces(admin_encodings, encoding)
                    if any(match):
                        color = (0, 255, 0)  # Green for admin
                        label = "Admin"
                        LAST_ADMIN_TIME = time.time()
                        match_found = True
                        print("✅ Admin detected.")
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        label = "Unknown"

                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(annotated_frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if not match_found:
                    handle_unknown_person()
            else:
                print("🧍 Human detected but no face visible.")
                handle_unknown_person()

    prev_frame = frame.copy()

    cv2.imshow("Live", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

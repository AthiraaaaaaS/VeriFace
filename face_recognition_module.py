import cv2
import numpy as np
import sqlite3
from utils import get_known_faces, save_attendance
from numpy import dot
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# Load InsightFace model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)


face_net = cv2.dnn.readNetFromTensorflow("models/opencv_face_detector.pb", "models/opencv_face_detector.pbtxt")

arcface_net = cv2.dnn.readNetFromONNX("models/arcface.onnx")

def detect_faces_dnn(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((y1, x2, y2, x1))  # top, right, bottom, left (to match dlib style)
    return boxes

def extract_faces_and_embeddings(frame):
    """
    Detects faces and returns their embeddings with bounding boxes.
    """
    results = app.get(frame)
    detections = []
    for face in results:
        embedding = face.embedding
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        detections.append((embedding, (x1, y1, x2, y2)))
    return detections

def recognize_embedding(embedding):
    known_names, known_encodings, user_ids = get_known_faces()
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    best_match_index = -1
    best_similarity = -1

    for idx, known_encoding in enumerate(known_encodings):
        known_encoding = known_encoding / np.linalg.norm(known_encoding)
        similarity = cosine_similarity(known_encoding, embedding)
        print(f"🔹 Comparing with {known_names[idx]}: Similarity = {similarity}")

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_index = idx

    if best_similarity > 0.6:
        save_attendance(user_ids[best_match_index])
        print(f"Attendance marked for {known_names[best_match_index]} with similarity {best_similarity}")
        conn.close()
        return known_names[best_match_index]
    else:
        conn.close()
        return "Unknown"


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

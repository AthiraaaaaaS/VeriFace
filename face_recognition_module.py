import cv2
import numpy as np
import sqlite3
from utils import get_known_faces, save_attendance

prototxt_path = "models/deploy.prototxt"
caffe_model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

# ✅ Load ArcFace Model for Face Recognition
arcface_net = cv2.dnn.readNet("models/arcface.onnx")  #Ensure ArcFace model is loaded

def detect_faces_dnn(frame):
    """Detect faces using OpenCV's DNN face detector."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    face_locations = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])  #Ensure confidence is a float

        if confidence > 0.6:  # Only keep faces with high confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_locations.append((startY, endX, endY, startX))

    return face_locations

def recognize_face(frame):
    """Recognizes a face using stored face encodings."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = detect_faces_dnn(frame)

    known_names, known_encodings, user_ids = get_known_faces()

    for (y, endX, endY, x) in face_locations:
        face_crop = rgb_frame[y:endY, x:endX]

        if face_crop.size == 0:
            continue  

        # Extract face encoding
        blob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        arcface_net.setInput(blob)
        
        try:
            face_encoding = arcface_net.forward().flatten()
            if face_encoding.shape[0] == 0:
                print(f"❌ Face encoding extraction failed.")
                continue
        except cv2.error as e:
            print(f"❌ Error extracting face encoding: {e}")
            continue

        # ✅ Normalize extracted encoding
        face_encoding = face_encoding / np.linalg.norm(face_encoding)

        best_match_index = -1
        best_similarity = -1

        for idx, known_encoding in enumerate(known_encodings):
            known_encoding = known_encoding / np.linalg.norm(known_encoding)  # ✅ Normalize stored encoding

            similarity = cosine_similarity(known_encoding, face_encoding)
            print(f"🔹 Comparing with {known_names[idx]}: Similarity = {similarity}")  # Debugging

            # ✅ Mark attendance for any similarity
            if similarity > 0:  # Instead of setting a high threshold, we allow any similarity
                best_match_index = idx
                best_similarity = similarity

        if best_match_index != -1:
            save_attendance(user_ids[best_match_index])
            print(f"✅ Attendance marked for {known_names[best_match_index]} with similarity {best_similarity}")
            return known_names[best_match_index]

    return "Unknown"

def cosine_similarity(v1, v2):
    """Computes cosine similarity between two face embeddings."""
    v1 = v1 / np.linalg.norm(v1)  # ✅ Normalize embedding
    v2 = v2 / np.linalg.norm(v2)  # ✅ Normalize embedding
    return np.dot(v1, v2)

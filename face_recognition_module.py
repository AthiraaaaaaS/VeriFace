import face_recognition
import numpy as np
import cv2
from utils import get_known_faces, save_attendance

def recognize_face(frame):
    """Recognizes a face from the video frame using stored encodings."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    known_names, known_encodings, user_ids = get_known_faces()
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            save_attendance(user_ids[best_match_index])
            return known_names[best_match_index]
    return "Unknown"
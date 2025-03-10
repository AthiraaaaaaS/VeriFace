import face_recognition
import os
import numpy as np
import cv2

def load_known_faces():
    """Load known face images and encode them."""
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir('known_faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(f"known_faces/{filename}")

            # Detect face locations before encoding
            face_locations = face_recognition.face_locations(image)
            encoding = face_recognition.face_encodings(image, face_locations)

            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

def recognize_face(frame, known_face_encodings, known_face_names):
    """Recognizes a face from the video frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        return name

    return None

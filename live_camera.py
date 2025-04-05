import cv2
from face_recognition_module import extract_faces_and_embeddings, recognize_embedding
import cv2
import numpy as np
from face_recognition_module import detect_faces_dnn, arcface_net

def run_live_camera():
    """Runs a continuous live CCTV camera feed while marking attendance."""
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to access camera")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None or frame.size == 0:
            print("Failed to grab valid frame.")
            continue

        face_data = extract_faces_and_embeddings(frame)
        for embedding, bbox in face_data:
            detected_name = recognize_embedding(embedding)
            print(f"Detected: {detected_name}")


        cv2.imshow("Live CCTV Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (112, 112))
    face_normalized = face_resized.astype(np.float32) / 255.0
    face_input = np.transpose(face_normalized, (2, 0, 1))
    face_input = np.expand_dims(face_input, axis=0)
    return face_input

def get_arcface_embedding(face_input):
    arcface_net.setInput(face_input)
    embedding = arcface_net.forward()
    return embedding.flatten()

def extract_arcface_encodings_from_frame(frame):
    """Detects faces in a frame and returns their ArcFace encodings with bounding boxes."""
    face_locations = detect_faces_dnn(frame)
    face_data = []

    for (y1, x2, y2, x1) in face_locations:
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        face_input = preprocess_face(face_img)
        embedding = get_arcface_embedding(face_input)
        face_data.append((embedding, (x1, y1, x2, y2)))

    return face_data

if __name__ == "__main__":
    run_live_camera()

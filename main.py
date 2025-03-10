import cv2
import face_recognition
from face_recognition_module import load_known_faces, recognize_face
from attendance import mark_attendance

def main():
    print("Starting Attendance Monitoring System...")

    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()

    # Open webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        # Recognize face
        name = recognize_face(frame, known_face_encodings, known_face_names)

        if name:
            print(f"Recognized: {name}")
            mark_attendance(name)
        else:
            print("No face recognized.")

        # Display the frame
        cv2.imshow("Attendance System", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

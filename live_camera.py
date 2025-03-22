import cv2
from face_recognition_module import recognize_face

def run_live_camera():
    """Runs a continuous live CCTV camera feed while marking attendance."""
    video_capture = cv2.VideoCapture(0)  # Open webcam

    if not video_capture.isOpened():
        print("❌ Error: Unable to access camera")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            continue

        detected_names = recognize_face(frame)  # Recognize faces continuously

        # Display live CCTV feed
        cv2.imshow("Live CCTV Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_camera()

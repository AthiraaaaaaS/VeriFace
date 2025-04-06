import cv2
import numpy as np
import time
from datetime import datetime
import sqlite3
import os
import sys

# Add the current directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_recognition import FaceRecognizer

def run_live_camera():
    """Runs a continuous live CCTV camera feed while marking attendance."""
    # Initialize face recognizer
    recognizer = FaceRecognizer()
    
    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    if not video_capture.isOpened():
        print("Error: Unable to access camera")
        return
    
    # Dictionary to track recognized users and their last attendance time
    recognized_users = {}
    attendance_cooldown = 60  # seconds between attendance records for the same user
    
    print("Camera started. Press 'q' to quit.")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None or frame.size == 0:
            print("Failed to grab valid frame.")
            continue
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Recognize faces in the frame
        user_id, user_name, confidence = recognizer.recognize_face_from_frame(frame)
        
        # Draw FPS on frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if user_id is not None:
            # Draw bounding box and name
            height, width = frame.shape[:2]
            box_size = min(width, height) // 4
            x1 = (width - box_size) // 2
            y1 = (height - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw name and confidence
            text = f"{user_name} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Record attendance if cooldown period has passed
            current_time = time.time()
            if (user_id not in recognized_users or 
                current_time - recognized_users[user_id] > attendance_cooldown):
                
                if recognizer.record_attendance(user_id):
                    print(f"Attendance recorded for {user_name}")
                    recognized_users[user_id] = current_time
        else:
            # Draw "No face detected" message
            cv2.putText(frame, "No face detected", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("VeriFace - Live Recognition", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")

if __name__ == "__main__":
    run_live_camera()

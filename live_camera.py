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
from insightface.app import FaceAnalysis

def run_live_camera():
    """Runs a continuous live CCTV camera feed while marking attendance."""
    # Initialize InsightFace app with modified parameters
    app = FaceAnalysis(name='buffalo_l', 
                      providers=['CPUExecutionProvider'],
                      allowed_modules=['detection', 'recognition'])
    # Prepare with modified detection parameters
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Initialize face recognizer
    recognizer = FaceRecognizer()
    print("\nInitializing face recognition system...")
    print("Available users:", recognizer.user_names)
    if recognizer.classifier is not None:
        print("Model classes:", recognizer.classifier.classes_)
    else:
        print("No classifier loaded!")
    
    # Initialize camera
    video_capture = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    if not video_capture.isOpened():
        print("Error: Unable to access camera")
        return
    
    # Dictionary to track recognized users and their last attendance time
    recognized_users = {}
    attendance_cooldown = 60  # seconds between attendance records
    
    print("Camera started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret or frame is None:
                continue
            
            try:
                # Get all faces in the frame
                faces = app.get(frame)
                
                if faces:
                    print(f"\nDetected {len(faces)} faces in frame")
                    for face in faces:
                        try:
                            # Get face embedding
                            face_encoding = face.embedding.reshape(1, -1)
                            
                            if recognizer.classifier is not None:
                                # Get predictions and probabilities
                                user_id = recognizer.classifier.predict(face_encoding)[0]
                                probs = recognizer.classifier.predict_proba(face_encoding)[0]
                                confidence = max(probs)
                                user_name = recognizer.user_names.get(user_id, "Unknown")
                                
                                # Print top 3 predictions for debugging
                                top_3_idx = np.argsort(probs)[-3:][::-1]
                                print("\nTop 3 predictions:")
                                for idx in top_3_idx:
                                    pred_id = recognizer.classifier.classes_[idx]
                                    pred_name = recognizer.user_names.get(pred_id, "Unknown")
                                    pred_conf = probs[idx]
                                    print(f"{pred_name} (ID: {pred_id}): {pred_conf:.2f}")
                                
                                if confidence >= recognizer.confidence_threshold:
                                    # Get face box coordinates
                                    bbox = face.bbox.astype(int)
                                    x1, y1, x2, y2 = bbox
                                    
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
                                        
                                        print(f"\nAttempting to record attendance for {user_name} (ID: {user_id})")
                                        success = recognizer.record_attendance(user_id)
                                        if success:
                                            print(f"✅ Successfully recorded attendance for {user_name}")
                                            recognized_users[user_id] = current_time
                                            # Draw success message
                                            cv2.putText(frame, "Attendance Recorded!", 
                                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                                      1, (0, 255, 0), 2)
                                        else:
                                            print(f"❌ Failed to record attendance for {user_name}")
                                            # Draw failure message
                                            cv2.putText(frame, "Failed to Record!", 
                                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                                      1, (0, 0, 255), 2)
                            else:
                                print("No classifier loaded in recognizer")
                        except Exception as e:
                            print(f"Error processing face: {str(e)}")
                            continue
                
                # Display the frame
                cv2.imshow("VeriFace - Live Recognition", frame)
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Camera error: {str(e)}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    run_live_camera()

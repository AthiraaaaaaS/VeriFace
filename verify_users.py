import sqlite3
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_users_and_encodings():
    """
    Verify all users in database and their face encodings.
    Optionally recompute encodings if needed.
    """
    try:
        # Initialize InsightFace
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute("SELECT user_id, name, encoding, email FROM users")
        users = cursor.fetchall()
        
        print(f"\nFound {len(users)} users in database:")
        for user in users:
            user_id, name, encoding_blob, email = user
            print(f"\nUser ID: {user_id}")
            print(f"Name: {name}")
            print(f"Email: {email}")
            
            if encoding_blob:
                # Convert blob to numpy array
                try:
                    encoding = np.frombuffer(encoding_blob, dtype=np.float32)
                    print(f"Encoding shape: {encoding.shape}")
                    print(f"Encoding valid: Yes")
                except Exception as e:
                    print(f"Error with encoding: {e}")
            else:
                print("No encoding found!")
        
        # Check known_faces directory
        print("\nChecking known_faces directory:")
        import os
        if os.path.exists("known_faces"):
            for file in os.listdir("known_faces"):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"\nFound image: {file}")
                    img_path = os.path.join("known_faces", file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        faces = app.get(img)
                        if faces:
                            print(f"Face detected: Yes")
                            print(f"Number of faces: {len(faces)}")
                        else:
                            print(f"No face detected in {file}")
                    else:
                        print(f"Could not read image: {file}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error verifying users: {e}")
        return False

if __name__ == "__main__":
    print("Verifying users and their face encodings...")
    verify_users_and_encodings() 
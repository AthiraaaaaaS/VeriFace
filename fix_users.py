import sqlite3
import os
import shutil
import logging
from insightface.app import FaceAnalysis
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_users_database():
    """
    Completely rebuild the users database:
    1. Backup existing database
    2. Create fresh database
    3. Re-add users with correct IDs
    4. Recompute face encodings
    """
    try:
        # Initialize InsightFace
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        
        # 1. Backup existing database
        if os.path.exists("attendance.db"):
            backup_name = "attendance_backup.db"
            shutil.copy2("attendance.db", backup_name)
            logger.info(f"Created database backup: {backup_name}")
        
        # 2. Create fresh database
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("DROP TABLE IF EXISTS attendance")
        
        # Create fresh tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            encoding BLOB,
            phone TEXT,
            email TEXT
        )
    """)

    # Create attendance table with proper structure
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            user_id TEXT,
            first_seen TEXT,
            last_seen TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    """)
        
        # 3. Re-add users from known_faces directory
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            
        print("\nProcessing users:")
        for file in os.listdir(known_faces_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(file)[0]
                image_path = os.path.join(known_faces_dir, file)
                
                print(f"\nProcessing: {name}")
                print(f"Image: {image_path}")
                
                # Read image and extract face encoding
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Could not read image: {image_path}")
                    continue
                
                faces = app.get(img)
                if not faces:
                    print(f"No face detected in: {image_path}")
                    continue
                
                face_encoding = faces[0].embedding
                face_encoding_blob = face_encoding.astype(np.float32).tobytes()
                
                try:
                    # Add user with minimal info first (can be updated later)
                    cursor.execute("""
                        INSERT INTO users (name, encoding, phone, email)
                        VALUES (?, ?, ?, ?)
                    """, (name, face_encoding_blob, "", ""))
                    
                    user_id = cursor.lastrowid
                    print(f"Added user: {name} (ID: {user_id})")
                    
                except sqlite3.IntegrityError:
                    print(f"User {name} already exists, skipping...")
                    continue
                
        conn.commit()
        
        # Verify the results
        cursor.execute("SELECT user_id, name FROM users ORDER BY user_id")
        users = cursor.fetchall()
        
        print("\nFinal user list:")
        for user_id, name in users:
            print(f"ID: {user_id}, Name: {name}")
        
        conn.close()
        
        print("\n✅ Database rebuilt successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}")
        return False

if __name__ == "__main__":
    print("WARNING: This will rebuild the entire users database!")
    print("1. Your existing database will be backed up")
    print("2. All users will be re-added with fresh IDs")
    print("3. Face encodings will be recomputed")
    
    response = input("\nContinue? (yes/no): ")
    
    if response.lower() == 'yes':
        if rebuild_users_database():
            print("\nNext steps:")
            print("1. Verify the users list above")
            print("2. Run the model training script")
            print("3. Test the face recognition system")
        else:
            print("\n❌ Error during database rebuild. Check the logs above.")
    else:
        print("Operation cancelled.") 
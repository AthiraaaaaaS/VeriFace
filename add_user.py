import cv2
import numpy as np
import sqlite3
import sys
import re
from insightface.app import FaceAnalysis
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def validate_email(email):
    """Validates email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    """Validates phone number format."""
    # Remove any non-digit characters
    digits = re.sub(r'\D', '', phone)
    # Check if we have exactly 10 digits
    return len(digits) == 10

def get_next_user_id():
    """Get the next available user ID in S101+ format."""
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Get the highest existing user_id
        cursor.execute("SELECT user_id FROM users ORDER BY user_id DESC LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            # Extract the number from existing highest ID (e.g., 'S101' -> 101)
            current_num = int(result[0][1:])  # Skip 'S' and convert to int
            next_num = current_num + 1
        else:
            # If no existing users, start with S101
            next_num = 101
            
        conn.close()
        return f"S{next_num}"
        
    except Exception as e:
        logger.error(f"Error getting next user ID: {e}")
        return None

def extract_face_encoding(image_path):
    """Extract face encoding from image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return None
            
        faces = app.get(img)
        if not faces:
            logger.error("No face detected in the image")
            return None
            
        if len(faces) > 1:
            logger.warning("Multiple faces detected in the image. Using the first one.")
            
        return faces[0].embedding
        
    except Exception as e:
        logger.error(f"Error extracting face encoding: {e}")
        return None

def add_user(name, image_path, email, phone):
    """Add a new user to the system."""
    # Validate inputs
    if not name or not name.strip():
        logger.error("Name cannot be empty.")
        return False
        
    if not validate_email(email):
        logger.error(f"Invalid email format: {email}")
        logger.error("Email should be in format: username@domain.com")
        return False
        
    if not validate_phone(phone):
        logger.error("Invalid phone number format.")
        logger.error("Phone number should be 10 digits.")
        return False
        
    # Format phone number (remove non-digits)
    formatted_phone = re.sub(r'\D', '', phone)
    
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        # Create users table if it doesn't exist (with new structure)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                encoding BLOB,
                phone TEXT,
                email TEXT
            )
        """)

        # Check if user already exists
        cursor.execute("SELECT user_id FROM users WHERE name = ? OR email = ?", (name, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            logger.error(f"User with this name or email already exists (ID: {existing_user[0]})")
            return False

        # Get face encoding
        face_encoding = extract_face_encoding(image_path)
        if face_encoding is None:
            logger.error(f"Failed to add user {name}")
            return False

        # Get next available user ID
        user_id = get_next_user_id()
        if user_id is None:
            logger.error("Failed to generate user ID")
            return False

        # Convert encoding to blob
        face_encoding_blob = face_encoding.astype(np.float32).tobytes()
        
        # Insert new user
        cursor.execute("""
            INSERT INTO users (user_id, name, encoding, phone, email) 
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, name, face_encoding_blob, formatted_phone, email))
        
        # Save the image to known_faces directory
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            
        # Copy image to known_faces directory
        image_ext = os.path.splitext(image_path)[1].lower()
        new_image_path = os.path.join(known_faces_dir, f"{name}{image_ext}")
        cv2.imwrite(new_image_path, cv2.imread(image_path))
        
        conn.commit()
        logger.info(f"✅ User {name} added successfully with ID: {user_id}!")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error adding user: {e}")
        return False

def list_users():
    """List all users in the database."""
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT user_id, name, email, phone FROM users ORDER BY user_id")
        users = cursor.fetchall()
        
        if not users:
            print("\nNo users found in database.")
            return
            
        print("\nRegistered Users:")
        print("-" * 70)
        print(f"{'ID':<8} {'Name':<20} {'Email':<25} {'Phone':<15}")
        print("-" * 70)
        
        for user in users:
            print(f"{user[0]:<8} {user[1]:<20} {user[2]:<25} {user[3]:<15}")
            
        print("-" * 70)
        print(f"Total users: {len(users)}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("\nUsage:")
        print("python add_user.py <name> <image_path> <phone> <email>")
        print("\nExample:")
        print('python add_user.py "John Doe" "path/to/image.jpg" "1234567890" "john@example.com"')
        print("\nCurrent users in database:")
        list_users()
        sys.exit(1)
        
    name = sys.argv[1]
    image_path = sys.argv[2]
    phone = sys.argv[3]
    email = sys.argv[4]
    
    if add_user(name, image_path, phone, email):
        print("\nNext steps:")
        print("1. Verify the user was added by checking the list below")
        print("2. Run the model training script (python retrain_model.py)")
        list_users()
    else:
        print("\n❌ Failed to add user. Check the error messages above.")

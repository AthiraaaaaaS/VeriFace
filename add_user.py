import cv2
import numpy as np
import sqlite3
import sys
import re
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def validate_email(email):
    """Validates email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    """Validates phone number format."""
    # Remove any non-digit characters
    phone = re.sub(r'\D', '', phone)
    # Check if it's a valid length (between 10-15 digits)
    return 10 <= len(phone) <= 15

def extract_face_encoding(image_path):
    """Extracts embedding from image using InsightFace."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    faces = app.get(img)
    if not faces:
        print(f"No face detected in image: {image_path}")
        return None

    return faces[0].embedding  # First face

def add_user(name, image_path, phone, email):
    # Validate inputs
    if not name or not name.strip():
        print("❌ Name cannot be empty.")
        return False
        
    if not validate_email(email):
        print("❌ Invalid email format.")
        return False
        
    if not validate_phone(phone):
        print("❌ Invalid phone number format.")
        return False
        
    # Format phone number (remove non-digits)
    formatted_phone = re.sub(r'\D', '', phone)
    
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding BLOB,
            phone TEXT,
            email TEXT
        )
    """)

    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
    existing_user = cursor.fetchone()
    
    if existing_user:
        print(f"⚠️ User {name} already exists. Updating information...")
        cursor.execute("DELETE FROM users WHERE name = ?", (name,))
        conn.commit()

    face_encoding = extract_face_encoding(image_path)
    if face_encoding is None:
        print(f"❌ Failed to add user {name}.")
        return False

    face_encoding_blob = face_encoding.astype(np.float32).tobytes()
    cursor.execute("INSERT INTO users (name, encoding, phone, email) VALUES (?, ?, ?, ?)", 
                   (name, face_encoding_blob, formatted_phone, email))
    conn.commit()
    conn.close()

    print(f"✅ User {name} added successfully!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python add_user.py <name> <image_path> <phone> <email>")
        sys.exit(1)

    name = sys.argv[1]
    image_path = sys.argv[2]
    phone = sys.argv[3]
    email = sys.argv[4]

    add_user(name, image_path, phone, email)

import cv2
import numpy as np
import sqlite3
import sys

# Load ArcFace Model
arcface_net = cv2.dnn.readNet("models/arcface.onnx")

def extract_face_encoding(image_path):
    """Extracts a 512D face embedding using ArcFace model."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Unable to load image at {image_path}")
        return None

    print(f"✅ Image loaded: {image_path}, Shape: {image.shape}")

    # Resize image for ArcFace (Expected input size: 112x112)
    image = cv2.resize(image, (112, 112))
    
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (112, 112), (0, 0, 0), swapRB=True, crop=False)
    arcface_net.setInput(blob)

    try:
        encoding = arcface_net.forward().flatten()
        if encoding.shape[0] != 512:
            print(f"❌ Face encoding shape mismatch: {encoding.shape}")
            return None
    except cv2.error as e:
        print(f"❌ OpenCV error during face encoding extraction: {e}")
        return None

    print(f"✅ Face encoding extracted successfully with shape {encoding.shape}")
    return encoding

def add_user(name, image_path, phone, email):
    """Adds a new user with name, phone number, and email ID."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding BLOB,
            phone TEXT,
            email TEXT
        )
    """)

    # Delete existing user with the same name
    cursor.execute("DELETE FROM users WHERE name = ?", (name,))
    conn.commit()

    face_encoding = extract_face_encoding(image_path)
    if face_encoding is None:
        print(f"Failed to add user {name} due to encoding issues.")
        return

    face_encoding_blob = face_encoding.tobytes()
    cursor.execute("INSERT INTO users (name, encoding, phone, email) VALUES (?, ?, ?, ?)", 
                   (name, face_encoding_blob, phone, email))
    conn.commit()
    conn.close()
    
    print(f"✅ User {name} added successfully with phone {phone} and email {email}!")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python add_user.py <name> <image_path> <phone> <email>")
        sys.exit(1)

    name = sys.argv[1]
    image_path = sys.argv[2]
    phone = sys.argv[3]
    email = sys.argv[4]

    add_user(name, image_path, phone, email)
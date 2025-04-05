import cv2
import numpy as np
import sqlite3
import sys
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

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

    cursor.execute("DELETE FROM users WHERE name = ?", (name,))
    conn.commit()

    face_encoding = extract_face_encoding(image_path)
    if face_encoding is None:
        print(f"❌ Failed to add user {name}.")
        return

    face_encoding_blob = face_encoding.astype(np.float32).tobytes()
    cursor.execute("INSERT INTO users (name, encoding, phone, email) VALUES (?, ?, ?, ?)", 
                   (name, face_encoding_blob, phone, email))
    conn.commit()
    conn.close()

    print(f"User {name} added successfully with phone {phone} and email {email}!")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python add_user.py <name> <image_path> <phone> <email>")
        sys.exit(1)

    name = sys.argv[1]
    image_path = sys.argv[2]
    phone = sys.argv[3]
    email = sys.argv[4]

    add_user(name, image_path, phone, email)

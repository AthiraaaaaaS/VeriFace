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

    # Resize image for ArcFace model (112x112)
    image = cv2.resize(image, (112, 112))

    # Convert to blob & pass through model
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

def add_user(name, image_path):
    """Deletes existing user and adds a new user with ArcFace face encoding."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Ensure `users` table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding BLOB
        )
    """)

    # Delete existing user if already present
    cursor.execute("DELETE FROM users WHERE name = ?", (name,))
    conn.commit()

    # Extract face encoding
    face_encoding = extract_face_encoding(image_path)
    if face_encoding is None:
        print(f"❌ Failed to add user {name} due to encoding issues.")
        return

    # ✅ Store the face encoding properly as BLOB
    face_encoding_blob = np.array(face_encoding, dtype=np.float32).tobytes()
    cursor.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, face_encoding_blob))
    conn.commit()
    conn.close()

    print(f"✅ User {name} added successfully with proper 512D face encoding!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_user.py <name> <image_path>")
        sys.exit(1)

    name = sys.argv[1]
    image_path = sys.argv[2]
    add_user(name, image_path)

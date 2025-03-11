import sqlite3
import face_recognition
import numpy as np

import sqlite3
import face_recognition
import numpy as np

def add_user(name, image_path):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if not encodings:
        print(f"Error: No face detected in {image_path}. Try another image.")
        return

    encoding = encodings[0]  # Get the first face encoding (128 dimensions)

    # Ensure encoding shape is correct before saving
    if encoding.shape[0] != 128:
        print(f"Error: Face encoding for {name} has incorrect shape {encoding.shape}.")
        return

    encoding_bytes = np.array(encoding, dtype=np.float32).tobytes()  # Convert to byte format
    
    cursor.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, encoding_bytes))
    conn.commit()
    conn.close()
    print(f"User {name} added successfully!")

# Example Usage
add_user("Aneesh", "known_faces/aneesh.png")
add_user("Athira", "known_faces/Athira.jpg")

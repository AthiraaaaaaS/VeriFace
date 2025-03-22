import sqlite3
import numpy as np
import pickle
from datetime import datetime  


def save_attendance(user_id):
    """Marks first seen & last seen time for a user in the database."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # ✅ Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # ✅ No more AttributeError

    # ✅ Check if user has already been recorded
    cursor.execute("SELECT first_seen FROM attendance WHERE user_id = ?", (user_id,))
    record = cursor.fetchone()

    if record is None:
        # ✅ First detection → Insert a new row
        cursor.execute("INSERT INTO attendance (user_id, first_seen, last_seen) VALUES (?, ?, ?)",
                       (user_id, current_time, current_time))
    else:
        # ✅ Update last seen time
        cursor.execute("UPDATE attendance SET last_seen = ? WHERE user_id = ?", (current_time, user_id))

    conn.commit()
    conn.close()



def get_known_faces():
    """Fetches stored face encodings and names from the database."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encoding FROM users")
    
    known_names = []
    known_encodings = []
    user_ids = []

    for user_id, name, encoding_blob in cursor.fetchall():
        try:
            encoding_array = np.frombuffer(encoding_blob, dtype=np.float32)  # ✅ Correctly decode BLOB data
            if encoding_array.shape[0] == 512:  # ✅ Ensure correct encoding size
                known_names.append(name)
                known_encodings.append(encoding_array)
                user_ids.append(user_id)
            else:
                print(f"⚠ Warning: Skipping {name} due to incorrect encoding shape {encoding_array.shape}")
        except Exception as e:
            print(f"❌ Error decoding face encoding for {name}: {e}")

    conn.close()
    return known_names, known_encodings, user_ids

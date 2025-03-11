import sqlite3
import numpy as np

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        encoding BLOB NOT NULL)''')
    
    # Create attendance table
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def save_attendance(user_id):
    """Logs attendance for a recognized user."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (user_id, timestamp) VALUES (?, datetime('now'))", (user_id,))
    conn.commit()
    conn.close()

def get_known_faces():
    """Fetch known faces and their encodings from the database."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encoding FROM users")
    known_faces = cursor.fetchall()
    conn.close()
    
    names, encodings, ids = [], [], []
    for user_id, name, encoding in known_faces:
        decoded_encoding = np.frombuffer(encoding, dtype=np.float32)  # Convert BLOB back to NumPy array

        # Ensure encoding has the correct shape (128,)
        if decoded_encoding.shape[0] == 128:
            names.append(name)
            ids.append(user_id)
            encodings.append(decoded_encoding)
        else:
            print(f"Warning: Skipping user {name} due to incorrect encoding shape {decoded_encoding.shape}")

    return names, encodings, ids


# Run this function once to set up the database
init_db()
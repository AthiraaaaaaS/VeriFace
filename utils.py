import sqlite3
import numpy as np
from datetime import datetime  
import os
import sys

# Add the current directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_recognition import FaceRecognizer

WORK_END_TIME = "20:00:00"  # 8 PM

def fetch_attendance(user_phone):
    """Fetch attendance from the database using the user's phone number."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT users.name, attendance.first_seen, attendance.last_seen
        FROM attendance
        INNER JOIN users ON attendance.user_id = users.id
        WHERE users.phone = ? ORDER BY attendance.first_seen DESC LIMIT 1
    """, (user_phone,))
    
    record = cursor.fetchone()
    conn.close()
    
    if record:
        name, first_seen, last_seen = record
        return f"Hello {name}, your attendance:\nFirst Seen: {first_seen}\ Last Seen: {last_seen}"
    else:
        return "No attendance record found for today."

import datetime

def save_attendance(user_id):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    date_today = datetime.date.today().isoformat()
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Optional: Avoid duplicate entries for the same user on the same day
    cursor.execute(
        "SELECT * FROM attendance WHERE user_id = ? AND date(first_seen) = ?",
        (user_id, date_today)
    )
    entry = cursor.fetchone()

    if not entry:
        # If no entry exists, insert the first_seen timestamp
        cursor.execute(
            "INSERT INTO attendance (user_id, first_seen, last_seen) VALUES (?, ?, ?)",
            (user_id, time_now, time_now)
        )
        conn.commit()
    else:
        # If the entry exists, update the last_seen timestamp
        cursor.execute(
            "UPDATE attendance SET last_seen = ? WHERE user_id = ? AND date(first_seen) = ?",
            (time_now, user_id, date_today)
        )
        conn.commit()

    conn.close()


def get_known_faces():
    """
    Get all known faces from the database.
    
    Returns:
        Tuple of (known_names, known_encodings, user_ids)
    """
    # Initialize face recognizer
    recognizer = FaceRecognizer()
    
    # Use the method from the FaceRecognizer class
    return recognizer.get_known_faces()


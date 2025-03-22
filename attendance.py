import sqlite3
from datetime import datetime

def fetch_attendance_records():
    """Fetches attendance records from the database."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT users.name, attendance.first_seen, attendance.last_seen
        FROM attendance
        INNER JOIN users ON attendance.user_id = users.id
    """)
    records = cursor.fetchall()
    conn.close()
    return records

def save_attendance(user_id):
    """Records first seen and last seen time for a user."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("SELECT first_seen FROM attendance WHERE user_id = ?", (user_id,))
    record = cursor.fetchone()

    if record is None:
        cursor.execute("INSERT INTO attendance (user_id, first_seen, last_seen) VALUES (?, ?, ?)",
                       (user_id, current_time, current_time))
    else:
        cursor.execute("UPDATE attendance SET last_seen = ? WHERE user_id = ?", (current_time, user_id))

    conn.commit()
    conn.close()

def initialize_database():
    """Ensures the `attendance` table has the correct structure."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # ✅ Drop old table if it has the wrong structure
    cursor.execute("DROP TABLE IF EXISTS attendance")

    # ✅ Create new table with `first_seen` & `last_seen`
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

# ✅ Run this function once to recreate the table
initialize_database()

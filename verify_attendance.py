import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_todays_attendance():
    """Verify attendance records for today."""
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get all attendance records for today
        cursor.execute("""
            SELECT 
                u.user_id,
                u.name,
                a.first_seen,
                a.last_seen
            FROM users u
            LEFT JOIN attendance a ON u.user_id = a.user_id
            WHERE DATE(a.first_seen) = ?
            ORDER BY a.first_seen DESC
        """, (today,))
        
        records = cursor.fetchall()
        
        print(f"\nAttendance records for {today}:")
        if records:
            for record in records:
                print(f"\nUser ID: {record[0]}")
                print(f"Name: {record[1]}")
                print(f"First seen: {record[2]}")
                print(f"Last seen: {record[3]}")
        else:
            print("No attendance records found for today")
        
        # Show all users for reference
        cursor.execute("SELECT user_id, name FROM users")
        users = cursor.fetchall()
        
        print("\nAll registered users:")
        for user in users:
            print(f"ID: {user[0]}, Name: {user[1]}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error verifying attendance: {e}")

if __name__ == "__main__":
    verify_todays_attendance() 
import sqlite3
import numpy as np
from datetime import datetime  
import requests

# ✅ Define Meta API Credentials
PHONE_NUMBER_ID = "565809386623131"  # Your Meta API Phone Number ID
ACCESS_TOKEN = "EAATjzccDz1oBO36fHbbBzJhNj6YrSbU40tGydnooie6dWc0YdeNfOn8W9BvZC99nTPLQFjP1x9w2t5KUtgO9EmekKoU0JSs4mj80u9uAiwXhhqOZBzRMO0flh5aNMtw7g4t4MtI6LfmCj5xiCYrlGUAAx7nbehqBxuefoh7E1UK8FSkzUDbPg4q47QijItcgzI2TxYY4ZCkKpt0h5sqbivZAFn4G"

WORK_END_TIME = "20:00:00"  # 8 PM

def get_user_phone(user_id):
    """Fetch user's WhatsApp number from the database."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("SELECT phone FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]  # ✅ Return the phone number
    return None  # ❌ No phone number found

def send_whatsapp_template(user_id, name):
    """Sends a pre-approved WhatsApp message template to initiate conversation."""
    user_phone = get_user_phone(user_id)
    if not user_phone:
        print(f"❌ No phone number found for {name}. Cannot send message.")
        return

    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    data = {
        "messaging_product": "whatsapp",
        "to": user_phone,  # ✅ Get the number from DB
        "type": "template",
        "template": {
            "name": "attendance_update",  # ✅ Your approved template name in Meta API
            "language": {"code": "en_US"},
            "components": [
                {
                    "type": "body",
                    "parameters": [{"type": "text", "text": name}]
                }
            ]
        }
    }

    response = requests.post(url, json=data, headers=headers)
    print(f"📨 WhatsApp API Response: {response.json()}")  # ✅ Debugging

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
        return f"✅ Hello {name}, your attendance:\n📍 First Seen: {first_seen}\n📍 Last Seen: {last_seen}"
    else:
        return "❌ No attendance record found for today."

def send_whatsapp_message(user_phone, message):
    """Send a WhatsApp message using Meta Cloud API."""
    url = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    data = {
        "messaging_product": "whatsapp",
        "to": user_phone,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, json=data, headers=headers)
    print(f"📨 WhatsApp API Response: {response.json()}")  # ✅ Debugging

def handle_whatsapp_message(user_phone, user_message):
    """Handle incoming WhatsApp messages and respond."""
    user_message = user_message.lower().strip()
    
    if "check my attendance" in user_message:
        attendance_info = fetch_attendance(user_phone)
        send_whatsapp_message(user_phone, attendance_info)
    else:
        send_whatsapp_message(user_phone, "❌ Invalid command. Send 'Check my attendance' to get your record.")

def save_attendance(user_id):
    """Marks first_seen and updates last_seen, sending only final notification at end of day."""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H:%M:%S")

    # Fetch user details
    cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return

    name = result[0]

    # Check if user already has an entry for today
    cursor.execute("""
        SELECT id, first_seen, last_seen FROM attendance 
        WHERE user_id = ? AND DATE(first_seen) = ?
    """, (user_id, current_date))

    attendance_record = cursor.fetchone()

    if attendance_record:
        attendance_id, first_seen, last_seen = attendance_record

        # Update last_seen but do not send a notification yet
        cursor.execute("""
            UPDATE attendance 
            SET last_seen = ? 
            WHERE id = ?
        """, (current_time, attendance_id))

        # ✅ Only send WhatsApp message at the end of the day (8 PM)
        if current_hour >= WORK_END_TIME:
            send_whatsapp_template(user_id, name)  # ✅ WhatsApp message

    else:
        # Create a new record for today
        cursor.execute("""
            INSERT INTO attendance (user_id, first_seen, last_seen) 
            VALUES (?, ?, ?)
        """, (user_id, current_time, current_time))

        # ✅ Send WhatsApp message when user enters for the first time
        send_whatsapp_template(user_id, name)  # ✅ WhatsApp message

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

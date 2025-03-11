import csv
from datetime import datetime

def mark_attendance(name):
    """Marks attendance in CSV file."""
    with open("attendance.csv", "a", newline="") as file:
        writer = csv.writer(file)

        # Check if attendance is already marked for the day
        with open("attendance.csv", "r") as read_file:
            reader = csv.reader(read_file)
            for row in reader:
                if row and row[0] == name and row[1] == datetime.now().date().strftime("%Y-%m-%d"):
                    print(f"Attendance already marked for {name}")
                    return

        # Write attendance entry
        writer.writerow([name, datetime.now().date().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")])
        print(f"Attendance marked for {name}")
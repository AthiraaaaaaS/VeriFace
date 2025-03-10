import os

def create_csv_file():
    """Creates an attendance.csv file if it does not exist."""
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w", newline="") as file:
            file.write("Name,Date,Time\n")
            print("attendance.csv file created.")

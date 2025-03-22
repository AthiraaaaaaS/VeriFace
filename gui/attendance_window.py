from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from attendance import fetch_attendance_records

class AttendanceWindow(QWidget):
    """Admin Attendance Log Window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance Records")
        self.setGeometry(200, 200, 500, 400)
        self.setup_ui()

    def setup_ui(self):
        """Sets up the attendance log UI."""
        layout = QVBoxLayout()
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(3)
        self.attendance_table.setHorizontalHeaderLabels(["Name", "First Seen", "Last Seen"])
        layout.addWidget(self.attendance_table)

        self.setLayout(layout)
        self.display_attendance_table()

    def display_attendance_table(self):
        """Fetches and displays attendance records in the table."""
        records = fetch_attendance_records()
        self.attendance_table.setRowCount(len(records))

        for row_idx, (name, first_seen, last_seen) in enumerate(records):
            self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(name))
            self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(first_seen))
            self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(last_seen))

        # ✅ Increase column width for full visibility
        self.attendance_table.setColumnWidth(0, 150)
        self.attendance_table.setColumnWidth(1, 200)
        self.attendance_table.setColumnWidth(2, 200)

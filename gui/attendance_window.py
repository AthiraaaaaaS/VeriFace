import sys
import csv
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem

class AttendanceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance Records")
        self.setGeometry(150, 150, 600, 400)
        self.setup_ui()
        self.load_attendance()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.load_attendance)
        layout.addWidget(self.refresh_button)

        self.setLayout(layout)

    def load_attendance(self):
        with open("attendance.csv", "r") as file:
            reader = csv.reader(file)
            data = list(reader)

        self.table.setRowCount(len(data))
        self.table.setColumnCount(len(data[0]))
        self.table.setHorizontalHeaderLabels(["Name", "Date", "Time"])

        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(cell))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceWindow()
    window.show()
    sys.exit(app.exec_())

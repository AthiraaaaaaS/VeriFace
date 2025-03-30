from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QDateEdit, QFileDialog, QMessageBox, QHeaderView
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
import sqlite3
from attendance import fetch_attendance_records
import pandas as pd

class AttendanceWindow(QWidget):
    """Admin Attendance Log Window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance Records")
        self.setGeometry(200, 200, 500, 450)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        # Title Section
        self.title_label = QLabel("Attendance Records")
        self.title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(self.title_label)

        # Date Filter Section
        date_label = QLabel("Select Date:")
        date_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(date_label)

        # Date Picker
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())  # Default to today
        layout.addWidget(self.date_picker)

        self.filter_button = QPushButton("Filter Attendance")
        self.filter_button.setFont(QFont("Segoe UI", 10))
        self.filter_button.clicked.connect(self.display_attendance_table)
        layout.addStretch()
        layout.addWidget(self.filter_button,alignment=Qt.AlignRight)
        layout.addStretch()

        # Attendance Table
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(3)
        self.attendance_table.setHorizontalHeaderLabels(["Name", "First Seen", "Last Seen"])
        
        # Adjust table header
        header = self.attendance_table.horizontalHeader()
        header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.attendance_table)

        # Excel button
        self.export_button = QPushButton("Export to Excel")
        self.export_button.setFont(QFont("Segoe UI", 10))
        self.export_button.clicked.connect(self.export_to_excel)
        layout.addStretch()
        layout.addWidget(self.export_button,alignment=Qt.AlignRight)
        layout.addStretch()

        self.setLayout(layout)

    def apply_styles(self):
        """Apply custom styling to the window and its components."""
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #2c3e50;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
            }
        
            QLabel {
                color: #808080;
                font-size: 10pt;
                font-weight: 500;
            }
        
            QDateEdit {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 5px;
                min-height: 30px;
            }
        
            QPushButton {
                background-color: #009999;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
        
            QPushButton:hover {
                background-color: #00cccc;
            }
        
            QTableWidget {
                background-color: white;
                border: none;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
            }
        
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #667085;
                padding: 10px;
                border: none;
                font-weight: 500;
                text-transform: uppercase;
                font-size: 12px;
                border-bottom: 1px solid #E4E7EC;
            }
        
            QTableWidget::item {
                padding: 12px;
                border-bottom: 1px solid #E4E7EC;
                font-size: 14px;
                color: #101828;
                font-weight: 400;
            }
        
            QTableWidget::item:selected {
                background-color: #e6f2ff;
                color: #2196F3;
                border: none;
            }
    """)

    def display_attendance_table(self):
        """Fetches attendance records for the selected date and updates the table."""
        selected_date = self.date_picker.date().toString("yyyy-MM-dd")

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        query = """
        SELECT users.name, attendance.first_seen, attendance.last_seen 
        FROM attendance
        INNER JOIN users ON attendance.user_id = users.id
        WHERE DATE(attendance.first_seen) = ?
        """
        cursor.execute(query, (selected_date,))
        records = cursor.fetchall()

        conn.close()

        # Update Table
        self.attendance_table.setRowCount(len(records))
        self.attendance_table.setShowGrid(False)  # Remove grid lines
    
        for row_idx, (name, first_seen, last_seen) in enumerate(records):
            name_item = QTableWidgetItem(name)
            first_seen_item = QTableWidgetItem(first_seen)
            last_seen_item = QTableWidgetItem(last_seen)
        
            # Center align text
            name_item.setTextAlignment(Qt.AlignCenter)
            first_seen_item.setTextAlignment(Qt.AlignCenter)
            last_seen_item.setTextAlignment(Qt.AlignCenter)
        
            self.attendance_table.setItem(row_idx, 0, name_item)
            self.attendance_table.setItem(row_idx, 1, first_seen_item)
            self.attendance_table.setItem(row_idx, 2, last_seen_item)
           

    def export_to_excel(self):
        """Exports attendance records to an Excel file."""
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        query = """
        SELECT users.name, attendance.first_seen, attendance.last_seen 
        FROM attendance
        INNER JOIN users ON attendance.user_id = users.id
        """
        cursor.execute(query)
        records = cursor.fetchall()
        conn.close()

        if not records:
            QMessageBox.warning(self, "No Data", "No attendance records available to export.")
            return

        # Convert records to a pandas DataFrame
        df = pd.DataFrame(records, columns=["Name", "First Seen", "Last Seen"])

        # Ask user where to save the file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Attendance", "", "Excel Files (*.xlsx);;All Files (*)", options=options)

        if file_path:
            if not file_path.endswith(".xlsx"):
                file_path += ".xlsx"  # Ensure correct file extension

            df.to_excel(file_path, index=False, engine="openpyxl")
            QMessageBox.information(self, "Export Successful", f"Attendance saved to:\n{file_path}")
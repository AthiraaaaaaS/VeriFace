from PyQt5.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QDateEdit, QFileDialog, QMessageBox, QHeaderView, QHBoxLayout
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont, QIcon, QPixmap
import sqlite3
import pandas as pd
import os

class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Veriface - Attendance System")
        self.setGeometry(100, 100, 800, 600)  # Increased window size

         # Load and display logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons","veriface.png") # Replace with your logo path
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
        else:
            print("Logo file not found.")
        
        self.login_screen = LoginScreen(self)
        self.attendance_screen = AttendanceScreen(self)
        
        self.addWidget(self.login_screen)
        self.addWidget(self.attendance_screen)
        
        self.setCurrentWidget(self.login_screen)
        
class LoginScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Load and display logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons","veriface.png") # Replace with your logo path
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            layout.addWidget(logo_label, alignment=Qt.AlignCenter)
        else:
            print("Logo file not found.")

        self.title_label = QLabel("Welcome Back")
        self.title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(self.title_label)
        
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter Admin Password")
        self.password_field.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_field)
        
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.admin_login)
        self.login_button.setObjectName("signinButton")
        layout.addWidget(self.login_button)

        
        self.setLayout(layout)
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QLineEdit {
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
                color: #2C3E50;
                margin-bottom: 10px;
            }

            QLineEdit::placeholder {
                color: #BDC3C7;
            }

            #signinButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            #signinButton:hover {
                background-color: #2980B9;
            }
        """)
        
    def admin_login(self):
        if self.password_field.text() == "admin123":
            self.main_window.setCurrentWidget(self.main_window.attendance_screen)
        else:
            QMessageBox.warning(self, "Access Denied", "Incorrect Password")

class AttendanceScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.title_label = QLabel("Attendance Records")
        self.title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        top_layout.addWidget(self.title_label)
        
        self.logout_button = QPushButton()
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "switch.png")
        print(f"Icon path: {icon_path}")
        if os.path.exists(icon_path):
            print("Icon file exists.")
            self.logout_button.setIcon(QIcon(icon_path))
        else:
            print("Icon file does not exist.")
            
        self.logout_button.setObjectName("logoutButton")
        self.logout_button.clicked.connect(self.logout)
        self.logout_button.setFixedSize(40, 40)
        top_layout.addStretch()
        top_layout.addWidget(self.logout_button)
        layout.addLayout(top_layout)
        
        date_filter_layout = QHBoxLayout()
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setObjectName("datePicker")
        date_filter_layout.addWidget(self.date_picker)
        
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self.display_attendance_table)
        self.filter_button.setObjectName("filterButton")
        date_filter_layout.addWidget(self.filter_button)
        date_filter_layout.addStretch()

        layout.addLayout(date_filter_layout)
        
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(3)
        self.attendance_table.setHorizontalHeaderLabels(["Name", "First Seen", "Last Seen"])
        self.attendance_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: none;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                margin-bottom:16px;
                border: 1px solid #b6b6b6;
                margin-top: 8px;
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
        header = self.attendance_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.attendance_table)
        
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_button = QPushButton("Export to Excel")
        self.export_button.clicked.connect(self.export_to_excel)
        self.export_button.setObjectName("csvButton")
        export_layout.addWidget(self.export_button)
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        self.setLayout(layout)
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QLabel{
                margin-top: 4px;
                margin-bottom: 24px;
            }
            
            #datePicker{
                padding: 8px;
                border: 1px solid #b6b6b6;
                border-radius: 4px;
                font-size: 14px;
                width: 310px;
            }
            
            #filterButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 100px;
            }
            
            #filterButton:hover {
                background-color: #2980B9;
            }

            #csvButton {
                background-color: #0ea70e;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                width: 150px;
            }
            
            #csvButton:hover {
                background-color: #136113;
            }
        """)
        
    def display_attendance_table(self):
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
        
        self.attendance_table.setRowCount(len(records))
        for row_idx, (name, first_seen, last_seen) in enumerate(records):
            self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(name))
            self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(first_seen))
            self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(last_seen))
        
    def export_to_excel(self):
        row_count = self.attendance_table.rowCount()

        if row_count == 0:
            QMessageBox.warning(self, "No Data", "No attendance records available to export.")
            return

        data = []
        for row in range(row_count):
            row_data = []
            for col in range(self.attendance_table.columnCount()):
                item = self.attendance_table.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append("")  # Handle empty cells
            data.append(row_data)

        df = pd.DataFrame(data, columns=["Name", "First Seen", "Last Seen"])
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Attendance", "", "Excel Files (*.xlsx);;All Files (*)")
        if file_path:
            if not file_path.endswith(".xlsx"):
                file_path += ".xlsx"
            df.to_excel(file_path, index=False, engine="openpyxl")
            QMessageBox.information(self, "Export Successful", f"Attendance saved to:\n{file_path}")
        
    def logout(self):
        self.main_window.setCurrentWidget(self.main_window.login_screen)
        self.main_window.login_screen.password_field.clear()
    
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
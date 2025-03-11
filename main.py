import sys
from PyQt5.QtWidgets import QApplication
from gui.camera_module import AttendanceApp  # Import PyQt UI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec_())
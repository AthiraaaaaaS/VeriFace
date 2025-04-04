import threading
from PyQt5.QtWidgets import QApplication
import sys
from live_camera import run_live_camera
from gui.attendance_ui import MainWindow

def start_cctv():
    """Runs CCTV face detection in the background."""
    run_live_camera()

if __name__ == "__main__":
    # Start CCTV in a separate thread
    threading.Thread(target=start_cctv, daemon=True).start()

    # Start the GUI application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

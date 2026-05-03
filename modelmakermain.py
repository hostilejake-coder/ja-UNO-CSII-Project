"""
CSCI-1620 — Startup Module
"""
try:
    from PyQt6 import QtWidgets
    from PyQt6.QtWidgets import QApplication
    import sys
    from gui_logic import MainWindow
except Exception as e:
    print(f"Error loading imports: {e}")

if __name__ == "__main__":
    print("Creating window... Please wait.")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
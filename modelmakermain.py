from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication
import sys
from gui_logic import MainWindow

if __name__ == "__main__":
    print("Creating window... Please wait.")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
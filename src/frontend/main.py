import sys
from PyQt6.QtWidgets import QApplication
from frontend.main_window import MainWindow
from frontend.screens.phase1_screen import Phase1Screen
from frontend.styles.theme import GLOBAL_QSS


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)

    window = MainWindow()
    screen = Phase1Screen()
    window.add_screen(screen)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

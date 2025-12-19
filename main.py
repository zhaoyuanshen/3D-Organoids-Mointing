import sys
import os
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QMovie
from PySide6.QtCore import Qt, QThread, Signal


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(600, 300, 400, 300)

        layout = QVBoxLayout()

        # --- gif 动画 ---
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        gif_path = resource_path("Material wave loading.gif")
        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.label.setMovie(self.movie)
            self.movie.start()
        else:
            self.label.setText("Loading...")

        layout.addWidget(self.label)

        # --- 提示文字 ---
        self.tip = QLabel("正在加载主程序，请稍候…")
        self.tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.tip)

        self.setLayout(layout)


# --- 后台线程：加载主 GUI ---
class LoaderThread(QThread):
    finished = Signal(object)

    def run(self):
        from GUI import CellAnalysisApp   # ✅ 在后台导入 GUI
        self.finished.emit(CellAnalysisApp)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()

    def start_main(CellAnalysisApp):
        global window
        window = CellAnalysisApp()
        window.show()
        splash.close()

    # 用后台线程加载 GUI
    loader = LoaderThread()
    loader.finished.connect(start_main)
    loader.start()

    sys.exit(app.exec())

import sys
import os
import torch
import re
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QCheckBox, QFileDialog, QRadioButton, QButtonGroup, QFrame, QGroupBox,
    QMessageBox, QInputDialog
)
from PySide6.QtGui import QPixmap, QMovie
from PySide6.QtCore import Qt, QThread, Signal

from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from cellpose import models as cp_models


# --- 工具函数：处理 PyInstaller 打包后的资源路径 ---
def resource_path(relative_path):
    """获取资源文件的正确路径，支持 PyInstaller 打包后的环境"""
    if getattr(sys, 'frozen', False):  # 打包后的 exe
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- 后台分割线程 ---
class SegWorker(QThread):
    finished = Signal(dict, str)  # 结果字典, 输出目录

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self, skimage=None):
        # 重载耗时操作到子线程，避免卡 UI
        from cellpose import io, plot
        from skimage import measure
        import matplotlib.pyplot as plt
        import pandas as pd
        import os

        # 读取图像并执行分割
        img = io.imread(self.image_path)
        masks, flows, styles = self.model.eval(
            img, batch_size=8, flow_threshold=0.4, cellprob_threshold=0.0
        )

        # 输出目录：原始图像所在目录下的 "原始图像名_outpost"
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        out_dir = os.path.join(os.path.dirname(self.image_path), f"{base}_outpost")
        os.makedirs(out_dir, exist_ok=True)

        mask_path = os.path.join(out_dir, f"{base}_masks.tif")
        io.imsave(mask_path, masks.astype("uint16"))

        # 保存 outlines / overlay / cellpose 三张图
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks, flows[0])
        plt.tight_layout()

        axes = fig.axes
        if len(axes) >= 4:
            outlines_path = os.path.join(out_dir, f"{base}_outlines.png")
            overlay_path = os.path.join(out_dir, f"{base}_overlay.png")
            cellpose_path = os.path.join(out_dir, f"{base}_cellpose.png")

            plt.imsave(outlines_path, axes[1].images[0].get_array())
            plt.imsave(overlay_path, axes[2].images[0].get_array())
            plt.imsave(cellpose_path, axes[3].images[0].get_array())

        plt.close(fig)

        # --- 统计与指标计算 ---
        if img.ndim == 2:
            H, W = img.shape
        else:
            H, W = img.shape[:2]
        total_pixels = H * W

        props = measure.regionprops_table(masks, properties=['area'])
        df_props = pd.DataFrame(props)
        cell_count = len(df_props)
        mean_area = float(df_props['area'].mean()) if cell_count > 0 else 0.0
        total_cell_area = int(df_props['area'].sum()) if cell_count > 0 else 0

        # 细胞所占面积（百分比）
        area_percentage = (total_cell_area / total_pixels) if total_pixels > 0 else 0.0
        # 平均细胞面积（像素）
        avg_cell_area = mean_area

        result = {
            "cell_count": cell_count,
            "mean_area": mean_area,
            "total_cell_area": total_cell_area,
            "image_pixels": total_pixels,
            "area_percentage": area_percentage,
            "avg_cell_area": avg_cell_area
        }

        # 发回主线程处理
        self.finished.emit(result, out_dir)

class CellAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("类器官3D打印细胞在线分析系统")
        self.setGeometry(200, 100, 1200, 700)

        # --- 状态变量 ---
        self.use_gpu = torch.cuda.is_available()
        self.force_use_gpu = False
        self.image_path = None
        self.week_model = None
        self.cellpose_model = None
        self.last_segmentation_dir = None
        self.segmentation_summary = {}

        # --- 初始化界面 ---
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        # 左侧功能栏
        left_panel = QVBoxLayout()

        # --- Logo (大) ---
        logo_label = QLabel()
        logo_path = resource_path("CELLlogo.png")  # ✅ 改为相对路径
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(480, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("LOGO 占位")
            logo_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(logo_label)

        # ================= 分区 1：系统信息与图像加载 =================
        group_system = QGroupBox("系统信息 & 图像加载")
        system_layout = QVBoxLayout()

        # ---- GPU 检测 ----
        if self.use_gpu:
            gpu_label_text = f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}"
        else:
            gpu_label_text = "⚠️ 未检测到GPU，仅使用CPU"
        self.gpu_label = QLabel(gpu_label_text)
        system_layout.addWidget(self.gpu_label)

        self.gpu_checkbox = QCheckBox("使用GPU")
        self.gpu_checkbox.setEnabled(self.use_gpu)
        self.gpu_checkbox.setChecked(self.use_gpu)
        self.gpu_checkbox.stateChanged.connect(self.toggle_gpu)
        system_layout.addWidget(self.gpu_checkbox)

        # ---- 图片加载 ----
        self.load_img_btn = QPushButton("加载图像")
        self.load_img_btn.clicked.connect(self.load_image)
        system_layout.addWidget(self.load_img_btn)

        self.img_name_label = QLabel("未选择图片 请确保图像路径全英文")
        system_layout.addWidget(self.img_name_label)

        # 分割栏
        group_system.setLayout(system_layout)
        left_panel.addWidget(group_system)

        # ================= 分区 2：分化预测功能 =================
        group_week = QGroupBox("细胞分化预测")
        week_layout = QVBoxLayout()

        # ---- 周数预测 ----
        self.load_week_model_btn = QPushButton("加载分化预测模型")
        self.load_week_model_btn.clicked.connect(self.load_week_model)
        week_layout.addWidget(self.load_week_model_btn)

        self.run_week_btn = QPushButton("运行预测")
        self.run_week_btn.clicked.connect(self.run_week_prediction)
        week_layout.addWidget(self.run_week_btn)

        self.week_result_label = QLabel("预测结果: -")
        week_layout.addWidget(self.week_result_label)

        group_week.setLayout(week_layout)
        # 分隔栏
        left_panel.addWidget(group_week)

        # ================= 分区 3：细胞分割功能 =================
        group_cp = QGroupBox("细胞分割")
        cp_layout = QVBoxLayout()

        # ---- Cellpose 分割 ----
        self.load_cp_model_btn = QPushButton("加载细胞分割模型")
        self.load_cp_model_btn.clicked.connect(self.load_cellpose_model)
        cp_layout.addWidget(self.load_cp_model_btn)

        self.run_cp_btn = QPushButton("运行分割")
        self.run_cp_btn.clicked.connect(self.run_cellpose)
        cp_layout.addWidget(self.run_cp_btn)

        # 输出结果选择（单选）
        cp_layout.addWidget(QLabel("选择显示结果:"))
        self.radio_group = QButtonGroup(self)
        self.radio_outlines = QRadioButton("Predicted Outlines")
        self.radio_masks = QRadioButton("Predicted Masks (Overlay)")
        self.radio_cellpose = QRadioButton("Predicted Cell Pose")
        self.radio_group.addButton(self.radio_outlines)
        self.radio_group.addButton(self.radio_masks)
        self.radio_group.addButton(self.radio_cellpose)
        self.radio_masks.setChecked(True)   # 默认显示overlay

        self.radio_outlines.toggled.connect(self.update_result_image)
        self.radio_masks.toggled.connect(self.update_result_image)
        self.radio_cellpose.toggled.connect(self.update_result_image)

        cp_layout.addWidget(self.radio_outlines)
        cp_layout.addWidget(self.radio_masks)
        cp_layout.addWidget(self.radio_cellpose)

        group_cp.setLayout(cp_layout)
        left_panel.addWidget(group_cp)

        # 左栏放到主布局
        main_layout.addLayout(left_panel, 2)

        # ================= 右侧显示栏 =================
        right_panel = QVBoxLayout()

        self.orig_img_label = QLabel("原始图像")
        self.orig_img_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.orig_img_label)

        self.result_img_label = QLabel("计算结果图像")
        self.result_img_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.result_img_label)

        self.summary_label = QLabel("细胞数量: - | 细胞所占比: - | 平均细胞面积：-")
        self.summary_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.summary_label)

        # --- 右下角 mini logo ---
        self.mini_logo = QLabel()
        mini_logo_path = resource_path("logo.png")
        if os.path.exists(mini_logo_path):
            mini_pixmap = QPixmap(mini_logo_path).scaled(200, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.mini_logo.setPixmap(mini_pixmap)
        self.mini_logo.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        right_panel.addWidget(self.mini_logo)

        main_layout.addLayout(right_panel, 5)


    # --- GPU 选择 ---
    def toggle_gpu(self, state):
        self.force_use_gpu = (state == Qt.Checked)

    # 检查英文路径
    def check_english_path(self, path: str) -> bool:
        """检查路径是否全英文"""
        if re.search(r'[\u4e00-\u9fff]', path):
            QMessageBox.warning(
                self,
                "路径错误",
                "❌ 检测到中文路径，请使用全英文路径！"
            )
            return False
        return True

    # 裁剪逻辑代码
    def crop_image(self, img, region="left_top"):
        w, h = img.size
        crop_size = min(224, w, h)

        if region == "left_top":
            return img.crop((0, 0, crop_size, crop_size))
        elif region == "right_top":
            return img.crop((w - crop_size, 0, w, crop_size))
        elif region == "left_bottom":
            return img.crop((0, h - crop_size, crop_size, h))
        elif region == "right_bottom":
            return img.crop((w - crop_size, h - crop_size, w, h))
        elif region == "center":
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            return img.crop((cx - half, cy - half, cx + half, cy + half))
        return img

    # --- 加载图像 ---
    def load_image(self):
        while True:
            fname, _ = QFileDialog.getOpenFileName(
                self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.tif)"
            )
            if not fname:
                return

            # 路径检查
            if not self.check_english_path(fname):
                continue

            img = Image.open(fname)
            w, h = img.size

            # 分辨率检查
            if w > 224 or h > 224:
                regions = ["左上角", "右上角", "左下角", "右下角", "中心"]
                region_keys = ["left_top", "right_top", "left_bottom", "right_bottom", "center"]

                region, ok = QInputDialog.getItem(
                    self, "选择裁剪区域",
                    f"⚠️ 当前图片分辨率为 {w}x{h}\n请选择一个裁剪区域生成 224×224 小图：",
                    regions, 0, False
                )
                if not ok:  # 用户取消
                    return

                # 裁剪
                idx = regions.index(region)
                cropped_img = self.crop_image(img, region_keys[idx])
                temp_path = os.path.join(os.path.dirname(fname), "temp_cropped.png")
                cropped_img.save(temp_path)
                fname = temp_path

                QMessageBox.information(
                    self, "裁剪完成",
                    f"已裁剪 {region} 区域为 224×224，用于后续分析。"
                )

            # 保存路径 & 显示
            self.image_path = fname
            self.img_name_label.setText(os.path.basename(fname))
            pixmap = QPixmap(fname).scaled(300, 300, Qt.KeepAspectRatio)
            self.orig_img_label.setPixmap(pixmap)
            break

    # --- 加载分化模型 ---
    def load_week_model(self):
        while True:
            model_path, _ = QFileDialog.getOpenFileName(
                self, "选择分化预测模型", "", "Model (*.pth)"
            )
            if not model_path:
                return
            if not self.check_english_path(model_path):
                continue

            # 正常加载
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 4)
            device = torch.device("cuda:0" if (self.force_use_gpu and torch.cuda.is_available()) else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            self.week_model = model
            self.week_device = device
            self.week_result_label.setText("✅ 分化预测模型已加载")
            break

    # --- 周数预测 ---
    def run_week_prediction(self):
        if not self.image_path or not self.week_model:
            self.week_result_label.setText("❌ 请先加载图片和模型")
            return
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(self.image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(self.week_device)

        with torch.no_grad():
            outputs = self.week_model(img_tensor)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probs[0, preds.item()].item()

        idx_to_class = {0: "week1", 1: "week2", 2: "week3", 3: "week4"}
        self.week_result_label.setText(
            f"预测结果: {idx_to_class[preds.item()]} (置信度 {confidence:.2%})"
        )

    # --- 加载Cellpose模型 ---
    def load_cellpose_model(self):
        while True:
            model_path, _ = QFileDialog.getOpenFileName(
                self, "选择细胞模型文件", "", "All Files (*)"
            )
            if not model_path:
                return
            if not self.check_english_path(model_path):
                continue

            # 正常加载
            self.cellpose_model = cp_models.CellposeModel(
                gpu=(self.force_use_gpu and torch.cuda.is_available()),
                pretrained_model=model_path
            )
            self.summary_label.setText(f"✅ 细胞模型已加载: {os.path.basename(model_path)}")
            break

    # --- 显示加载动画 ---
    def show_loading(self):
        loading_gif = resource_path("Material wave loading.gif")
        if os.path.exists(loading_gif):
            self.loading_movie = QMovie(loading_gif)
            self.result_img_label.setMovie(self.loading_movie)
            self.loading_movie.start()
        else:
            self.result_img_label.setText("⏳ 正在分割，请稍候...")

    # --- 运行Cellpose分割（改为后台线程） ---
    def run_cellpose(self):
        if not self.image_path or not self.cellpose_model:
            self.summary_label.setText("❌ 请先加载图片和细胞分割模型")
            return

        # 显示加载中动画
        self.show_loading()
        QApplication.processEvents()  # 确保动画立即刷新

        # 启动后台线程
        self.seg_worker = SegWorker(self.cellpose_model, self.image_path)
        self.seg_worker.finished.connect(self.on_segmentation_done)
        self.seg_worker.start()

    # --- 分割完成回调（运行在主线程） ---
    def on_segmentation_done(self, result: dict, out_dir: str):
        # 停止动画
        if self.loading_movie:
            self.loading_movie.stop()
            self.loading_movie = None

        # 更新统计
        self.segmentation_summary = result
        self.last_segmentation_dir = out_dir

        self.summary_label.setText(
            f"细胞数量: {result['cell_count']} | "
            f"细胞所占比: {result['area_percentage']:.2%} | "
            f"平均细胞面积: {result['avg_cell_area']:.1f}"
        )

        # 默认展示 overlay
        self.radio_masks.setChecked(True)
        self.update_result_image()

        # 释放线程对象引用
        self.seg_worker = None

    # --- 显示计算结果图 ---
    def update_result_image(self):
        if not self.last_segmentation_dir or not self.image_path:
            return

        base = os.path.splitext(os.path.basename(self.image_path))[0]
        out_dir = self.last_segmentation_dir
        img_to_show = None

        if self.radio_outlines.isChecked():
            img_to_show = os.path.join(out_dir, f"{base}_outlines.png")
        elif self.radio_masks.isChecked():
            img_to_show = os.path.join(out_dir, f"{base}_overlay.png")
        elif self.radio_cellpose.isChecked():
            img_to_show = os.path.join(out_dir, f"{base}_cellpose.png")

        if img_to_show and os.path.exists(img_to_show):
            pixmap = QPixmap(img_to_show).scaled(300, 300, Qt.KeepAspectRatio)
            self.result_img_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CellAnalysisApp()
    window.show()
    sys.exit(app.exec())

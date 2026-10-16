from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from adapters import ADAPTER_NAMES
from camera.camera_utils import directshow_camera_names, discover_camera_indices
from camera.video_worker import VideoFileThread
from camera.worker import CameraThread
from core.device import detect_devices
from core.inference_worker import InferenceThread
from core.profile import load_profile, save_profile
from core.results import BenchmarkResult, InferenceResult, ModelInfo
from processing.renderer import render_result
from ui.preview_widget import RoiPreviewWidget


APP_TITLE = "Universal Vision Model Tester"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1640, 960)
        self.setMinimumSize(1180, 760)

        self.device_status = detect_devices()
        self.current_frame: Optional[np.ndarray] = None
        self.last_result: Optional[InferenceResult] = None
        self.current_source_kind = "none"
        self.current_source_name = ""
        self.current_camera_index: Optional[int] = None
        self.active_width = 0
        self.active_height = 0
        self.camera_thread: Optional[CameraThread] = None
        self.video_thread: Optional[VideoFileThread] = None
        self.folder_images: list[Path] = []
        self.folder_index = -1
        self.loaded_labels: list[str] = []
        self.model_is_loaded = False
        self.inference_busy = False
        self.anti_flicker_ready = True
        self._closing = False

        self.inference_thread = InferenceThread(self.device_status, self)
        self.inference_thread.model_loaded.connect(self.on_model_loaded)
        self.inference_thread.model_load_failed.connect(self.on_model_load_failed)
        self.inference_thread.model_unloaded.connect(self.on_model_unloaded)
        self.inference_thread.result_ready.connect(self.on_inference_result)
        self.inference_thread.benchmark_ready.connect(self.on_benchmark_result)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.status.connect(self.on_inference_status)
        self.inference_thread.busy_changed.connect(self.on_inference_busy)
        self.inference_thread.start()

        self._build_ui()
        self._apply_style()
        self._connect_shortcuts()
        self._set_adapter_specific_visibility("Auto")
        self._update_action_states()
        QTimer.singleShot(150, self.refresh_cameras)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("Title")
        subtitle = QLabel("Camera • image • video • ROI • CUDA/CPU • detection • segmentation • classification • embeddings")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.device_badge = QLabel(self.device_status.summary)
        self.device_badge.setObjectName("Badge")
        header.addWidget(self.device_badge)
        self.status_badge = QLabel("Starting")
        self.status_badge.setObjectName("Badge")
        header.addWidget(self.status_badge)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)

        # Preview / results side.
        preview_card = QFrame()
        preview_card.setObjectName("Card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(8)

        self.preview = RoiPreviewWidget()
        self.preview.roi_changed.connect(self.on_roi_changed)
        preview_layout.addWidget(self.preview, 1)

        info_row = QHBoxLayout()
        self.source_label = QLabel("Source: —")
        self.resolution_label = QLabel("Resolution: —")
        self.camera_fps_label = QLabel("Source FPS: —")
        self.roi_label = QLabel("ROI: none")
        info_row.addWidget(self.source_label)
        info_row.addSpacing(12)
        info_row.addWidget(self.resolution_label)
        info_row.addSpacing(12)
        info_row.addWidget(self.camera_fps_label)
        info_row.addSpacing(12)
        info_row.addWidget(self.roi_label)
        info_row.addStretch(1)
        preview_layout.addLayout(info_row)

        perf = QFrame()
        perf.setObjectName("InnerCard")
        perf_layout = QHBoxLayout(perf)
        perf_layout.setContentsMargins(10, 7, 10, 7)
        self.preprocess_label = QLabel("Pre: —")
        self.infer_label = QLabel("Infer: —")
        self.post_label = QLabel("Post: —")
        self.total_label = QLabel("Total: —")
        self.model_fps_label = QLabel("Model FPS: —")
        self.result_count_label = QLabel("Results: —")
        for w in (self.preprocess_label, self.infer_label, self.post_label, self.total_label,
                  self.model_fps_label, self.result_count_label):
            perf_layout.addWidget(w)
        perf_layout.addStretch(1)
        preview_layout.addWidget(perf)

        result_row = QHBoxLayout()
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(135)
        self.result_text.setPlaceholderText("Inference output will appear here.")
        result_row.addWidget(self.result_text, 1)
        preview_layout.addLayout(result_row)

        body.addWidget(preview_card, 1)

        # Scrollable control side.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(445)
        scroll.setObjectName("ControlScroll")
        controls_host = QWidget()
        controls = QVBoxLayout(controls_host)
        controls.setContentsMargins(8, 8, 8, 8)
        controls.setSpacing(9)
        scroll.setWidget(controls_host)

        controls.addWidget(self._build_model_group())
        controls.addWidget(self._build_input_group())
        controls.addWidget(self._build_roi_group())
        controls.addWidget(self._build_inference_group())
        controls.addWidget(self._build_specific_group())
        controls.addStretch(1)
        body.addWidget(scroll)
        root.addLayout(body, 1)

    def _build_model_group(self):
        group = QGroupBox("Model")
        layout = QVBoxLayout(group)

        source_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Model file, local model directory, or Hugging Face ID")
        self.model_path_edit.editingFinished.connect(self.on_model_source_changed)
        browse = QPushButton("File")
        browse.clicked.connect(self.choose_model_file)
        browse_dir = QPushButton("Dir")
        browse_dir.clicked.connect(self.choose_model_directory)
        source_row.addWidget(self.model_path_edit, 1)
        source_row.addWidget(browse)
        source_row.addWidget(browse_dir)
        layout.addLayout(source_row)

        form = QFormLayout()
        self.adapter_combo = QComboBox()
        self.adapter_combo.addItems(ADAPTER_NAMES)
        self.adapter_combo.currentTextChanged.connect(self._set_adapter_specific_visibility)
        form.addRow("Adapter", self.adapter_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (best available)", "auto")
        if self.device_status.cuda_available:
            for i, name in enumerate(self.device_status.cuda_devices):
                self.device_combo.addItem(f"CUDA:{i} • {name}", f"cuda:{i}")
        self.device_combo.addItem("CPU", "cpu")
        form.addRow("Device", self.device_combo)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.setObjectName("PrimaryButton")
        self.load_model_button.clicked.connect(self.load_model)
        self.unload_model_button = QPushButton("Unload")
        self.unload_model_button.clicked.connect(self.unload_model)
        self.save_profile_button = QPushButton("Save Profile")
        self.save_profile_button.clicked.connect(self.save_current_profile)
        button_row.addWidget(self.load_model_button, 1)
        button_row.addWidget(self.unload_model_button)
        button_row.addWidget(self.save_profile_button)
        layout.addLayout(button_row)

        self.model_info_label = QLabel("No model loaded")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setObjectName("InfoBox")
        layout.addWidget(self.model_info_label)
        return group

    def _build_input_group(self):
        group = QGroupBox("Input Source")
        layout = QVBoxLayout(group)
        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_cameras)
        self.open_camera_button = QPushButton("Open Camera")
        self.open_camera_button.clicked.connect(self.open_selected_camera)
        camera_row.addWidget(self.camera_combo, 1)
        camera_row.addWidget(refresh)
        camera_row.addWidget(self.open_camera_button)
        layout.addLayout(camera_row)

        file_row = QHBoxLayout()
        image_btn = QPushButton("Image")
        image_btn.clicked.connect(self.open_image)
        video_btn = QPushButton("Video")
        video_btn.clicked.connect(self.open_video)
        folder_btn = QPushButton("Folder")
        folder_btn.clicked.connect(self.open_image_folder)
        file_row.addWidget(image_btn)
        file_row.addWidget(video_btn)
        file_row.addWidget(folder_btn)
        layout.addLayout(file_row)

        folder_nav = QHBoxLayout()
        self.prev_image_button = QPushButton("◀ Previous")
        self.prev_image_button.clicked.connect(lambda: self.navigate_folder(-1))
        self.folder_position_label = QLabel("—")
        self.folder_position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_image_button = QPushButton("Next ▶")
        self.next_image_button.clicked.connect(lambda: self.navigate_folder(1))
        folder_nav.addWidget(self.prev_image_button)
        folder_nav.addWidget(self.folder_position_label, 1)
        folder_nav.addWidget(self.next_image_button)
        layout.addLayout(folder_nav)

        anti_form = QFormLayout()
        self.anti_flicker_combo = QComboBox()
        for label, key in [
            ("Adaptive", "adaptive"), ("Normal Raw", "normal"),
            ("Driver 50 Hz", "driver_50"), ("Driver 60 Hz", "driver_60"),
            ("Average 30", "average_30"), ("Average 90", "average_90"),
            ("Row Correction", "row"), ("Average 90 + Row", "hybrid"),
        ]:
            self.anti_flicker_combo.addItem(label, key)
        self.anti_flicker_combo.currentIndexChanged.connect(self.on_anti_flicker_changed)
        anti_form.addRow("Anti-flicker", self.anti_flicker_combo)
        layout.addLayout(anti_form)

        af_row = QHBoxLayout()
        recalibrate = QPushButton("Recalibrate")
        recalibrate.clicked.connect(self.recalibrate_anti_flicker)
        driver = QPushButton("Driver Settings")
        driver.setEnabled(sys.platform == "win32")
        driver.setToolTip("Camera driver settings dialog is available on Windows only.")
        driver.clicked.connect(self.open_driver_settings)
        self.flicker_status_label = QLabel("Anti-flicker: —")
        self.flicker_status_label.setWordWrap(True)
        af_row.addWidget(recalibrate)
        af_row.addWidget(driver)
        layout.addLayout(af_row)
        layout.addWidget(self.flicker_status_label)
        return group

    def _build_roi_group(self):
        group = QGroupBox("ROI")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        full = QPushButton("Full Frame")
        full.clicked.connect(self.preview.set_full_frame_roi)
        clear = QPushButton("Clear ROI")
        clear.clicked.connect(self.preview.clear_roi)
        row.addWidget(full)
        row.addWidget(clear)
        layout.addLayout(row)
        self.use_roi_checkbox = QCheckBox("Infer only inside drawn ROI")
        self.use_roi_checkbox.setChecked(True)
        layout.addWidget(self.use_roi_checkbox)
        hint = QLabel("Drag on the preview to draw a persistent ROI. Detections/masks are remapped back to full-frame coordinates.")
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        layout.addWidget(hint)
        return group

    def _build_inference_group(self):
        group = QGroupBox("Inference")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        form.addRow("Confidence", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        form.addRow("IoU", self.iou_spin)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        form.addRow("Image size", self.imgsz_spin)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 100)
        self.topk_spin.setValue(5)
        form.addRow("Top-K", self.topk_spin)
        layout.addLayout(form)

        labels_row = QHBoxLayout()
        self.labels_path_edit = QLineEdit()
        self.labels_path_edit.setPlaceholderText("Optional labels .txt / .json")
        labels_btn = QPushButton("Labels")
        labels_btn.clicked.connect(self.choose_labels_file)
        labels_row.addWidget(self.labels_path_edit, 1)
        labels_row.addWidget(labels_btn)
        layout.addLayout(labels_row)

        action_row = QHBoxLayout()
        self.capture_infer_button = QPushButton("Capture + Infer")
        self.capture_infer_button.setObjectName("PrimaryButton")
        self.capture_infer_button.clicked.connect(self.capture_and_infer)
        self.live_checkbox = QCheckBox("Live Inference")
        self.live_checkbox.toggled.connect(self.on_live_toggled)
        action_row.addWidget(self.capture_infer_button, 1)
        action_row.addWidget(self.live_checkbox)
        layout.addLayout(action_row)

        second_row = QHBoxLayout()
        self.benchmark_button = QPushButton("Benchmark ×20")
        self.benchmark_button.clicked.connect(self.run_benchmark)
        save_result = QPushButton("Save Annotated")
        save_result.clicked.connect(self.save_annotated_result)
        clear_result = QPushButton("Clear Results")
        clear_result.clicked.connect(self.clear_results)
        second_row.addWidget(self.benchmark_button)
        second_row.addWidget(save_result)
        second_row.addWidget(clear_result)
        layout.addLayout(second_row)
        return group

    def _build_specific_group(self):
        holder = QGroupBox("Model-specific Settings")
        holder_layout = QVBoxLayout(holder)

        self.classifier_group = QGroupBox("Classifier / Generic Tensor")
        form = QFormLayout(self.classifier_group)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["resnet50", "resnet18", "mobilenet_v3_large", "efficientnet_b0"])
        form.addRow("Architecture", self.arch_combo)
        self.classifier_size_spin = QSpinBox()
        self.classifier_size_spin.setRange(32, 2048)
        self.classifier_size_spin.setValue(224)
        form.addRow("Input size", self.classifier_size_spin)
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(0, 100000)
        self.num_classes_spin.setSpecialValueText("Auto / labels")
        form.addRow("Classes", self.num_classes_spin)
        self.normalize_checkbox = QCheckBox("Scale pixels to 0–1")
        self.normalize_checkbox.setChecked(True)
        form.addRow(self.normalize_checkbox)
        self.imagenet_checkbox = QCheckBox("ImageNet mean/std (PyTorch classifier)")
        self.imagenet_checkbox.setChecked(True)
        form.addRow(self.imagenet_checkbox)
        holder_layout.addWidget(self.classifier_group)

        self.rfdetr_group = QGroupBox("RF-DETR")
        rfd_form = QFormLayout(self.rfdetr_group)
        self.rfdetr_variant_combo = QComboBox()
        self.rfdetr_variant_combo.addItems(["auto", "base", "nano", "small", "medium", "large"])
        self.rfdetr_variant_combo.setCurrentText("auto")
        rfd_form.addRow("Variant", self.rfdetr_variant_combo)
        holder_layout.addWidget(self.rfdetr_group)

        self.fastsam_group = QGroupBox("FastSAM")
        fs_form = QFormLayout(self.fastsam_group)
        self.fastsam_prompt_combo = QComboBox()
        self.fastsam_prompt_combo.addItem("Everything", "everything")
        self.fastsam_prompt_combo.addItem("ROI as box prompt", "roi_box")
        self.fastsam_prompt_combo.addItem("ROI center point", "center_point")
        self.fastsam_prompt_combo.addItem("Text prompt", "text")
        fs_form.addRow("Prompt", self.fastsam_prompt_combo)
        self.fastsam_text_edit = QLineEdit()
        self.fastsam_text_edit.setPlaceholderText("e.g. gold chain")
        fs_form.addRow("Text", self.fastsam_text_edit)
        self.retina_masks_checkbox = QCheckBox("Retina masks")
        self.retina_masks_checkbox.setChecked(True)
        fs_form.addRow(self.retina_masks_checkbox)
        holder_layout.addWidget(self.fastsam_group)

        self.siglip_group = QGroupBox("SigLIP / Text Similarity")
        sig_layout = QVBoxLayout(self.siglip_group)
        sig_layout.addWidget(QLabel("One text prompt per line:"))
        self.text_prompts_edit = QPlainTextEdit()
        self.text_prompts_edit.setPlaceholderText("gold necklace\ngold ring\nsilver jewellery\nfake jewellery")
        self.text_prompts_edit.setMaximumHeight(110)
        sig_layout.addWidget(self.text_prompts_edit)
        holder_layout.addWidget(self.siglip_group)
        return holder

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#101318; color:#E8EAED; font-family:Segoe UI, Arial; font-size:13px; }
            QLabel#Title { font-size:25px; font-weight:700; color:#FFFFFF; }
            QLabel#Subtitle, QLabel#Hint { color:#9EA4AE; }
            QLabel#Badge { background:#202630; border:1px solid #343C49; border-radius:11px; padding:7px 12px; }
            QFrame#Card, QGroupBox { background:#171B22; border:1px solid #2A303A; border-radius:10px; }
            QFrame#InnerCard { background:#11151B; border:1px solid #2A303A; border-radius:7px; }
            QGroupBox { margin-top:12px; padding-top:8px; font-weight:600; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; color:#FFFFFF; }
            QLabel#InfoBox { background:#11151B; border:1px solid #2A303A; border-radius:7px; padding:8px; color:#B9C0CA; }
            QPushButton { background:#262D37; border:1px solid #39424F; border-radius:7px; padding:8px 10px; color:#F1F3F4; }
            QPushButton:hover { background:#303846; }
            QPushButton:disabled { color:#707781; background:#1A1E24; border-color:#2A303A; }
            QPushButton#PrimaryButton { background:#00A67E; border:1px solid #17C39A; color:#FFFFFF; font-weight:700; }
            QPushButton#PrimaryButton:hover { background:#0BB58B; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background:#11151B; border:1px solid #353D49; border-radius:6px; padding:6px; selection-background-color:#00A67E;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus { border:1px solid #00A67E; }
            QScrollArea#ControlScroll { border:none; background:#101318; }
            QScrollBar:vertical { background:#11151B; width:10px; margin:0; }
            QScrollBar::handle:vertical { background:#3B4451; min-height:30px; border-radius:5px; }
        """)

    def _connect_shortcuts(self):
        self.capture_shortcut = QShortcut(QKeySequence("F8"), self)
        self.capture_shortcut.activated.connect(self.capture_and_infer)
        self.clear_roi_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.clear_roi_shortcut.activated.connect(self.preview.clear_roi)

    # ------------------------------------------------------------ model logic

    def choose_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Model", "",
            "Model files (*.pt *.pth *.onnx *.keras *.h5 *.tflite *.torchscript *.jit);;All files (*.*)")
        if path:
            self.model_path_edit.setText(path)
            self.on_model_source_changed()

    def choose_model_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Local Model Directory")
        if path:
            self.model_path_edit.setText(path)
            self.on_model_source_changed()

    def on_model_source_changed(self):
        source = self.model_path_edit.text().strip()
        profile = load_profile(source) if source else {}
        if profile:
            self._apply_profile(profile)
            self.set_status("Profile loaded")

    def load_model(self):
        source = self.model_path_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "Model", "Choose a model file/directory or enter a Hugging Face model ID.")
            return
        self.live_checkbox.setChecked(False)
        self.model_is_loaded = False
        self._update_action_states()
        self.set_status("Loading model...")
        self.model_info_label.setText("Loading...")
        self.load_model_button.setEnabled(False)
        self.last_result = None
        self.inference_thread.load_model(source, self.adapter_combo.currentText(), self.current_settings())

    def on_model_load_failed(self, message: str):
        self.model_is_loaded = False
        self.load_model_button.setEnabled(True)
        self.model_info_label.setText("Model failed to load")
        self.last_result = None
        self._update_action_states()
        self.set_status("Model load failed", bad=True)
        self.result_text.setPlainText(message)
        QMessageBox.critical(self, "Model Load Error", message)

    def unload_model(self):
        self.live_checkbox.setChecked(False)
        self.inference_thread.unload_model()

    def on_model_loaded(self, info: ModelInfo):
        self.model_is_loaded = True
        self.load_model_button.setEnabled(True)
        effective_imgsz = info.details.get("effective_imgsz")
        if isinstance(effective_imgsz, (list, tuple)):
            if len(effective_imgsz) == 2 and int(effective_imgsz[0]) == int(effective_imgsz[1]):
                effective_imgsz = int(effective_imgsz[0])
            else:
                effective_imgsz = None
        if effective_imgsz is not None:
            effective_imgsz = int(effective_imgsz)
            if self.imgsz_spin.value() != effective_imgsz:
                self.imgsz_spin.setValue(effective_imgsz)
        self.model_info_label.setText(
            f"Adapter: {info.adapter}\nTask: {info.task}\nDevice: {info.device}\n"
            f"Input: {info.input_description}\nOutput: {info.output_description}")
        self._set_adapter_specific_visibility(info.adapter)
        self.set_status("Model ready", good=True)
        self._update_action_states()
        # Save known-good settings beside local model files.
        self.save_current_profile(silent=True)

    def on_model_unloaded(self):
        self.model_is_loaded = False
        self.model_info_label.setText("No model loaded")
        self.last_result = None
        self.set_status("Model unloaded")
        self._refresh_preview()
        self._update_action_states()

    def save_current_profile(self, silent: bool = False):
        source = self.model_path_edit.text().strip()
        if not source:
            if not silent:
                QMessageBox.information(self, "Profile", "Select a local model file first.")
            return
        data = self.current_settings()
        data["adapter"] = self.adapter_combo.currentText()
        data["labels_path"] = self.labels_path_edit.text().strip()
        path = save_profile(source, data)
        if path is None:
            if not silent:
                QMessageBox.information(self, "Profile", "Profiles are saved beside local model files. This source is not a local file.")
            return
        if not silent:
            self.set_status(f"Saved profile: {path.name}", good=True)

    def _apply_profile(self, p: dict):
        if p.get("adapter") in ADAPTER_NAMES:
            self.adapter_combo.setCurrentText(p["adapter"])
        self._set_device_value(str(p.get("device", "auto")))
        self.conf_spin.setValue(float(p.get("conf", self.conf_spin.value())))
        self.iou_spin.setValue(float(p.get("iou", self.iou_spin.value())))
        self.imgsz_spin.setValue(int(p.get("imgsz", self.imgsz_spin.value())))
        self.topk_spin.setValue(int(p.get("topk", self.topk_spin.value())))
        self.arch_combo.setCurrentText(str(p.get("architecture", self.arch_combo.currentText())))
        self.classifier_size_spin.setValue(int(p.get("classifier_size", self.classifier_size_spin.value())))
        self.num_classes_spin.setValue(int(p.get("num_classes", self.num_classes_spin.value()) or 0))
        self.normalize_checkbox.setChecked(bool(p.get("normalize", True)))
        self.imagenet_checkbox.setChecked(bool(p.get("imagenet_normalize", True)))
        self.rfdetr_variant_combo.setCurrentText(str(p.get("rfdetr_variant", "auto")))
        idx = self.fastsam_prompt_combo.findData(p.get("fastsam_prompt", "everything"))
        if idx >= 0:
            self.fastsam_prompt_combo.setCurrentIndex(idx)
        self.fastsam_text_edit.setText(str(p.get("fastsam_text", "")))
        self.text_prompts_edit.setPlainText(str(p.get("text_prompts", "")))
        labels_path = str(p.get("labels_path", ""))
        if labels_path and Path(labels_path).exists():
            self.labels_path_edit.setText(labels_path)
            self.loaded_labels = self._read_labels(Path(labels_path))

    def _set_device_value(self, value: str):
        idx = self.device_combo.findData(value)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    def current_settings(self) -> dict:
        return {
            "device": self.device_combo.currentData() or "auto",
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "topk": self.topk_spin.value(),
            "labels": list(self.loaded_labels),
            "normalize": self.normalize_checkbox.isChecked(),
            "architecture": self.arch_combo.currentText(),
            "classifier_size": self.classifier_size_spin.value(),
            "num_classes": self.num_classes_spin.value(),
            "imagenet_normalize": self.imagenet_checkbox.isChecked(),
            "rfdetr_variant": self.rfdetr_variant_combo.currentText(),
            "fastsam_prompt": self.fastsam_prompt_combo.currentData() or "everything",
            "fastsam_text": self.fastsam_text_edit.text().strip(),
            "retina_masks": self.retina_masks_checkbox.isChecked(),
            "text_prompts": self.text_prompts_edit.toPlainText(),
        }

    def _set_adapter_specific_visibility(self, adapter: str):
        self.classifier_group.setVisible(adapter in {"Auto", "PyTorch Classifier", "TensorFlow / Keras", "ONNX Runtime"})
        self.rfdetr_group.setVisible(adapter == "RF-DETR")
        self.fastsam_group.setVisible(adapter == "FastSAM")
        self.siglip_group.setVisible(adapter == "SigLIP / Transformers")

    # -------------------------------------------------------------- labels

    def choose_labels_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Labels", "", "Labels (*.txt *.json);;All files (*.*)")
        if not path:
            return
        try:
            labels = self._read_labels(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Labels", f"Could not read labels: {exc}")
            return
        self.labels_path_edit.setText(path)
        self.loaded_labels = labels
        if self.num_classes_spin.value() == 0 and labels:
            self.num_classes_spin.setValue(len(labels))
        self.set_status(f"Loaded {len(labels)} labels", good=True)

    @staticmethod
    def _read_labels(path: Path) -> list[str]:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                def keyfn(k):
                    try:
                        return int(k)
                    except Exception:
                        return 10**9
                return [str(data[k]) for k in sorted(data, key=keyfn)]
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ------------------------------------------------------------ input source

    def stop_sources(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.wait(3500)
            self.camera_thread = None
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait(2500)
            self.video_thread = None

    def refresh_cameras(self):
        self.set_status("Scanning cameras...")
        QApplication.processEvents()
        previous = self.camera_combo.currentData()
        names = directshow_camera_names()
        if names:
            cameras = list(range(len(names)))
        else:
            cameras = discover_camera_indices()
        self.camera_combo.clear()
        for idx in cameras:
            name = names[idx] if idx < len(names) else f"Camera {idx}"
            self.camera_combo.addItem(f"{idx}: {name}", idx)
        if not cameras:
            self.camera_combo.addItem("No camera found", None)
            self.set_status("No camera found")
        else:
            row = self.camera_combo.findData(previous)
            if row >= 0:
                self.camera_combo.setCurrentIndex(row)
            self.set_status(f"Found {len(cameras)} camera(s)", good=True)

    def open_selected_camera(self):
        idx = self.camera_combo.currentData()
        if idx is None:
            QMessageBox.information(self, "Camera", "No usable camera is selected.")
            return
        self.stop_sources()
        self.folder_images = []
        self.folder_index = -1
        self.current_camera_index = int(idx)
        self.current_source_kind = "camera"
        self.current_source_name = self.camera_combo.currentText()
        self.last_result = None
        mode = str(self.anti_flicker_combo.currentData() or "adaptive")
        thread = CameraThread(int(idx), mode, self)
        thread.frame_ready.connect(self.on_source_frame)
        thread.camera_ready.connect(self.on_camera_ready)
        thread.camera_error.connect(self.on_source_error)
        thread.probe_status.connect(self.on_source_status)
        thread.measured_fps.connect(lambda fps: self.camera_fps_label.setText(f"Source FPS: {fps:.1f}"))
        thread.anti_flicker_status.connect(self.on_anti_flicker_status)
        self.camera_thread = thread
        self.source_label.setText(f"Source: {self.current_source_name}")
        self.resolution_label.setText("Resolution: probing...")
        thread.start()

    def on_camera_ready(self, info: dict):
        self.active_width = int(info.get("width", 0))
        self.active_height = int(info.get("height", 0))
        self.resolution_label.setText(f"Resolution: {self.active_width} × {self.active_height} • {info.get('backend', 'OpenCV')}")
        self.set_status("Camera ready", good=True)
        self._update_action_states()

    def on_anti_flicker_changed(self):
        if self.camera_thread is not None:
            self.camera_thread.set_anti_flicker_mode(str(self.anti_flicker_combo.currentData() or "adaptive"))

    def recalibrate_anti_flicker(self):
        if self.camera_thread is not None:
            self.camera_thread.request_recalibration()

    def open_driver_settings(self):
        if self.camera_thread is not None:
            self.camera_thread.request_driver_settings()
        else:
            QMessageBox.information(self, "Camera", "Open a camera first.")

    def on_anti_flicker_status(self, info: dict):
        status = str(info.get("status", "—"))
        method = str(info.get("method", "—"))
        score = info.get("score")
        frames = int(info.get("averaged_frames", 0) or 0)
        target = int(info.get("target_frames", 0) or 0)
        extra = f" • score {float(score)*100:.1f}%" if score is not None else ""
        if target > 1 and frames < target:
            extra += f" • {frames}/{target} frames"
        self.anti_flicker_ready = bool(info.get("ready", False))
        self.flicker_status_label.setText(f"{method} • {status}{extra}")
        self.flicker_status_label.setToolTip(str(info.get("detail", "")))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if path:
            self.stop_sources()
            self.folder_images = []
            self.folder_index = -1
            self._load_still_image(Path(path), "image")

    def open_image_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if not directory:
            return
        paths = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            QMessageBox.information(self, "Folder", "No supported images found in this folder.")
            return
        self.stop_sources()
        self.folder_images = paths
        self.folder_index = 0
        self._load_still_image(paths[0], "folder")
        self._update_folder_position()

    def navigate_folder(self, delta: int):
        if not self.folder_images:
            return
        self.folder_index = max(0, min(len(self.folder_images) - 1, self.folder_index + delta))
        self._load_still_image(self.folder_images[self.folder_index], "folder")
        self._update_folder_position()

    def _update_folder_position(self):
        if not self.folder_images:
            self.folder_position_label.setText("—")
            return
        self.folder_position_label.setText(f"{self.folder_index+1}/{len(self.folder_images)}")

    def _load_still_image(self, path: Path, kind: str):
        frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            QMessageBox.critical(self, "Image", f"Could not read image: {path}")
            return
        self.current_source_kind = kind
        self.current_source_name = str(path)
        self.last_result = None
        self.on_source_frame(frame)
        self.source_label.setText(f"Source: {path.name}")
        self.camera_fps_label.setText("Source FPS: still")
        self.set_status("Image loaded", good=True)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*.*)")
        if not path:
            return
        self.stop_sources()
        self.folder_images = []
        self.folder_index = -1
        self.current_source_kind = "video"
        self.current_source_name = path
        self.last_result = None
        thread = VideoFileThread(path, self)
        thread.frame_ready.connect(self.on_source_frame)
        thread.info_ready.connect(self.on_video_info)
        thread.error.connect(self.on_source_error)
        thread.finished_playback.connect(lambda: self.set_status("Video finished"))
        self.video_thread = thread
        self.source_label.setText(f"Source: {Path(path).name}")
        thread.start()

    def on_video_info(self, info: dict):
        self.active_width = int(info.get("width", 0))
        self.active_height = int(info.get("height", 0))
        fps = float(info.get("fps", 0.0) or 0.0)
        self.resolution_label.setText(f"Resolution: {self.active_width} × {self.active_height}")
        self.camera_fps_label.setText(f"Source FPS: {fps:.1f}" if fps else "Source FPS: —")
        self.set_status("Video playing", good=True)

    def on_source_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return
        self.current_frame = frame.copy()
        h, w = frame.shape[:2]
        self.active_width, self.active_height = w, h
        if self.current_source_kind not in {"camera", "video"}:
            self.resolution_label.setText(f"Resolution: {w} × {h}")
        self._refresh_preview()
        self._update_action_states()
        if self.live_checkbox.isChecked() and self.model_is_loaded:
            self._submit_current()

    def on_source_error(self, message: str):
        self.set_status("Source error", bad=True)
        QMessageBox.critical(self, "Input Source", message)

    def on_source_status(self, text: str):
        self.set_status(text)

    # -------------------------------------------------------------- ROI/result

    def on_roi_changed(self, roi: Optional[QRectF]):
        if roi is None or self.active_width <= 0 or self.active_height <= 0:
            self.roi_label.setText("ROI: none")
        else:
            x = int(round(roi.left() * self.active_width))
            y = int(round(roi.top() * self.active_height))
            w = int(round(roi.width() * self.active_width))
            h = int(round(roi.height() * self.active_height))
            self.roi_label.setText(f"ROI: {w}×{h} @ {x},{y}")
        self._refresh_preview()

    def _roi_tuple(self):
        roi = self.preview.roi_norm
        if roi is None:
            return None
        return (roi.left(), roi.top(), roi.width(), roi.height())

    def _refresh_preview(self):
        if self.current_frame is None:
            return
        display = render_result(self.current_frame, self.last_result)
        self.preview.set_frame(display)

    def clear_results(self):
        self.last_result = None
        self.result_text.clear()
        self.preprocess_label.setText("Pre: —")
        self.infer_label.setText("Infer: —")
        self.post_label.setText("Post: —")
        self.total_label.setText("Total: —")
        self.model_fps_label.setText("Model FPS: —")
        self.result_count_label.setText("Results: —")
        self._refresh_preview()

    # -------------------------------------------------------------- inference

    def capture_and_infer(self):
        if not self.model_is_loaded:
            QMessageBox.information(self, "Inference", "Load a model first.")
            return
        if self.current_frame is None:
            QMessageBox.information(self, "Inference", "Open a camera/image/video first.")
            return
        self._submit_current()

    def _submit_current(self):
        if self.current_frame is None:
            return
        self.inference_thread.submit_latest(
            self.current_frame,
            self.current_settings(),
            self._roi_tuple(),
            self.use_roi_checkbox.isChecked(),
        )

    def on_live_toggled(self, checked: bool):
        if checked and (not self.model_is_loaded or self.current_frame is None):
            self.live_checkbox.blockSignals(True)
            self.live_checkbox.setChecked(False)
            self.live_checkbox.blockSignals(False)
            QMessageBox.information(self, "Live Inference", "Load a model and open an input source first.")
            return
        if checked:
            self.set_status("Live inference enabled", good=True)
            self._submit_current()
        else:
            self.set_status("Live inference stopped")

    def on_inference_result(self, result: InferenceResult):
        self.last_result = result
        self._refresh_preview()
        self.preprocess_label.setText(f"Pre: {result.preprocess_ms:.2f} ms")
        self.infer_label.setText(f"Infer: {result.inference_ms:.2f} ms")
        self.post_label.setText(f"Post: {result.postprocess_ms:.2f} ms")
        self.total_label.setText(f"Total: {result.total_ms:.2f} ms")
        fps = 1000.0 / result.total_ms if result.total_ms > 0 else 0.0
        self.model_fps_label.setText(f"Model FPS: {fps:.1f}")
        count = len(result.detections) or len(result.masks) or len(result.classifications)
        self.result_count_label.setText(f"Results: {count}")
        lines = [
            f"Adapter: {result.adapter} | Task: {result.task} | Device: {result.device}",
            result.summary or "Inference completed",
        ]
        if result.classifications:
            lines.append("Top results:")
            lines.extend(f"  {i+1}. {c.label}: {c.score*100:.2f}%" for i, c in enumerate(result.classifications[:10]))
        if "cosine_similarity" in result.metadata:
            lines.append("Raw cosine similarity:")
            lines.extend(f"  {k}: {v:.4f}" for k, v in result.metadata["cosine_similarity"].items())
        if result.metadata.get("output_shapes"):
            lines.append(f"ONNX outputs: {result.metadata['output_shapes']}")
        self.result_text.setPlainText("\n".join(lines))
        self.set_status("Inference complete", good=True)

    def on_inference_error(self, message: str):
        self.set_status("Inference/model error", bad=True)
        self.result_text.setPlainText(message)
        QMessageBox.critical(self, "Model / Inference Error", message)

    def on_inference_status(self, text: str):
        self.set_status(text)

    def on_inference_busy(self, busy: bool):
        self.inference_busy = busy
        self.benchmark_button.setEnabled(not busy and self.model_is_loaded and self.current_frame is not None)

    def run_benchmark(self):
        if not self.model_is_loaded or self.current_frame is None:
            QMessageBox.information(self, "Benchmark", "Load a model and input first.")
            return
        self.live_checkbox.setChecked(False)
        self.inference_thread.benchmark(
            self.current_frame, self.current_settings(), self._roi_tuple(), self.use_roi_checkbox.isChecked(),
            runs=20, warmup=3)

    def on_benchmark_result(self, result: BenchmarkResult):
        text = (
            f"Benchmark ({result.runs} runs, {result.warmup_runs} warmup)\n"
            f"Average: {result.average_ms:.2f} ms  |  Median: {result.median_ms:.2f} ms\n"
            f"Min: {result.minimum_ms:.2f} ms  |  Max: {result.maximum_ms:.2f} ms  |  P95: {result.p95_ms:.2f} ms\n"
            f"Throughput: {result.fps:.2f} FPS"
        )
        self.result_text.setPlainText(text)
        self.set_status("Benchmark complete", good=True)

    def save_annotated_result(self):
        if self.current_frame is None:
            return
        annotated = render_result(self.current_frame, self.last_result)
        path, selected = QFileDialog.getSaveFileName(self, "Save Annotated Image", "inference_result.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        ext = Path(path).suffix.lower() or ".png"
        if not Path(path).suffix:
            path += ".png"
            ext = ".png"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg", ".jpeg"} else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ok, enc = cv2.imencode(ext, annotated, params)
        if ok:
            enc.tofile(path)
            self.set_status(f"Saved {Path(path).name}", good=True)
        else:
            QMessageBox.critical(self, "Save", "OpenCV could not encode the annotated image.")

    # --------------------------------------------------------------- utility

    def _update_action_states(self):
        has_frame = self.current_frame is not None
        self.capture_infer_button.setEnabled(self.model_is_loaded and has_frame)
        self.benchmark_button.setEnabled(self.model_is_loaded and has_frame and not self.inference_busy)
        self.unload_model_button.setEnabled(self.model_is_loaded)
        self.prev_image_button.setEnabled(bool(self.folder_images) and self.folder_index > 0)
        self.next_image_button.setEnabled(bool(self.folder_images) and self.folder_index < len(self.folder_images) - 1)

    def set_status(self, text: str, good: bool = False, bad: bool = False):
        self.status_badge.setText(text)
        if good:
            self.status_badge.setStyleSheet("background:#12352D;border:1px solid #1C735D;color:#A9F3DE;border-radius:11px;padding:7px 12px;")
        elif bad:
            self.status_badge.setStyleSheet("background:#3A1C20;border:1px solid #7A343D;color:#FFC5CC;border-radius:11px;padding:7px 12px;")
        else:
            self.status_badge.setStyleSheet("")

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self.live_checkbox.setChecked(False)
            self.stop_sources()
            self.inference_thread.stop()
        if self.inference_thread.isRunning():
            # A native framework load cannot be safely killed mid-call. Keep the
            # window alive until it returns so Qt never destroys a running QThread.
            self.set_status("Waiting for model loader to stop...")
            event.ignore()
            QTimer.singleShot(250, self.close)
            return
        event.accept()

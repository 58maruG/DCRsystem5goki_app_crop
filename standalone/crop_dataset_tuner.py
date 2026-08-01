"""
既存データセット用クロップチューナー（HSV検出ゲート調整＋一括クロップ実行）

シーズン終了で新規のライブカメラ映像が撮れないため、`hsv_calibration.py`の
ライブカメラプレビューを「既存の生果実画像フォルダ」に差し替えたもの。
スライダーで調整するHSVゲート・虚像除去・果柄除去・最小面積・推論ゲートの
定義とget_target_info同一手順の再現(analyze関数)はhsv_calibration.pyと共通。

hsv_calibration.py同様、module_yolo_csv4_v3(ultralytics/torch)は読み込まない
軽量ツール。crop_from_stat()はmodule_yolo_csv4_v3.ImageProcessor.dynamic_cropと
同じ計算式をここで再実装している（変更する場合は両方揃えること）。

使い方:
  1. 右側のタブでカメラを選び、画像を送りながらスライダーでHSVゲートを調整する
     （「次の失敗へ」で検出できていない画像に素早く移動できる）
  2. 「保存」でjson/hsv_config_{cam}.jsonへ書き込む（本番の get_target_info と共有）
  3. 下部の入出力フォルダを確認し、「一括クロップ実行」で全カテゴリを処理する。
     一括実行は各カメラの「保存済みJSON」を使う（未保存のスライダー値は反映されない）。
     ファイル名にカメラ名が無い画像は、ネイティブ解像度640・既定値HSVで自動処理する
     （どのカメラの較正か特定できないため）。
  4. 検出できた画像はクロップ先へ640x640で保存、できなかった画像は元のまま除外先へコピーする。
"""

import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QSpinBox,
    QGroupBox, QGridLayout, QTabWidget, QComboBox,
    QLineEdit, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from hsv_mask_utils import (  # noqa: E402
    mask_from_hsv, remove_stem, remove_reflection_sat,
)

# ==========================================================
# 設定（module_yolo_csv4_v3 と同じ既定値。変更する場合は両方揃えること）
# ==========================================================
PREVIEW_SIZE = 480

HSV_JSON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json")

DEFAULT_CFG = {
    "lower1": [0, 100, 100],   "upper1": [32, 255, 255],
    "lower2": [160, 100, 100], "upper2": [180, 255, 255],
    "reflect_sat_lo": 0, "reflect_sat_hi": 255,
    "reflect_v_lo":   0, "reflect_v_hi":   255,
    "stem_open":      0,
    "min_blob_area":  500,
}

DEFAULT_FRUIT_SPEED_PX = {
    "cam_top": 87, "cam_under": 87, "cam_inside": 91, "cam_outside": 87,
}
FALLBACK_FRUIT_SPEED_PX = 87
INFER_FRAMES_PER_CAM = 5   # module_yolo_csv4_v3 側と必ず揃えること
YOLO_IMG_SIZE = 640        # module_yolo_csv4_v3.YOLO_IMG_SIZE と必ず揃えること

# 4カメラのタブ順（cam_name, ラベル）
CAMERAS = [
    ("cam_inside",  "inside"),
    ("cam_outside", "outside"),
    ("cam_under",   "under"),
    ("cam_top",     "top"),
]

# 保存時の640x640への引き伸ばし元＝カメラのネイティブ解像度（現物のcam_pfsで確認済み）
NATIVE_SIZE = {"cam_top": 640, "cam_under": 640, "cam_inside": 560, "cam_outside": 500}
DEFAULT_NATIVE = 640  # カメラ名が特定できない画像はcam_top/cam_underと同じ640ネイティブとみなす

CAM_RE = re.compile(r"cam_(top|under|inside|outside)")

GROUP_DEFS = [
    ("HSVマスク（1段目）", [
        ("H1 最小", "h1_lo", 0, 180, "赤色範囲1の色相 下限"),
        ("H1 最大", "h1_hi", 0, 180, "赤色範囲1の色相 上限"),
        ("H2 最小", "h2_lo", 0, 180, "赤色範囲2の色相 下限"),
        ("H2 最大", "h2_hi", 0, 180, "赤色範囲2の色相 上限"),
        ("S  最小", "s_lo",  0, 255, "彩度の下限（H1/H2共通）。上げると淡い色を拾わなくなる"),
        ("S  最大", "s_hi",  0, 255, "彩度の上限（H1/H2共通）"),
        ("V  最小", "v_lo",  0, 255, "明度の下限（H1/H2共通）。上げると暗部を拾わなくなる"),
        ("V  最大", "v_hi",  0, 255, "明度の上限（H1/H2共通）"),
    ]),
    ("虚像除去（アクリル反射）", [
        ("S 下限", "reflect_sat_lo", 0, 255, "虚像とみなす彩度の下限"),
        ("S 上限", "reflect_sat_hi", 0, 255, "虚像とみなす彩度の上限。255のままだと彩度条件は無効"),
        ("V 下限", "reflect_v_lo",   0, 255, "虚像とみなす明度の下限。0のままだと明度条件は無効"),
        ("V 上限", "reflect_v_hi",   0, 255, "虚像とみなす明度の上限"),
    ]),
    ("果柄除去・面積", [
        ("果柄 半径", "stem_open",     0,  40, "開処理の半径。これより細い突起を切り離す。0で無効"),
        ("最小面積",  "min_blob_area", 0, 50000, "これ未満のブロブは果実とみなさない"),
    ]),
    ("推論ゲート", [
        ("搬送速度", "fruit_speed_px", 1, 300,
         "果実が1フレームで進む距離[px]。推論する中心窓の半幅＝速度×枚数÷2 で決まる"),
    ]),
]
ALL_KEYS = [d[1] for _, defs in GROUP_DEFS for d in defs]
VIEW_MODES = ["検出結果", "1段目マスク", "虚像除去後", "果柄除去後"]
EDGE_MARGIN = 5  # get_target_info と同値


# ==========================================================
# 日本語パス対応 I/O
# ==========================================================
def imread_ja(path: str):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_ja(path: str, img) -> bool:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(path)
    return ok


# ==========================================================
# JSON 入出力（hsv_calibration.py と同一仕様）
# ==========================================================
def config_path(cam_name: str) -> str:
    return os.path.join(HSV_JSON_DIR, f"hsv_config_{cam_name}.json")


def load_raw_config(cam_name: str) -> dict:
    path = config_path(cam_name)
    if not os.path.exists(path):
        print(f"[{cam_name}] 設定ファイルが見つかりません（既定値で開始）: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print(f"[{cam_name}] 設定の読込に失敗しました（既定値で開始）: {path} - {e}")
        return {}


def save_config(cam_name: str, raw: dict) -> str:
    path = config_path(cam_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=4, ensure_ascii=False)
    return path


def flat_from_raw(raw: dict, cam_name: str | None) -> dict:
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in raw.items() if k in DEFAULT_CFG})
    speed = raw.get("fruit_speed_px",
                    DEFAULT_FRUIT_SPEED_PX.get(cam_name, FALLBACK_FRUIT_SPEED_PX))
    return {
        "h1_lo": cfg["lower1"][0], "h1_hi": cfg["upper1"][0],
        "h2_lo": cfg["lower2"][0], "h2_hi": cfg["upper2"][0],
        "s_lo":  cfg["lower1"][1], "s_hi":  cfg["upper1"][1],
        "v_lo":  cfg["lower1"][2], "v_hi":  cfg["upper1"][2],
        "reflect_sat_lo": int(cfg["reflect_sat_lo"]),
        "reflect_sat_hi": int(cfg["reflect_sat_hi"]),
        "reflect_v_lo":   int(cfg["reflect_v_lo"]),
        "reflect_v_hi":   int(cfg["reflect_v_hi"]),
        "stem_open":      int(cfg["stem_open"]),
        "min_blob_area":  int(cfg["min_blob_area"]),
        "fruit_speed_px": int(speed),
    }


def raw_from_flat(flat: dict, base: dict, roi_width: int | None) -> dict:
    raw = dict(base)
    raw.update({
        "lower1": [flat["h1_lo"], flat["s_lo"], flat["v_lo"]],
        "upper1": [flat["h1_hi"], flat["s_hi"], flat["v_hi"]],
        "lower2": [flat["h2_lo"], flat["s_lo"], flat["v_lo"]],
        "upper2": [flat["h2_hi"], flat["s_hi"], flat["v_hi"]],
        "reflect_sat_lo": flat["reflect_sat_lo"],
        "reflect_sat_hi": flat["reflect_sat_hi"],
        "reflect_v_lo":   flat["reflect_v_lo"],
        "reflect_v_hi":   flat["reflect_v_hi"],
        "stem_open":      flat["stem_open"],
        "min_blob_area":  flat["min_blob_area"],
        "fruit_speed_px": flat["fruit_speed_px"],
    })
    if roi_width:
        raw["roi_width_at_tuning"] = int(roi_width)
    return raw


def mask_params(flat: dict) -> dict:
    return {
        "lower1": [flat["h1_lo"], flat["s_lo"], flat["v_lo"]],
        "upper1": [flat["h1_hi"], flat["s_hi"], flat["v_hi"]],
        "lower2": [flat["h2_lo"], flat["s_lo"], flat["v_lo"]],
        "upper2": [flat["h2_hi"], flat["s_hi"], flat["v_hi"]],
    }


def window_half_px(speed: int) -> int:
    return max(1, int(round(speed * INFER_FRAMES_PER_CAM / 2.0)))


# ==========================================================
# 画像処理（get_target_info と同一手順。hsv_calibration.py と共通）
# ==========================================================
def analyze(frame: np.ndarray, flat: dict) -> dict:
    """1フレームを本番と同じ手順で処理し、各段階のマスクと判定結果を返す。

    status: 'ok'（検出あり） / 'no_blob'（候補なし） / 'too_small'（面積不足）
            / 'edge'（画面端に接触して棄却）"""
    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    res = {
        "status": "no_blob", "area": 0, "stat": None,
        "mask1": None, "mask_reflect": None, "mask_final": None,
        "labels": None, "max_index": 0, "in_band": False,
    }

    mask1 = mask_from_hsv(hsv, mask_params(flat))
    res["mask1"] = res["mask_reflect"] = res["mask_final"] = mask1

    n1, labels1, stats1, cents1 = cv2.connectedComponentsWithStats(mask1)
    if n1 <= 1:
        return res

    idx = int(np.argmax(stats1[1:, cv2.CC_STAT_AREA])) + 1
    min_area = int(flat["min_blob_area"])
    if stats1[idx, cv2.CC_STAT_AREA] < min_area:
        res["status"] = "too_small"
        res["area"]   = int(stats1[idx, cv2.CC_STAT_AREA])
        res["stat"]   = stats1[idx]
        return res

    s_active = not (flat["reflect_sat_lo"] <= 0 and flat["reflect_sat_hi"] >= 255)
    v_active = not (flat["reflect_v_lo"]   <= 0 and flat["reflect_v_hi"]   >= 255)
    refine   = bool(flat["stem_open"] > 0 or s_active or v_active)

    if not refine:
        s = stats1[idx]
        res["area"] = int(s[cv2.CC_STAT_AREA])
        res["stat"] = s
        if _touches_edge(s, w, h):
            res["status"] = "edge"
            return res
        res.update(status="ok", labels=labels1, max_index=idx,
                   mx=int(cents1[idx][0]), my=int(cents1[idx][1]))
        res["in_band"] = _check_band(res["mx"], w, window_half_px(flat["fruit_speed_px"]))
        return res

    pad = max(int(flat["stem_open"]), 4) + 1
    bx, by = int(stats1[idx, cv2.CC_STAT_LEFT]),  int(stats1[idx, cv2.CC_STAT_TOP])
    bw, bh = int(stats1[idx, cv2.CC_STAT_WIDTH]), int(stats1[idx, cv2.CC_STAT_HEIGHT])
    x0, y0 = max(0, bx - pad),      max(0, by - pad)
    x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)

    sub = mask1[y0:y1, x0:x1]
    sub_r = remove_reflection_sat(
        sub, hsv[y0:y1, x0:x1, 1], hsv[y0:y1, x0:x1, 2],
        flat["reflect_sat_lo"], flat["reflect_sat_hi"],
        flat["reflect_v_lo"],   flat["reflect_v_hi"])
    sub_s = remove_stem(sub_r, int(flat["stem_open"]))

    res["mask_reflect"] = _paste(sub_r, h, w, x0, y0)
    res["mask_final"]   = _paste(sub_s, h, w, x0, y0)

    n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(sub_s)
    if n2 <= 1:
        return res

    j  = int(np.argmax(st2[1:, cv2.CC_STAT_AREA])) + 1
    s2 = st2[j].copy()
    res["area"] = int(s2[cv2.CC_STAT_AREA])
    s2[cv2.CC_STAT_LEFT] += x0
    s2[cv2.CC_STAT_TOP]  += y0
    res["stat"] = s2

    if res["area"] < min_area:
        res["status"] = "too_small"
        return res
    if _touches_edge(s2, w, h):
        res["status"] = "edge"
        return res

    labels_full = np.zeros((h, w), dtype=np.int32)
    labels_full[y0:y1, x0:x1] = (lab2 == j)
    res.update(status="ok", labels=labels_full, max_index=1,
               mx=int(ce2[j][0]) + x0, my=int(ce2[j][1]) + y0)
    res["in_band"] = _check_band(res["mx"], w, window_half_px(flat["fruit_speed_px"]))
    return res


def _paste(sub, h, w, x0, y0):
    full = np.zeros((h, w), dtype=np.uint8)
    full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
    return full


def _touches_edge(stat, w, h) -> bool:
    return bool(stat[0] <= EDGE_MARGIN or stat[1] <= EDGE_MARGIN
                or (stat[0] + stat[2]) >= (w - EDGE_MARGIN)
                or (stat[1] + stat[3]) >= (h - EDGE_MARGIN))


def band_lines(w, half):
    cx = w / 2.0
    xl = min(max(int(round(cx - half)), 0), w - 1)
    xr = min(max(int(round(cx + half)), 0), w - 1)
    return xl, xr


def _check_band(mx, w, half) -> bool:
    return abs(mx - w / 2.0) <= half


def render(frame: np.ndarray, res: dict, flat: dict, mode: str) -> np.ndarray:
    if mode == "1段目マスク":
        base = cv2.cvtColor(res["mask1"], cv2.COLOR_GRAY2BGR)
    elif mode == "虚像除去後":
        base = cv2.cvtColor(res["mask_reflect"], cv2.COLOR_GRAY2BGR)
    elif mode == "果柄除去後":
        base = cv2.cvtColor(res["mask_final"], cv2.COLOR_GRAY2BGR)
    else:
        base = frame.copy()

    h, w = base.shape[:2]
    xl, xr = band_lines(w, window_half_px(flat["fruit_speed_px"]))
    band_color = (0, 255, 0) if res["in_band"] else (0, 165, 255)
    cv2.line(base, (xl, 0), (xl, h - 1), band_color, 2)
    cv2.line(base, (xr, 0), (xr, h - 1), band_color, 2)

    stat = res["stat"]
    if stat is not None:
        color = {"ok": (0, 255, 0), "too_small": (0, 0, 255),
                 "edge": (0, 255, 255)}.get(res["status"], (0, 0, 255))
        x, y, bw, bh = int(stat[0]), int(stat[1]), int(stat[2]), int(stat[3])
        cv2.rectangle(base, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(base, f"{res['area']:,}", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if res.get("mx") is not None:
        cv2.drawMarker(base, (int(res["mx"]), int(res["my"])), band_color,
                       cv2.MARKER_CROSS, 26, 2)
    return base


def to_pixmap(frame: np.ndarray, size: int) -> QPixmap:
    rgb = np.ascontiguousarray(
        cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB))
    img = QImage(rgb.data, size, size, 3 * size, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


# ==========================================================
# クロップ（module_yolo_csv4_v3.ImageProcessor.dynamic_crop と同一式）
# ==========================================================
def crop_box_from_stat(stat, mx, my, w, h) -> tuple[int, int, int, int]:
    size = min(int(max(stat[2], stat[3]) * 1.5), w, h)
    x1 = max(0, mx - size // 2)
    y1 = max(0, my - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    if x2 == w:
        x1 = max(0, w - size)
    if y2 == h:
        y1 = max(0, h - size)
    return x1, y1, x2, y2


def apply_crop(frame: np.ndarray, res: dict) -> np.ndarray | None:
    if res["status"] != "ok" or res["stat"] is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_box_from_stat(res["stat"], res["mx"], res["my"], w, h)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))


# ==========================================================
# 画像ブラウザ（カメラ別ファイル一覧・ネイティブ解像度への読込）
# ==========================================================
def scan_dataset(root: str) -> dict[str, list[str]]:
    """root配下(カテゴリ別サブフォルダ)を再帰的に走査し、カメラ別ファイル一覧を返す。
    カメラ名不明の画像はブラウズ対象に含めない（一括実行では既定値で自動処理する）。"""
    result: dict[str, list[str]] = {cam: [] for cam, _ in CAMERAS}
    if not root or not os.path.isdir(root):
        return result
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if not f.lower().endswith(".jpg"):
                continue
            m = CAM_RE.search(f)
            if not m:
                continue
            cam_name = f"cam_{m.group(1)}"
            result[cam_name].append(os.path.join(cls_dir, f))
    return result


def load_native(path: str, cam_name: str) -> np.ndarray | None:
    img = imread_ja(path)
    if img is None:
        return None
    native = NATIVE_SIZE.get(cam_name, DEFAULT_NATIVE)
    if img.shape[0] != native or img.shape[1] != native:
        img = cv2.resize(img, (native, native), interpolation=cv2.INTER_AREA)
    return img


# ==========================================================
# 1カメラ分のスライダーパネル
# ==========================================================
class ParamPanel(QWidget):
    def __init__(self, cam_name: str, on_change=None):
        super().__init__()
        self.cam_name = cam_name
        self._on_change_cb = on_change
        self._raw = load_raw_config(cam_name)
        self._flat = flat_from_raw(self._raw, cam_name)
        self._roi_width: int | None = None
        self._sliders: dict[str, QSlider] = {}
        self._spins: dict[str, QSpinBox] = {}
        self._build_ui()
        self._apply_flat_to_widgets()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        for group_name, defs in GROUP_DEFS:
            grp = QGroupBox(group_name)
            grid = QGridLayout(grp)
            grid.setContentsMargins(6, 4, 6, 4)
            grid.setSpacing(3)
            grid.setColumnStretch(1, 1)

            for row, (name, key, lo, hi, tip) in enumerate(defs):
                lbl = QLabel(name)
                lbl.setFixedWidth(72)
                lbl.setToolTip(tip)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(lo, hi)
                slider.setFixedHeight(18)
                slider.setToolTip(tip)

                spin = QSpinBox()
                spin.setRange(lo, hi)
                spin.setFixedWidth(62)
                spin.setButtonSymbols(QSpinBox.NoButtons)
                spin.setToolTip(tip)

                slider.valueChanged.connect(spin.setValue)
                spin.valueChanged.connect(slider.setValue)
                slider.valueChanged.connect(self._on_change)

                self._sliders[key] = slider
                self._spins[key] = spin

                grid.addWidget(lbl,    row, 0)
                grid.addWidget(slider, row, 1)
                grid.addWidget(spin,   row, 2)

            layout.addWidget(grp)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("保存")
        btn_save.setFixedHeight(30)
        btn_save.clicked.connect(self.save)
        btn_reload = QPushButton("JSON再読込")
        btn_reload.setFixedHeight(30)
        btn_reload.clicked.connect(self.reload)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_reload)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

    def _apply_flat_to_widgets(self):
        for key in ALL_KEYS:
            val = int(self._flat[key])
            slider, spin = self._sliders[key], self._spins[key]
            val = max(slider.minimum(), min(slider.maximum(), val))
            slider.blockSignals(True)
            spin.blockSignals(True)
            slider.setValue(val)
            spin.setValue(val)
            slider.blockSignals(False)
            spin.blockSignals(False)
        self._on_change()

    def _on_change(self):
        self._flat = {key: self._sliders[key].value() for key in ALL_KEYS}
        if self._on_change_cb:
            self._on_change_cb()

    def set_roi_width(self, width: int):
        if self._roi_width == width:
            return
        self._roi_width = width
        upper = max(1, int(width / INFER_FRAMES_PER_CAM))
        for w in (self._sliders["fruit_speed_px"], self._spins["fruit_speed_px"]):
            w.blockSignals(True)
            w.setMaximum(upper)
            w.blockSignals(False)
        self._sliders["fruit_speed_px"].setValue(
            min(self._sliders["fruit_speed_px"].value(), upper))
        self._on_change()

    def get_flat(self) -> dict:
        return self._flat

    def save(self):
        try:
            raw = raw_from_flat(self._flat, self._raw, self._roi_width)
            path = save_config(self.cam_name, raw)
            self._raw = raw
            self._status.setText(f"保存しました: {os.path.basename(path)}")
            self._status.setStyleSheet("color:#00aa33;")
            return True
        except Exception as e:
            self._status.setText(f"保存に失敗しました: {e}")
            self._status.setStyleSheet("color:#cc0000;")
            print(f"[{self.cam_name}] 保存に失敗しました: {e}")
            return False

    def reload(self):
        self._raw = load_raw_config(self.cam_name)
        self._flat = flat_from_raw(self._raw, self.cam_name)
        self._apply_flat_to_widgets()
        self._status.setText("JSONから再読込しました")
        self._status.setStyleSheet("color:#555555;")


# ==========================================================
# メインウィンドウ
# ==========================================================
class CropTunerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("既存データセット用クロップチューナー")
        self.resize(1500, 950)

        self._panels: dict[str, ParamPanel] = {}
        self._files: dict[str, list[str]] = {cam: [] for cam, _ in CAMERAS}
        self._index: dict[str, int] = {cam: 0 for cam, _ in CAMERAS}
        self._cur_cam = CAMERAS[0][0]
        self._cur_frame: np.ndarray | None = None
        self._cur_res: dict | None = None

        self._build_ui()
        self._rescan()

    # --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── 入力元フォルダ ──────────────────────────────
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("入力データセット:"))
        self._src_edit = QLineEdit(
            r"C:\Users\kotan\gohara\cherry_yolo\model作成用imageset\all_true")
        src_row.addWidget(self._src_edit)
        btn_src = QPushButton("参照...")
        btn_src.clicked.connect(self._browse_src)
        src_row.addWidget(btn_src)
        btn_rescan = QPushButton("再スキャン")
        btn_rescan.clicked.connect(self._rescan)
        src_row.addWidget(btn_rescan)
        root.addLayout(src_row)

        body = QHBoxLayout()
        body.setSpacing(8)
        root.addLayout(body, stretch=1)

        # ── 左: プレビュー ──────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("表示モード:"))
        self._mode = QComboBox()
        self._mode.addItems(VIEW_MODES)
        self._mode.currentTextChanged.connect(self._refresh_preview)
        mode_row.addWidget(self._mode)
        mode_row.addStretch()
        left.addLayout(mode_row)

        preview_row = QHBoxLayout()
        det_box = QVBoxLayout()
        det_box.addWidget(QLabel("検出結果"), alignment=Qt.AlignCenter)
        self._det_preview = QLabel()
        self._det_preview.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self._det_preview.setStyleSheet("background:#1a1a1a;")
        det_box.addWidget(self._det_preview)
        preview_row.addLayout(det_box)

        crop_box = QVBoxLayout()
        crop_box.addWidget(QLabel("クロップ後(640x640)"), alignment=Qt.AlignCenter)
        self._crop_preview = QLabel()
        self._crop_preview.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self._crop_preview.setAlignment(Qt.AlignCenter)
        self._crop_preview.setStyleSheet("background:#1a1a1a; color:#888;")
        crop_box.addWidget(self._crop_preview)
        preview_row.addLayout(crop_box)
        left.addLayout(preview_row)

        self._filename_lbl = QLabel("-")
        self._filename_lbl.setWordWrap(True)
        left.addWidget(self._filename_lbl)

        self._status_lbl = QLabel("-")
        self._status_lbl.setStyleSheet("font-size:13px; font-weight:bold;")
        left.addWidget(self._status_lbl)

        nav_row = QHBoxLayout()
        btn_prev = QPushButton("← 前")
        btn_prev.clicked.connect(lambda: self._step(-1))
        btn_next = QPushButton("次 →")
        btn_next.clicked.connect(lambda: self._step(1))
        btn_fail = QPushButton("次の失敗へ")
        btn_fail.clicked.connect(self._jump_next_fail)
        self._idx_lbl = QLabel("0 / 0")
        nav_row.addWidget(btn_prev)
        nav_row.addWidget(btn_next)
        nav_row.addWidget(btn_fail)
        nav_row.addWidget(self._idx_lbl)
        left.addLayout(nav_row)
        left.addStretch()
        body.addLayout(left, stretch=1)

        # ── 右: カメラ別スライダー ──────────────────────
        right = QVBoxLayout()
        self._tabs = QTabWidget()
        self._tabs.setFixedWidth(340)
        for cam_name, label in CAMERAS:
            panel = ParamPanel(cam_name, on_change=self._on_param_change)
            self._panels[cam_name] = panel
            self._tabs.addTab(panel, label)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        right.addWidget(self._tabs)
        right.addStretch()
        body.addLayout(right)

        # ── 下: 一括クロップ実行 ────────────────────────
        batch_box = QGroupBox("一括クロップ実行（各カメラの保存済みJSONを使用）")
        batch_layout = QVBoxLayout(batch_box)

        dst_row = QHBoxLayout()
        dst_row.addWidget(QLabel("クロップ先:"))
        self._dst_edit = QLineEdit(
            r"C:\Users\kotan\gohara\cherry_yolo\model作成用imageset\all_true_cropped")
        dst_row.addWidget(self._dst_edit)
        btn_dst = QPushButton("参照...")
        btn_dst.clicked.connect(lambda: self._browse_dir(self._dst_edit))
        dst_row.addWidget(btn_dst)
        batch_layout.addLayout(dst_row)

        exc_row = QHBoxLayout()
        exc_row.addWidget(QLabel("除外先:"))
        self._exc_edit = QLineEdit(
            r"C:\Users\kotan\gohara\cherry_yolo\model作成用imageset\all_true_excluded")
        exc_row.addWidget(self._exc_edit)
        btn_exc = QPushButton("参照...")
        btn_exc.clicked.connect(lambda: self._browse_dir(self._exc_edit))
        exc_row.addWidget(btn_exc)
        batch_layout.addLayout(exc_row)

        run_row = QHBoxLayout()
        btn_run = QPushButton("▶ 一括クロップ実行")
        btn_run.setFixedHeight(34)
        btn_run.clicked.connect(self._run_batch)
        run_row.addWidget(btn_run)
        self._batch_status = QLabel("")
        run_row.addWidget(self._batch_status, stretch=1)
        batch_layout.addLayout(run_row)

        root.addWidget(batch_box)

        self._sb = self.statusBar()

    # --------------------------------------------------
    def _browse_src(self):
        d = QFileDialog.getExistingDirectory(self, "入力データセットフォルダ", self._src_edit.text())
        if d:
            self._src_edit.setText(d)
            self._rescan()

    def _browse_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "フォルダを選択", edit.text())
        if d:
            edit.setText(d)

    def _rescan(self):
        self._files = scan_dataset(self._src_edit.text())
        for cam in self._index:
            self._index[cam] = 0
        counts = ", ".join(f"{c}:{len(self._files[c])}枚" for c, _ in CAMERAS)
        self._sb.showMessage(f"スキャン完了 — {counts}")
        self._load_current()

    def _on_tab_changed(self, i: int):
        self._cur_cam = CAMERAS[i][0]
        self._load_current()

    def _on_param_change(self):
        # 表示中のパネルの変更だけプレビューへ反映する
        if self._tabs.currentWidget() is self._panels.get(self._cur_cam):
            self._recompute_and_render()

    # --------------------------------------------------
    def _load_current(self):
        files = self._files.get(self._cur_cam, [])
        n = len(files)
        self._idx_lbl.setText(f"{(self._index[self._cur_cam] + 1) if n else 0} / {n}")
        if n == 0:
            self._cur_frame = None
            self._cur_res = None
            self._filename_lbl.setText("(このカメラの画像が見つかりません)")
            self._status_lbl.setText("-")
            self._det_preview.clear()
            self._crop_preview.clear()
            return

        idx = self._index[self._cur_cam] % n
        self._index[self._cur_cam] = idx
        path = files[idx]
        self._filename_lbl.setText(path)

        frame = load_native(path, self._cur_cam)
        if frame is None:
            self._status_lbl.setText("画像の読込に失敗しました")
            return
        self._cur_frame = frame
        self._panels[self._cur_cam].set_roi_width(frame.shape[1])
        self._recompute_and_render()

    def _step(self, delta: int):
        files = self._files.get(self._cur_cam, [])
        if not files:
            return
        self._index[self._cur_cam] = (self._index[self._cur_cam] + delta) % len(files)
        self._load_current()

    def _jump_next_fail(self):
        files = self._files.get(self._cur_cam, [])
        n = len(files)
        if n == 0:
            return
        flat = self._panels[self._cur_cam].get_flat()
        start = self._index[self._cur_cam]
        for step in range(1, n + 1):
            idx = (start + step) % n
            frame = load_native(files[idx], self._cur_cam)
            if frame is None:
                continue
            res = analyze(frame, flat)
            if res["status"] != "ok":
                self._index[self._cur_cam] = idx
                self._load_current()
                return
        self._sb.showMessage("このカメラに検出失敗の画像はありません")

    # --------------------------------------------------
    def _recompute_and_render(self):
        if self._cur_frame is None:
            return
        flat = self._panels[self._cur_cam].get_flat()
        res = analyze(self._cur_frame, flat)
        self._cur_res = res
        self._refresh_preview()
        self._update_status(res, flat)

    def _refresh_preview(self):
        if self._cur_frame is None or self._cur_res is None:
            return
        flat = self._panels[self._cur_cam].get_flat()
        mode = self._mode.currentText()
        view = render(self._cur_frame, self._cur_res, flat, mode)
        self._det_preview.setPixmap(to_pixmap(view, PREVIEW_SIZE))

        cropped = apply_crop(self._cur_frame, self._cur_res)
        if cropped is not None:
            self._crop_preview.setPixmap(to_pixmap(cropped, PREVIEW_SIZE))
        else:
            self._crop_preview.clear()
            self._crop_preview.setText("検出できず\nクロップ対象外")
            self._crop_preview.setAlignment(Qt.AlignCenter)

    def _update_status(self, res: dict, flat: dict):
        area = res["area"]
        area_txt = f"{area:,}" if area else "--"
        half = window_half_px(flat["fruit_speed_px"])
        if res["status"] == "ok":
            band = (f"推論窓の内側 ✓（±{half}px）" if res["in_band"]
                    else f"推論窓の外（±{half}px）")
            self._status_lbl.setText(f"面積: {area_txt} ／ 検出: あり ✓ ／ {band}")
            color = "#00aa33" if res["in_band"] else "#cc8800"
            self._status_lbl.setStyleSheet(f"color:{color}; font-size:13px; font-weight:bold;")
            return
        reason = {
            "no_blob":   "候補なし（HSV範囲を広げる）",
            "too_small": "面積不足（最小面積を下げる／マスクを広げる）",
            "edge":      "画面端に接触（対象外）",
        }.get(res["status"], res["status"])
        self._status_lbl.setText(f"面積: {area_txt} ／ 検出: なし ／ {reason}")
        self._status_lbl.setStyleSheet("color:#cc4444; font-size:13px; font-weight:bold;")

    # --------------------------------------------------
    def _run_batch(self):
        src_root = self._src_edit.text()
        dst_root = self._dst_edit.text()
        exc_root = self._exc_edit.text()
        if not os.path.isdir(src_root):
            QMessageBox.warning(self, "エラー", "入力データセットフォルダが見つかりません")
            return

        reply = QMessageBox.question(
            self, "確認",
            "各カメラの保存済みJSON設定を使って一括クロップを実行します。\n"
            "（未保存のスライダー値は反映されません）\n\n"
            f"入力: {src_root}\nクロップ先: {dst_root}\n除外先: {exc_root}\n\n実行しますか？")
        if reply != QMessageBox.Yes:
            return

        flat_by_cam = {cam: flat_from_raw(load_raw_config(cam), cam) for cam, _ in CAMERAS}
        default_flat = flat_from_raw({}, None)

        total = ok = 0
        cls_list = [d for d in sorted(os.listdir(src_root))
                    if os.path.isdir(os.path.join(src_root, d))]
        for ci, cls in enumerate(cls_list):
            cls_dir = os.path.join(src_root, cls)
            dst_dir = os.path.join(dst_root, cls)
            exc_dir = os.path.join(exc_root, cls)
            files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(".jpg")]
            for fi, f in enumerate(files):
                total += 1
                m = CAM_RE.search(f)
                cam_name = f"cam_{m.group(1)}" if m else None
                src_path = os.path.join(cls_dir, f)

                img = imread_ja(src_path)
                if img is None:
                    continue
                native = NATIVE_SIZE.get(cam_name, DEFAULT_NATIVE)
                if img.shape[0] != native or img.shape[1] != native:
                    img = cv2.resize(img, (native, native), interpolation=cv2.INTER_AREA)

                flat = flat_by_cam.get(cam_name, default_flat) if cam_name else default_flat
                res = analyze(img, flat)
                out = apply_crop(img, res)
                if out is None:
                    os.makedirs(exc_dir, exist_ok=True)
                    shutil.copy2(src_path, os.path.join(exc_dir, f))
                    continue

                os.makedirs(dst_dir, exist_ok=True)
                if imwrite_ja(os.path.join(dst_dir, f), out):
                    ok += 1

                if fi % 50 == 0:
                    self._batch_status.setText(
                        f"処理中... [{ci+1}/{len(cls_list)}] {cls} ({ok}/{total})")
                    QApplication.processEvents()

        self._batch_status.setText(f"完了: {ok}/{total} 件クロップ成功")
        QMessageBox.information(self, "完了", f"一括クロップが完了しました。\n成功: {ok}/{total}件")

    def closeEvent(self, event):
        event.accept()


# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CropTunerWindow()
    win.show()
    sys.exit(app.exec())

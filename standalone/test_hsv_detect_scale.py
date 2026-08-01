"""
HSV_DETECT_SCALE（帯外常時HSV監視の縮小率）の座標復元が正しいかを検証する。

合成画像（既知の位置・面積の矩形ブロブ）に対して get_target_info を
scale=1.0（無効・従来どおり）と scale=0.5 の両方で実行し、
返ってくる mx/my/area がネイティブ座標系でおおむね一致することを確認する。
実カメラ映像は不要（座標復元の数式が正しいかだけを確認するテスト）。

実行方法（アプリのルートディレクトリで）:
    python standalone/test_hsv_detect_scale.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import module_yolo as yolo_mod
from module_yolo import ImageProcessor

# json/hsv_config_{cam}.json は必須（無ければ _load_hsv_config が例外を送出する）ため、
# 「_refineなし」経路を検証するには実ファイルを持つ一時カメラ名を用意する必要がある。
NO_REFINE_CAM = "__scale_test_no_refine"


def write_no_refine_config() -> str:
    """虚像除去・果柄除去とも無効な設定を NO_REFINE_CAM 用に書き出し、パスを返す。"""
    cfg = {
        "lower1": [0, 100, 100],   "upper1": [32, 255, 255],
        "lower2": [160, 100, 100], "upper2": [180, 255, 255],
        "reflect_sat_lo": 0, "reflect_sat_hi": 255,
        "reflect_v_lo":   0, "reflect_v_hi":   255,
        "stem_open":      0,
        "min_blob_area":  500,
        "fruit_speed_px": 87,
    }
    path = os.path.join("json", f"hsv_config_{NO_REFINE_CAM}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def make_synthetic_frame(size: int, box, hue: int, sat: int, val: int, bg_val: int = 60) -> np.ndarray:
    """size×size のBGR画像に、box=(x0,y0,x1,y1) の矩形だけ指定HSV色を塗って返す。
    背景は彩度0（無彩色）にして、デフォルト/実運用どちらのHSV閾値でも確実にマスク対象外にする。"""
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :, 2] = bg_val  # 背景: S=0, V=bg_val → 無彩色
    x0, y0, x1, y1 = box
    hsv[y0:y1, x0:x1] = (hue, sat, val)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def run_case(label: str, frame: np.ndarray, cam_name, expect_refine: bool):
    results = {}
    for scale in (1.0, 0.5):
        yolo_mod.HSV_DETECT_SCALE = scale
        target = ImageProcessor.get_target_info(frame, cam_name)
        results[scale] = target
    yolo_mod.HSV_DETECT_SCALE = 1.0  # 後始末（他ケースに影響させない）

    t1, t2 = results[1.0], results[0.5]
    print(f"[{label}] refine={expect_refine}")
    if t1 is None or t2 is None:
        print(f"  NG: 検出失敗 scale=1.0:{t1 is not None} scale=0.5:{t2 is not None}")
        return False

    dmx = abs(t1['mx'] - t2['mx'])
    dmy = abs(t1['my'] - t2['my'])
    darea = abs(t1['area'] - t2['area']) / max(t1['area'], 1)
    print(f"  scale=1.0: mx={t1['mx']} my={t1['my']} area={t1['area']} stat={list(t1['stat'])}")
    print(f"  scale=0.5: mx={t2['mx']} my={t2['my']} area={t2['area']} stat={list(t2['stat'])}")
    print(f"  差分: |Δmx|={dmx} |Δmy|={dmy} 面積相対差={darea:.1%}")

    ok = dmx <= 5 and dmy <= 5 and darea <= 0.15
    print("  OK" if ok else "  NG: 許容誤差(±5px, 面積±15%)を超えている")
    return ok


def main():
    all_ok = True

    # ケース1: 虚像除去・果柄除去とも無効な設定（一時ファイル）→「_refineなし」経路
    no_refine_path = write_no_refine_config()
    try:
        frame1 = make_synthetic_frame(
            size=640, box=(180, 180, 380, 380), hue=15, sat=200, val=180)
        all_ok &= run_case("cfg=no_refine", frame1, cam_name=NO_REFINE_CAM, expect_refine=False)
    finally:
        os.remove(no_refine_path)

    # ケース2: 実運用JSON(cam_inside, 560x560, stem_open=20)を使う「_refineあり」経路
    frame2 = make_synthetic_frame(
        size=560, box=(180, 180, 380, 380), hue=15, sat=200, val=180)
    all_ok &= run_case("cfg=cam_inside(refine)", frame2, cam_name="cam_inside", expect_refine=True)

    print("=== 総合結果:", "OK" if all_ok else "NG", "===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""
HSV_DETECT_SCALE（帯外常時HSV監視の縮小率）を実画像で検証する。

学習データセット（cherry_yolo/model作成用imageset/all_true/<カテゴリ>/*.jpg）は
USE_CROP=False の下で保存された input_img_resized（帯内フレームをYOLO入力サイズ640へ
リサイズしたもの）であり、get_target_info に渡る対象と構図が一致する
（同じ画角・同じ1:1リサイズ、カメラごとの正方形native解像度→640への引き伸ばし）。
ファイル名末尾の cam_top/cam_under/cam_inside/cam_outside からカメラ別の実運用
HSV設定(json/hsv_config_{cam}.json)を引けるため、本番と同じ閾値で比較できる。

scale=1.0（現行・無効）を基準に、候補scaleでの検出一致率・座標誤差・処理時間を集計する。

実行方法（アプリのルートディレクトリで）:
    python standalone/verify_hsv_detect_scale_real.py "<画像フォルダ>" [--scales 0.5 0.7] [--limit-per-class 300]
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import module_yolo_csv4_v3 as yolo_mod
from module_yolo_csv4_v3 import ImageProcessor

CAM_RE = re.compile(r"cam_(top|under|inside|outside)")

# 学習データセットは全カメラ共通で640x640保存(=YOLO入力用にリサイズ済み)だが、
# get_target_info が本番で受け取るのはリサイズ前のネイティブ解像度。
# min_blob_area/stem_open はネイティブpx基準で較正されているため、
# 640x640のまま渡すとcam_inside/cam_outsideで較正値とズレる。
# 保存前の状態に近づけるため、ネイティブ解像度へ縮小してから渡す。
NATIVE_SIZE = {
    "cam_top": 640, "cam_under": 640, "cam_inside": 560, "cam_outside": 500,
}


def imread_ja(path: str):
    """日本語・スペースを含むパスでも読めるよう imdecode 経由で読む。"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def find_images(root: str, limit_per_class: int | None):
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(".jpg")]
        if limit_per_class:
            files = files[:limit_per_class]
        for f in files:
            m = CAM_RE.search(f)
            cam_name = f"cam_{m.group(1)}" if m else None
            yield cls, cam_name, os.path.join(cls_dir, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="all_true 直下のフォルダパス")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5])
    ap.add_argument("--limit-per-class", type=int, default=None)
    args = ap.parse_args()

    scales = [1.0] + [s for s in args.scales if s != 1.0]

    # cam ごと・scale ごとの集計
    stat = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "both": 0, "miss_at_scale": 0, "extra_at_scale": 0, "neither": 0,
        "dmx": [], "dmy": [], "darea": [], "ms": 0.0,
    }))
    unreadable = 0
    no_cam = 0
    miss_examples = defaultdict(list)  # scale -> [path,...]（1.0で検出したがscaleで検出漏れ）

    t_start = time.perf_counter()
    n_total = 0
    for cls, cam_name, path in find_images(args.root, args.limit_per_class):
        img = imread_ja(path)
        if img is None:
            unreadable += 1
            continue
        if cam_name is None:
            no_cam += 1
            continue
        n_total += 1

        native = NATIVE_SIZE[cam_name]
        if img.shape[0] != native or img.shape[1] != native:
            img = cv2.resize(img, (native, native), interpolation=cv2.INTER_AREA)

        results = {}
        for scale in scales:
            yolo_mod.HSV_DETECT_SCALE = scale
            t0 = time.perf_counter()
            results[scale] = ImageProcessor.get_target_info(img, cam_name)
            stat[cam_name][scale]["ms"] += (time.perf_counter() - t0) * 1000.0
            stat[cam_name][scale]["n"] += 1
        yolo_mod.HSV_DETECT_SCALE = 1.0

        base = results[1.0]
        for scale in scales:
            if scale == 1.0:
                continue
            cur = results[scale]
            s = stat[cam_name][scale]
            if base is not None and cur is not None:
                s["both"] += 1
                s["dmx"].append(abs(base["mx"] - cur["mx"]))
                s["dmy"].append(abs(base["my"] - cur["my"]))
                a0 = max(base["area"], 1)
                s["darea"].append(abs(base["area"] - cur["area"]) / a0)
            elif base is not None and cur is None:
                s["miss_at_scale"] += 1
                if len(miss_examples[scale]) < 20:
                    miss_examples[scale].append((cls, path))
            elif base is None and cur is not None:
                s["extra_at_scale"] += 1
            else:
                s["neither"] += 1

    elapsed = time.perf_counter() - t_start
    print(f"総画像数: {n_total}（読込失敗: {unreadable}, カメラ名不明でスキップ: {no_cam}）")
    print(f"総所要時間: {elapsed:.1f}s\n")

    for cam_name in sorted(stat.keys()):
        print(f"=== {cam_name} ===")
        base_ms = stat[cam_name][1.0]["ms"] / max(stat[cam_name][1.0]["n"], 1)
        print(f"  scale=1.0 平均処理時間: {base_ms:.3f} ms/枚 (n={stat[cam_name][1.0]['n']})")
        for scale in scales:
            if scale == 1.0:
                continue
            s = stat[cam_name][scale]
            n = s["n"]
            cur_ms = s["ms"] / max(n, 1)
            speedup = base_ms / cur_ms if cur_ms > 0 else float("inf")
            agree = s["both"] + s["neither"]
            print(f"  --- scale={scale} ---")
            print(f"    平均処理時間: {cur_ms:.3f} ms/枚 ({speedup:.2f}倍高速化)")
            print(f"    go/no-go一致: {agree}/{n} ({100*agree/n:.1f}%)"
                  f"  [両方検出:{s['both']} 両方非検出:{s['neither']}"
                  f" scale側で検出漏れ:{s['miss_at_scale']} scale側で余分検出:{s['extra_at_scale']}]")
            if s["dmx"]:
                dmx = np.array(s["dmx"]); dmy = np.array(s["dmy"]); darea = np.array(s["darea"])
                print(f"    座標誤差(両方検出時, n={len(dmx)}): "
                      f"|Δmx| 平均{dmx.mean():.1f} p95={np.percentile(dmx,95):.1f} 最大{dmx.max()}px / "
                      f"|Δmy| 平均{dmy.mean():.1f} p95={np.percentile(dmy,95):.1f} 最大{dmy.max()}px / "
                      f"面積相対誤差 平均{darea.mean():.1%} p95={np.percentile(darea,95):.1%}")
        print()

    for scale, examples in miss_examples.items():
        if examples:
            print(f"--- scale={scale} で検出漏れした例（最大20件） ---")
            for cls, path in examples:
                print(f"  [{cls}] {path}")


if __name__ == "__main__":
    main()

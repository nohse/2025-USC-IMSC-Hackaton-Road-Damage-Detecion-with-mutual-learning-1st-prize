# yolov10_eval_with_remap.py

import os
import glob
import yaml
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

# ── 여러분의 4클래스 정의 ────────────────────────────────
#   0->D40, 1->D20, 2->D10, 3->D00
user_names = {0: "D40", 1: "D20", 2: "D10", 3: "D00"}

# 원래 모델(5클래스) → 여러분(4클래스) 인덱스 매핑
#   0:Repair(버림), 1:D00→3, 2:D10→2, 3:D20→1, 4:D40→0
orig_to_user = {1: 3, 2: 2, 3: 1, 4: 0}


def remap_preds(boxes):
    """
    boxes: res.boxes (ultralytics Boxes object)
    returns: np.ndarray x1,y1,x2,y2 / np.ndarray mapped_cls / np.ndarray conf
    """
    mapped = []
    for xy, c, conf in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy()
    ):
        if c == 0:
            continue
        new_c = orig_to_user[c]
        mapped.append([*xy, new_c, conf])
    if not mapped:
        return np.zeros((0,4)), np.zeros((0,),dtype=int), np.zeros((0,))
    arr = np.array(mapped)
    return arr[:, :4], arr[:, 4].astype(int), arr[:, 5]


if __name__ == "__main__":
    # 1) 모델 가중치 경로 지정
    WEIGHTS = "/home/sejong/seonghyeon/road_damage_detection/FRDC-RDD/FRDC-RDD/yolov10_x_best.pt"
    print(f"▶︎ Using weights: {WEIGHTS}")

    # 2) data.yaml 로드
    DATA_YAML = Path("/home/sejong/seonghyeon/road_damage_detection/yolodata/finetune_data.yaml")
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    print(f"▶︎ Loaded dataset config: {DATA_YAML}")

    # 클래스 이름(ground-truth용)
    gt_names = data_cfg["names"]  # e.g. ["longitudinal crack", ...]
    nc_gt = len(gt_names)
    print(f"▶︎ GT classes ({nc_gt}): {gt_names}")

    # 3) 모델 로드 및 val 평가
    model = YOLO(WEIGHTS)
    print("▶︎ Running model.val() …")
    results = model.val(
        data=str(DATA_YAML),
        batch=64,
        imgsz=600,
        device=0,
        split="val",
        save=False
    )

    # 4) Overall / Box 지표 출력
    print("\n===== Overall Metrics =====")
    for m in results.summary():
        print(m)
    box = results.box
    print("\n===== Box Metrics =====")
    print(f" mp   (mean precision) : {box.mp:.4f}")
    print(f" mr   (mean recall)    : {box.mr:.4f}")
    print(f" mAP50               : {box.map50:.4f}")
    print(f" mAP50-95            : {box.map:.4f}")
    print(f" mAP75               : {box.map75:.4f}")

    print("\n===== Class-wise AP@0.5 =====")
    for i, ap50 in enumerate(box.ap50):
        name = gt_names[i] if i < nc_gt else f"Class {i}"
        print(f"  {i}: {name} → {ap50:.4f}")

    # 5) Confusion Matrix (remap 후)
    confmat = ConfusionMatrix(names=list(user_names.values()), task="detect")
    data_root  = DATA_YAML.parent
    im_dir     = data_root / "images" / "val"
    lbl_dir    = data_root / "labels" / "val"
    img_list   = sorted(im_dir.glob("*.jpg"))
    print(f"\n▶︎ Found {len(img_list)} validation images")

    for img_path in img_list:
        res = model(str(img_path))
        # remap preds
        xyxy, pred_cls, confs = remap_preds(res.boxes)
        # load GT
        gt = []
        with open(lbl_dir / f"{img_path.stem}.txt") as f:
            for line in f:
                gt_cls = int(line.split()[0])
                gt.append(gt_cls)
        gt = np.array(gt, dtype=int)

        confmat.process_batch(xyxy, confs, pred_cls, gt)

    print("\n===== Confusion Matrix (rows=GT, cols=Pred) =====")
    print(confmat.matrix)

    print("\n✅ Evaluation with remapping complete!")

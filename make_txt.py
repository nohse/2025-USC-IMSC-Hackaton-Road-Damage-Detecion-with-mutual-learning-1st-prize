#!/usr/bin/env python3
import os
from ultralytics import YOLO

# —— 설정 부분만 바꿔주세요 ——
WEIGHTS    = "/home/sejong/seonghyeon/road_damage_detection/runs/detect/train/yolo8_1/best.pt"
IMG_DIR     = "/home/sejong/seonghyeon/road_damage_detection/hack_data/images/val_all"
OUTPUT_DIR  = os.path.join(IMG_DIR, "predictions_txt")
CONF_THRESH = 0.25# 신뢰도 임계값
IOU_THRESH  = 0.80  # NMS IOU 임계값
IMG_SIZE    = 768    # 입력 이미지 사이즈
DEVICE      = "cuda:0"  # GPU 사용
# ——————————————————————

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모델 로드
model = YOLO(WEIGHTS)

# 이미지 폴더 순회
for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    # 추론 (imgsz, device 옵션 추가)
    results = model(
        img_path,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMG_SIZE,
        device=DEVICE
    )

    # 결과를 쓸 txt 파일 경로
    base, _ = os.path.splitext(fname)
    txt_path = os.path.join(OUTPUT_DIR, base + ".txt")

    # YOLO 형식: class x1 y1 x2 y2 confidence
    with open(txt_path, 'w') as f:
        boxes = results[0].boxes
        for xyxy, cls, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy()
            ):
            x1, y1, x2, y2 = xyxy
            class_id = int(cls)
            score    = float(conf)
            f.write(f"{class_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.4f}\n")

    print(f"[+] Saved: {txt_path}")

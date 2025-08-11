import os
import glob
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data import YOLODataset
from ultralytics.utils.metrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix

# 검증용 클래스 리스트
CLASSES = [
    "longitudinal crack",
    "alligator crack",
    "transverse crack",
    "pothole"
]

if __name__ == "__main__":
    # 1) 가장 최근 학습 실험의 best.pt 찾기
    exp_dirs = sorted(glob.glob("runs/detect/train/*"), key=os.path.getmtime)
    last_exp = exp_dirs[-1]
    best_weights = os.path.join(
        last_exp,
        "/home/sejong/seonghyeon/road_damage_detection/FRDC-RDD/FRDC-RDD/yolov10_x_best.pt".
    )
    print(f"▶︎ 평가에 사용할 가중치: {best_weights}")

    # 2) 데이터 YAML 경로
    DATA_YAML = Path("/home/sejong/seonghyeon/road_damage_detection/yolodata/finetune_data.yaml")
    print(f"▶︎ 데이터셋 구성 파일: {DATA_YAML}")

    # 3) 모델 로드
    model = YOLO(best_weights)

    # 4) validation만 수행
    print("▶︎ 검증 시작…")
    results = model.val(
        data=str(DATA_YAML),
        batch=64,
        imgsz=600,
        device=0,
        split="val",
        save=False
    )

    # 5) 전체 요약 지표 출력
    print("\n===== Overall Metrics =====")
    for metric in results.summary():
        print(metric)

    # 6) 박스 기반 지표 상세
    box = results.box
    print("\n===== Box Metrics =====")
    print(f"Mean Precision: {box.mp:.4f}")
    print(f"Mean Recall:    {box.mr:.4f}")
    print(f"mAP50:          {box.map50:.4f}")
    print(f"mAP50-95:       {box.map:.4f}")
    print(f"mAP75:          {box.map75:.4f}")

    # 7) 클래스별 AP@0.5 출력
    print("\n===== Class-wise AP@0.5 =====")
    for idx, ap50 in enumerate(box.ap50):
        name = CLASSES[idx] if idx < len(CLASSES) else f"Class {idx}"
        print(f"  {idx}: {name} → {ap50:.4f}")

    # 8) Confusion Matrix 계산 및 출력 (Ultralytics ConfusionMatrix 사용)
    print("\n===== Computing Confusion Matrix with Ultralytics ConfusionMatrix =====")
    # (a) ConfusionMatrix 초기화: names=클래스 리스트, task='detect'
    confmat = ConfusionMatrix(names=CLASSES, task='detect')

    # (b) val 이미지/라벨 디렉터리 경로 계산 (YAML 경로 기준)
    data_root   = DATA_YAML.parent                      # yolodata 루트
    images_dir  = data_root / "images" / "val"
    labels_dir  = data_root / "labels" / "val"
    val_images  = list(images_dir.glob("*.jpg"))
    print(f"▶︎ Found {len(val_images)} images in {images_dir}")

    # (c) 이미지 단위로 inference → ConfusionMatrix에 전달
    for img_path in val_images:
        res    = model(str(img_path))
        boxes  = res.boxes
        confmat.process_batch(
            boxes.xyxy.cpu().numpy(),                 # [M×4] 박스 좌표
            boxes.conf.cpu().numpy(),                 # [M] confidence
            boxes.cls.cpu().numpy().astype(int),      # [M] 예측 클래스
            np.array([int(line.split()[0]) for line in open(
                labels_dir / (img_path.stem + ".txt"))], dtype=int)
        )

    # (d) 최종 혼돈행렬
    cm = confmat.matrix  # shape (nc, nc)
    print("Rows = GT, Cols = Pred")
    print(cm)

    print("\n✅ 검증 지표 및 Confusion Matrix 출력 완료!")

import os
from ultralytics import YOLO

if __name__ == "__main__":

    print("▶︎ YOLOv8m COCO 사전학습 모델 로드 및 훈련 시작...")
    model = YOLO("yolov8m.pt")

    DATA_YAML = "/home/sejong/seonghyeon/road_damage_detection/yolodata/finetune_data.yaml"
    model.train(
        data=str(DATA_YAML),
        epochs=300,
        imgsz=640,
        batch=32,
        device=0,
        save_period=1,
        exist_ok=True,
        val=True,
        name="pretrain",
        half=True

        # # ——— 여기부터 Augmentation 옵션 ———
        # augment    = True,    # 기본 Augmentation on/off
        # mosaic     = 1.0,     # mosaic 비율 (0~1)
        # mixup      = 0.3,     # mixup 비율 (0~1)
        # hsv_h      = 0.5,   # 색조 변동 범위
        # hsv_s      = 0.5,     # 채도 변동 범위
        # hsv_v      = 0.5,     # 밝기 변동 범위
        # translate  = 0.1,     # 평행이동 비율
        # scale      = 0.5,     # 스케일 변동 비율
        # shear      = 2.0,     # 전단(shear) 각도
        # perspective= 0.0,     # 원근 변형 강도
        # flipud     = 0,     # 상하 뒤집기 확률
        # fliplr     = 0.5,     # 좌우 뒤집기 확률
        # # ————————————————————————
    )

    print("✅ 훈련 완료. runs/train/ 하위 폴더에서 best.pt & last.pt 확인하세요.")

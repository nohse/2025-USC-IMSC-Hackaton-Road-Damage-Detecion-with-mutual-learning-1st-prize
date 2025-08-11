from ultralytics import YOLO
import argparse
import os
import torch
import csv
import urllib.request

def download_if_missing(model_name):
    model_urls = {
        "yolov10s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt",
        "yolov10m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt",
        "yolov10l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10l.pt",
    }

    if model_name not in model_urls:
        print(f"⚠️ {model_name} is not a recognized YOLOv10 model, skipping download.")
        return

    if not os.path.exists(model_name):
        print(f"🔄 Downloading {model_name} ...")
        urllib.request.urlretrieve(model_urls[model_name], model_name)
        print(f"✅ {model_name} downloaded.")
    else:
        print(f"✅ {model_name} already exists.")

def export_tensor(pt_path):
    model = YOLO(pt_path)
    model.export(
        imgsz=224,
        format="engine",
        dynamic=False,
        batch=8,
        workspace=10,
        half=True,
        int8=False,
        data="train_full_pseudo.yaml",
        device=[0],
        opset=12,
    )

# Setup
torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser(description='YOLO Inference Script')
parser.add_argument('--model-file', type=str, help='model file name including directory name')
parser.add_argument('--source-path', type=str, help='Path to the directory containing images for inference')
parser.add_argument('--output-file', type=str, help='output CSV file name including directory name')
parser.add_argument('--engine', action='store_true', help='export tensorrt engine')
args = parser.parse_args()

# Load the YOLO model
model_path = "/home/sejong/seonghyeon/road_damage_detection/runs/detect/pretrain/weights/epoch75.pt"#이가은

# ✅ 자동 다운로드
download_if_missing(os.path.basename(model_path))

if args.engine:
    engine_path = model_path.replace('.pt', '.engine')
    if not os.path.exists(engine_path):
        print('🚀 Exporting to TensorRT engine...')
        export_tensor(model_path)
    print(f"[!] TensorRT export done, but note: Ultralytics 'YOLO()' cannot load '.engine' directly.")
    # model_path = engine_path  # 주석 유지: .engine 로드 불가능

# 모델 로딩
net = YOLO(model_path)

# 이미지 추론 경로
source_path = "/home/sejong/seonghyeon/road_damage_detection/yolodata/test/images"#이가은

# Run inference
results = net.predict(
    source=source_path,
    device=0,
    conf=0.30,#이가은
    max_det=20,
    augment=False,
    imgsz=640,
    classes=[0,1,2,3],
    batch=4,
    half=True
)

import os
import csv

# 결과 CSV 저장 경로
csv_file_path = "/home/sejong/seonghyeon/road_damage_detection/output/charmander_yolo8mmutual-tunedcort-0.30.csv"#이가은

# 1) 출력 디렉토리 생성 (없으면)
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# 2) 파일 존재 여부로 모드 결정
file_exists = os.path.isfile(csv_file_path)
mode = 'a' if file_exists else 'w'

with open(csv_file_path, mode=mode, newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # 3) 새로 만들 때만 헤더 작성
    if not file_exists:
        csv_writer.writerow(['ImageId', 'PredictionString'])
    
    # 4) 예측 결과를 파일에 기록
    for result in results:
        image_path = result.path
        image_name = os.path.basename(image_path)

        prediction_string = ""
        labels = result.boxes.cls.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for label, box in zip(labels, boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            prediction_string += f"{int(label)} {x_min} {y_min} {x_max} {y_max} "

        csv_writer.writerow([image_name, prediction_string.strip()])

print(f"✅ Predictions saved to: {csv_file_path}")
import os

# 5) baseline 폴더 생성
csv_dir = os.path.dirname(csv_file_path)
txt_dir = os.path.join(csv_dir, 'baseline')
os.makedirs(txt_dir, exist_ok=True)

# 6) results 순회하며 txt 생성 (한 줄에 하나의 detection)
for result in results:
    image_name = os.path.basename(result.path)
    image_id   = os.path.splitext(image_name)[0]

    # country별 이미지 크기 설정
    if image_name.startswith('country1_'):
        size = 720
    elif image_name.startswith('country2_'):
        size = 600
    elif image_name.startswith('country3_'):
        size = 640
    else:
        size = max(result.orig_shape[1], result.orig_shape[0])

    labels = result.boxes.cls.cpu().numpy()
    boxes  = result.boxes.xyxy.cpu().numpy()
    confs  = result.boxes.conf.cpu().numpy()

    txt_lines = []
    for lbl, box, conf in zip(labels, boxes, confs):
        x_min, y_min, x_max, y_max = box.astype(float)
        x_c = ((x_min + x_max) / 2) / size
        y_c = ((y_min + y_max) / 2) / size
        w   = (x_max - x_min) / size
        h   = (y_max - y_min) / size
        txt_lines.append(f"{int(lbl)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}")


    # 이제 각 detection을 줄바꿈으로 구분해 파일에 씁니다.
    txt_path = os.path.join(txt_dir, f"{image_id}.txt")
    with open(txt_path, 'w') as f:
        f.write("\n".join(txt_lines))

print(f"✅ Text files saved to: {txt_dir}")


from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import torch

# Parse arguments
parser = argparse.ArgumentParser(description='YOLO Inference Script')
parser.add_argument(
    'model_file',
    type=str,
    nargs='?',
    default='/home/sejong/seonghyeon/road_damage_detection/FRDC-RDD/FRDC-RDD/yolov10_x_best.pt',
    help='model file name including directory name'
)

parser.add_argument(
    'source_path',
    type=str,
    nargs='?',
    default='/home/sejong/seonghyeon/road_damage_detection/hack_data/images/val',
    help='Path to the directory containing images for inference'
)
parser.add_argument(
    'output_file',
    type=str,
    nargs='?',
    default='/home/sejong/seonghyeon/road_damage_detection/hack_data/images/predict',
    help='output file name including directory name'
)
args = parser.parse_args()
WEIGHTS    = "/home/sejong/seonghyeon/road_damage_detection/FRDC-RDD/FRDC-RDD/yolov10_x_best.pt"
print(args.model_file)
torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# Load the exported TensorRT model
model = YOLO(WEIGHTS, task="detect")
countries = ['country1', 'country2', 'country3']
task_path = Path(args.output_file).parent
task_num_path = str(task_path / f'countries')

os.makedirs(task_num_path, exist_ok=True)
with open(args.output_file, 'w') as fw:
    for country in countries:
        if country == 'Norway':
            results = model(args.source_path + f'/{country}/test/crop/images', conf=0.1, device=DEVICE, max_det=20,
                            augment=True, plots=True, batch=8,
                            imgsz=640, classes=[1, 2, 3, 4])
            with open(task_num_path + f'/{country}_results.txt', 'w') as f:
                for r in results:
                    txt = f'{Path(r.path).stem}.jpg,'
                    init_len = len(txt)
                    for idx, (cls, xyxy) in enumerate(
                            zip(r.boxes.cls, r.boxes.xyxy)):
                        xyxy = xyxy.cpu().numpy()
                        xyxy[3] += 1000
                        xyxy[1] += 1000
                        if idx == 0:
                            txt += f'{cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                        else:
                            txt += f' {cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                    f.write(txt + '\n')
                    fw.write(txt + '\n')
        else:
            results = model(args.source_path + f'/{country}', conf=0.1, device=DEVICE, max_det=20,
                            augment=True, plots=True, batch=8,
                            imgsz=640, classes=[1, 2, 3, 4])
            with open(task_num_path + f'/{country}_results.txt', 'w') as f:
                for r in results:
                    txt = f'{Path(r.path).stem}.jpg,'
                    init_len = len(txt)
                    for idx, (cls, xyxy) in enumerate(
                            zip(r.boxes.cls, r.boxes.xyxy)):
                        xyxy = xyxy.cpu().numpy()
                        if idx == 0:
                            txt += f'{cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                        else:
                            txt += f' {cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                    f.write(txt + '\n')
                    fw.write(txt + '\n')

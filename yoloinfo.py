# inspect_frdc_rdd.py

from ultralytics import YOLO

# 1) モデル読み込み
model_path = "/home/sejong/seonghyeon/road_damage_detection/FRDC-RDD/FRDC-RDD/yolov10_x_best.pt"
model = YOLO(model_path)

# 2) クラス数取得
#    YOLOv10/YOLOv8 系モデルは model.model.yaml に metadata が入っています
nc = model.model.yaml.get('nc', None)
names = model.model.yaml.get('names', None)

if nc is None or names is None:
    # fallback: ultralytics YOLO オブジェクトの .names 属性を見る
    names = getattr(model, 'names', None)
    nc = len(names) if isinstance(names, dict) else None

# 3) 結果を表示
print(f"Number of classes: {nc}")
print("Class index → Damage type mapping:")
for idx, label in names.items():
    print(f"  {idx:2d} → {label}")

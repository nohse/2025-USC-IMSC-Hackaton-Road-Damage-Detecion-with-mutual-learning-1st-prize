#!/usr/bin/env python3
import os
import glob
import json
import argparse
from PIL import Image

def yolo_to_coco(images_dir, labels_dir, output_json):
    # — 카테고리 매핑 (YOLO 클래스 0~3 → COCO id 1~4)
    category_map = {
        0: "longitudinal crack",
        1: "alligator crack",
        2: "transverse crack",
        3: "pothole",
    }
    categories = [
        {"id": cid+1, "name": name}
        for cid, name in category_map.items()
    ]

    images      = []
    annotations = []
    ann_id = 1
    img_id = 1

    # 모든 이미지 파일 순회 (재귀)
    for img_path in sorted(glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True)):
        # 1) 이미지 메타
        w, h = Image.open(img_path).size
        rel_path = os.path.relpath(img_path, images_dir)
        images.append({
            "id": img_id,
            "file_name": rel_path.replace("\\", "/"),
            "width": w,
            "height": h
        })

        # 2) 대응하는 YOLO txt 라벨 경로
        label_path = os.path.join(labels_dir, os.path.splitext(rel_path)[0] + ".txt")
        if os.path.isfile(label_path):
            # 3) 한 이미지당 모든 박스
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts)
                    # normalized → absolute
                    cx_abs = cx * w
                    cy_abs = cy * h
                    bw_abs = bw * w
                    bh_abs = bh * h
                    x1 = cx_abs - bw_abs / 2
                    y1 = cy_abs - bh_abs / 2

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls) + 1,
                        "bbox": [x1, y1, bw_abs, bh_abs],
                        "area": bw_abs * bh_abs,
                        "iscrowd": 0
                    })
                    ann_id += 1

        img_id += 1

    # 4) JSON으로 기록
    coco = {
        "images":      images,
        "annotations": annotations,
        "categories":  categories
    }
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved COCO annotations: {output_json}")
    print(f"   • images: {len(images)}")
    print(f"   • annotations: {len(annotations)}")
    print(f"   • categories: {len(categories)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YOLO txt labels to COCO JSON"
    )
    parser.add_argument(
        "--images_dir", "-i", required=True,
        help="root of train images (e.g. hack_data/images/train)"
    )
    parser.add_argument(
        "--labels_dir", "-l", required=True,
        help="root of YOLO txt labels (e.g. hack_data/labels/train_all)"
    )
    parser.add_argument(
        "--output_json", "-o", required=True,
        help="where to write the COCO JSON (e.g. hack_data/annotations/train.json)"
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    yolo_to_coco(args.images_dir, args.labels_dir, args.output_json)


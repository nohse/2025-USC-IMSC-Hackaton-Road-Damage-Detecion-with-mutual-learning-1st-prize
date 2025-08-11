#!/usr/bin/env python3
import os
import glob
import argparse
from PIL import Image

import torch
from torchvision.ops import nms

def process_file(in_path, out_path, img_dir, score_thresh, iou_thresh):
    # 1) 원본 dense output 파싱
    detections = []
    with open(in_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls, x1, y1, x2, y2, score = parts
            score = float(score)
            if score < score_thresh:
                continue
            detections.append([
                int(cls),
                float(x1), float(y1), float(x2), float(y2),
                score
            ])

    if not detections:
        open(out_path, 'w').close()
        return

    # 2) NMS 적용
    boxes  = torch.tensor([d[1:5] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d[5]    for d in detections], dtype=torch.float32)
    classes= [d[0] for d in detections]
    keep   = nms(boxes, scores, iou_thresh)

    # 3) 이미지 크기 읽기
    img_name = os.path.basename(in_path)[:-4] + '.jpg'
    img_path = os.path.join(img_dir, img_name)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_w, img_h = Image.open(img_path).size

    # 4) YOLO normalized 포맷으로 저장
    with open(out_path, 'w') as f:
        for idx in keep:
            cls = classes[idx]
            x1, y1, x2, y2 = boxes[idx].tolist()
            # center / width-height
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def main():
    parser = argparse.ArgumentParser(
        description="폴더 내 txt들을 score/NMS 필터링 후 YOLO normalized 포맷으로 저장"
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="원본 dense-output txt들이 있는 폴더"
    )
    parser.add_argument(
        "--img_dir", "-m", required=True,
        help="txt와 짝을 이루는 원본 이미지(.jpg)들이 있는 폴더"
    )
    parser.add_argument(
        "--output_dir", "-o", default="pseudo_labels",
        help="처리된 pseudo-label txt를 저장할 폴더 (없으면 생성)"
    )
    parser.add_argument(
        "--score_thresh", "-s", type=float, default=0.5,
        help="스코어 임계값 (default: 0.5)"
    )
    parser.add_argument(
        "--iou_thresh", "-t", type=float, default=0.5,
        help="NMS IoU 임계값 (default: 0.5)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    if not txt_files:
        print("입력 폴더에 .txt 파일이 없습니다.")
        return

    for in_path in txt_files:
        fname    = os.path.basename(in_path)
        out_path = os.path.join(args.output_dir, fname)
        process_file(
            in_path, out_path,
            img_dir     = args.img_dir,
            score_thresh= args.score_thresh,
            iou_thresh  = args.iou_thresh
        )
        print(f"[OK] {fname} → {out_path}")

if __name__ == "__main__":
    main()

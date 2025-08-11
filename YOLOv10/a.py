#!/usr/bin/env python3
import os

# ———— 설정 —————————————————————————————
# ① 원본 예측이 담긴 파일 (한 줄에 이미지 파일명,"," 그리고 박스 정보)
INPUT_FILE = "/home/sejong/seonghyeon/road_damage_detection/hack_data/images/countries/country3_results.txt"
# ② 분리된 txt 를 저장할 폴더
OUTPUT_DIR = "/home/sejong/seonghyeon/road_damage_detection/hack_data/images/predicts"
# ③ 이미지 가로·세로 크기 (YOLO 정규화에 사용)
IMG_SIZE = 640.0
# —————————————————————————————————————————

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, 'r') as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        # ex) "country1_000054.jpg,4 250 0 277 239 1 2 5 7 7"
        img_name, anno_str = line.split(',', 1)
        anno_str = anno_str.strip()
        base, _ = os.path.splitext(img_name)
        out_path = os.path.join(OUTPUT_DIR, base + ".txt")

        with open(out_path, 'w') as fout:
            if anno_str:
                nums = anno_str.split()
                # 5개씩(class, x1, y1, x2, y2) 묶어서 정규화
                for i in range(0, len(nums), 5):
                    chunk = nums[i:i+5]
                    if len(chunk) != 5:
                        continue
                    cls, x1, y1, x2, y2 = map(float, chunk)
                    # 픽셀 좌표 → YOLO 정규화
                    xc = (x1 + x2) / 2.0 / IMG_SIZE
                    yc = (y1 + y2) / 2.0 / IMG_SIZE
                    w  = (x2 - x1)       / IMG_SIZE
                    h  = (y2 - y1)       / IMG_SIZE
                    fout.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        print(f"[+] Wrote {out_path}")

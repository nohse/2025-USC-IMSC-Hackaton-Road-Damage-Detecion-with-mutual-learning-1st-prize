#!/usr/bin/env python3
import os
import json

# 1) 클래스 이름→인덱스 매핑
CLASS_MAP = {
    "pothole": 0,
    "alligator crack": 1,
    "transverse crack": 2,
    "longitudinal crack": 3
}

# 원본 JSON이 있는 디렉토리
src_dir = "/home/sejong/seonghyeon/country3/ann"
# 변환된 TXT를 저장할 디렉토리
dst_dir = "/home/sejong/seonghyeon/country3/label/train"

# 저장 디렉토리 생성 (없으면)
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".jpg.json"):
        continue

    fullpath = os.path.join(src_dir, fname)
    with open(fullpath, 'r') as f:
        j = json.load(f)

    # JSON의 size 필드에서 width, height 가져오기
    img_w = j.get("size", {}).get("width")
    img_h = j.get("size", {}).get("height")
    if not img_w or not img_h:
        print(f"⚠️ size 정보 누락, 스킵합니다: {fname}")
        continue

    lines = []
    for obj in j.get("objects", []):
        cls_name = obj.get("classTitle", "").lower()
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]

        pts = obj["points"]["exterior"]
        if len(pts) < 2:
            continue
        x1, y1 = pts[0]
        x2, y2 = pts[1]

        # YOLO 포맷 변환 & 정규화
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w  = abs(x2 - x1) / img_w
        h  = abs(y2 - y1) / img_h

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # 출력 파일명: 예) India_000000.jpg.json → India_000000.txt
    out_name = fname.replace(".jpg.json", ".txt")
    out_path = os.path.join(dst_dir, out_name)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"✅ {out_name} 생성 ({len(lines)}개체) in {dst_dir}")

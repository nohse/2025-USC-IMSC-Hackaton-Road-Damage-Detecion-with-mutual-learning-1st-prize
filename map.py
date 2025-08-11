import os
import glob

# —— 설정 ——
INPUT_DIR = "/home/sejong/seonghyeon/road_damage_detection/hack_data/images/val_all/predictions_txt"
# ————————

# 매핑 딕셔너리
mapping = {
    1: 3,
    2: 2,
    3: 1,
    4: 0
}

# 폴더 내 모든 txt 순회
for txt_path in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
    new_lines = []
    with open(txt_path, 'r') as fin:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            # 0 삭제, 매핑에 없는 클래스도 삭제
            if cls == 0 or cls not in mapping:
                continue
            # remap
            new_cls = mapping[cls]
            # 나머지 좌표·신뢰도
            rest = parts[1:]
            new_lines.append(" ".join([str(new_cls)] + rest))

    # 덮어쓰기
    with open(txt_path, 'w') as fout:
        fout.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    print(f"[+] Remapped: {os.path.basename(txt_path)}")
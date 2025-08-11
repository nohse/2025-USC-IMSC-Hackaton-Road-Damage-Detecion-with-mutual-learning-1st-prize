import os
import shutil
from glob import glob

# 원본/대상 디렉토리 경로 설정
src_dir = '/home/sejong/seonghyeon/road_damage_detection/yolodata/pretrain_data/labels/val'
dst_dir = '/home/sejong/seonghyeon/road_damage_detection/yolodata/pretrain_data/labels/refine_va;'
print("****")
# 대상 디렉토리 없으면 생성
os.makedirs(dst_dir, exist_ok=True)

# 복사할 파일 패턴 목록
patterns = ['India*.txt', 'Japn*.txt', 'United*.txt']

# 패턴별로 파일 검색 후 복사
for pattern in patterns:
    for src_path in glob(os.path.join(src_dir, pattern)):
        shutil.copy(src_path, dst_dir)
        print(f'Copied: {os.path.basename(src_path)}')

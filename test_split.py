import os
import random
import shutil
from pathlib import Path

# 설정
BASE_DIR = Path('/home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data')
TEST_DIR = BASE_DIR / 'test'    # test set 이 생성될 디렉토리
NUM_SAMPLES = 400               # 각 도시별로 뽑을 샘플 수

def make_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def split_city(city_dir: Path):
    imgs_dir   = city_dir / 'images'
    labs_dir   = city_dir / 'labels'
    city_name  = city_dir.name

    dest_img_dir = TEST_DIR / city_name / 'images'
    dest_lab_dir = TEST_DIR / city_name / 'labels'
    make_dirs(dest_img_dir)
    make_dirs(dest_lab_dir)

    img_paths = sorted(imgs_dir.glob('*.*'))
    if len(img_paths) < NUM_SAMPLES:
        raise ValueError(f"{city_name}의 이미지 수({len(img_paths)})가 {NUM_SAMPLES}개보다 적습니다.")

    sampled = random.sample(img_paths, NUM_SAMPLES)

    for img_path in sampled:
        lab_path = labs_dir / f"{img_path.stem}.txt"
        if not lab_path.exists():
            raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {lab_path}")

        # 복사가 아닌 이동
        shutil.move(str(img_path),  str(dest_img_dir / img_path.name))
        shutil.move(str(lab_path),  str(dest_lab_dir / lab_path.name))

    print(f"{city_name}: {NUM_SAMPLES}개 샘플을 test/{city_name}로 이동했습니다.")

def main():
    random.seed(42)
    make_dirs(TEST_DIR)

    for child in BASE_DIR.iterdir():
        if child.is_dir() and child.name.startswith('country'):
            split_city(child)

if __name__ == '__main__':
    main()

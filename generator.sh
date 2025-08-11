python pseudo_label_generator.py \
  -i /home/sejong/seonghyeon/road_damage_detection/infer/codetr_pseudo_best_country1 \
  -o /home/sejong/seonghyeon/road_damage_detection/infer/pseudo/0.2/country1_c \
  -s 0.2 -t 0.2\
  -m /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/all


python pseudo_label_generator.py \
  -i /home/sejong/seonghyeon/road_damage_detection/infer/codetr_pseudo_best_country2 \
  -o /home/sejong/seonghyeon/road_damage_detection/infer/pseudo/0.2/country2_c \
  -s 0.2 -t 0.2\
  -m /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/all


python pseudo_label_generator.py \
  -i /home/sejong/seonghyeon/road_damage_detection/infer/codetr_pseudo_best_country3 \
  -o /home/sejong/seonghyeon/road_damage_detection/infer/pseudo/0.2/country3_c \
  -s 0.2 -t 0.2

python pseudo_label_generator.py \
  -i /home/sejong/seonghyeon/road_damage_detection/infer/rtmdet_pseudo_best_country1 \
  -o /home/sejong/seonghyeon/road_damage_detection/infer/pseudo/0.2/country1_r \
  -s 0.2 -t 0.2\
  -m /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/all


python pseudo_label_generator.py \
  -i /home/sejong/seonghyeon/road_damage_detection/infer/rtmdet_pseudo_best_country2 \
  -o /home/sejong/seonghyeon/road_damage_detection/infer/pseudo/0.2/country2_r \
  -s 0.2 -t 0.2\
  -m /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/all

python pseudo_label_generator.py \
  -i /home/sejong/seonghyeon/road_damage_detection/infer/rtmdet_pseudo_best_country3 \
  -o /home/sejong/seonghyeon/road_damage_detection/infer/pseudo/0.2/country3_r \
  -s 0.2 -t 0.2\
  -m /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/all


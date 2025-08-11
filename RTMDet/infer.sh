#python infer.py \
#    --config work_dirs/rtmdet_x_syncbn_fast_8xb32-300e_coco/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
#    --ckpt work_dirs/rtmdet_x_syncbn_fast_8xb32-300e_coco/best_coco_bbox_mAP_epoch_292.pth \
#    --img_dir /data2/wangfj/data/road_damage/test_images \
#    --out_dir /data2/wangfj/data/road_damage/rtmdet_best

# python infer.py \
#     --config /home/sejong/seonghyeon/road_damage_detection/RTMDet/configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
#     --ckpt /home/sejong/seonghyeon/road_damage_detection/data/best_coco_bbox_mAP_epoch_292.pth \
#     --img_dir /home/sejong/seonghyeon/road_damage_detection/data/test/img \
#     --out_dir /home/sejong/seonghyeon/road_damage_detection/infer/rtdmdet_pseudo_best #\
#     # --prefix 
#  python infer.py \
#     --config work_dirs/rtmdet_pseudo/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
#     --ckpt work_dirs/rtmdet_pseudo/best_coco_bbox_mAP_epoch_292.pth \
#     --img_dir /data2/wangfj/data/road_damage/test_images \
#     --out_dir /data2/wangfj/data/road_damage/rtmdet_pseudo_best #\
#     # --prefix 
   python infer.py \
         --config /home/sejong/seonghyeon/road_damage_detection/RTMDet/configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
          --ckpt /home/sejong/seonghyeon/road_damage_detection/data/best_coco_bbox_mAP_epoch_292.pth \
          --img_dir /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/country1 \
          --out_dir /home/sejong/seonghyeon/road_damage_detection/infer/rtmdet_pseudo_best_country1

python infer.py \
         --config /home/sejong/seonghyeon/road_damage_detection/RTMDet/configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
          --ckpt /home/sejong/seonghyeon/road_damage_detection/data/best_coco_bbox_mAP_epoch_292.pth \
          --img_dir /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/country2 \
         --out_dir /home/sejong/seonghyeon/road_damage_detection/infer/rtmdet_pseudo_best_country2

python infer.py \
         --config /home/sejong/seonghyeon/road_damage_detection/RTMDet/configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
         --ckpt /home/sejong/seonghyeon/road_damage_detection/data/best_coco_bbox_mAP_epoch_292.pth \
         --img_dir /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/country3 \
         --out_dir /home/sejong/seonghyeon/road_damage_detection/infer/rtmdet_pseudo_best_country3

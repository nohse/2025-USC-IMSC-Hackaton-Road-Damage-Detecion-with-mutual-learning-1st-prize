#python infer.py \
#	--config work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
#	--ckpt work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/best_coco_bbox_mAP_epoch_16.pth \
#	--img_dir /data0/wangfj/road_damage/test_images \
#	--out_dir /data0/wangfj/road_damage/codetr_best_tta \
#	--tta

python infer.py \
        --config /home/sejong/seonghyeon/road_damage_detection/Co-DETR/configs/co_detr/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
        --ckpt /home/sejong/seonghyeon/road_damage_detection/data/best_coco_bbox_mAP_epoch_13.pth \
        --img_dir /home/sejong/seonghyeon/road_damage_detection/yolodata/test/images \
        --out_dir /home/sejong/seonghyeon/road_damage_detection/infer/codetr_pseudo_best_test

# python infer.py \
#         --config /home/sejong/seonghyeon/road_damage_detection/Co-DETR/configs/co_detr/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
#         --ckpt /home/sejong/seonghyeon/road_damage_detection/data/best_coco_bbox_mAP_epoch_13.pth \
#         --img_dir /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/country2 \
#         --out_dir /home/sejong/seonghyeon/road_damage_detection/infer/codetr_pseudo_best_country2

# python infer.py \
#         --config /home/sejong/seonghyeon/road_damage_detection/Co-DETR/configs/co_detr/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
#         --ckpt /home/sejong/seonghyeon/road_damage_detection/data/best_coco_bbox_mAP_epoch_13.pth \
#         --img_dir /home/sejong/seonghyeon/road_damage_detection/yolodata/hack_data/images/val/country3 \
#         --out_dir /home/sejong/seonghyeon/road_damage_detection/infer/codetr_pseudo_best_country3


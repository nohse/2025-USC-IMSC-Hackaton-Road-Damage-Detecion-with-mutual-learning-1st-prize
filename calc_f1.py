import os
import glob
from PIL import Image

# ———— 설정 —————————————————————————————
GT_DIR   = "/home/sejong/seonghyeon/road_damage_detection/hack_data/labels/val_all"
PRED_DIR = "/home/sejong/seonghyeon/road_damage_detection/hack_data/images/val_all/predictions_txt"
IMG_DIR  = "/home/sejong/seonghyeon/road_damage_detection/hack_data/images/val_all"
# —————————————————————————————————————————

def xywh_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return [x1, y1, x2, y2]

def load_gt(gt_path, img_path):
    img_w, img_h = Image.open(img_path).size
    boxes = []
    with open(gt_path) as f:
        for line in f:
            cls, *rest = map(float, line.split())
            boxes.append((int(cls), xywh_to_xyxy(rest, img_w, img_h)))
    return boxes

def load_pred(pred_path):
    preds = []
    with open(pred_path) as f:
        for line in f:
            parts = line.split()
            cls = int(parts[0])
            x1, y1, x2, y2, conf = map(float, parts[1:])
            preds.append((cls, [x1, y1, x2, y2], conf))
    return preds

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0

def evaluate_at_thresh(iou_thresh):
    TP = 0; FP = 0; FN = 0

    for gt_file in glob.glob(os.path.join(GT_DIR, "*.txt")):
        base = os.path.basename(gt_file)
        img_file  = os.path.join(IMG_DIR, base.replace(".txt", ".jpg"))
        pred_file = os.path.join(PRED_DIR, base)

        if not (os.path.isfile(img_file) and os.path.isfile(pred_file)):
            continue

        gt_boxes   = load_gt(gt_file, img_file)
        pred_boxes = load_pred(pred_file)
        used_gt = [False] * len(gt_boxes)

        # confidence 상관 없이 모두 매칭
        pred_boxes.sort(key=lambda x: x[2], reverse=True)

        # match predictions → TP/FP
        for cls_p, box_p, _ in pred_boxes:
            # find best matching unused GT of same class
            best_iou = 0; best_i = -1
            for i, (cls_g, box_g) in enumerate(gt_boxes):
                if used_gt[i] or cls_p != cls_g:
                    continue
                cur_iou = iou(box_p, box_g)
                if cur_iou > best_iou:
                    best_iou = cur_iou; best_i = i

            # → 아래처럼 수정
            if best_i >= 0 and best_iou >= iou_thresh:
                TP += 1
                used_gt[best_i] = True
            else:
                FP += 1


        # 남은 GT는 FN
        for used in used_gt:
            if not used:
                FN += 1

    # global precision/recall/F1
    prec = TP/(TP+FP) if TP+FP>0 else 0
    rec  = TP/(TP+FN) if TP+FN>0 else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    return TP, FP, FN, prec, rec, f1

def find_best_threshold():
    best = {"thresh":None, "F1":-1}
    print(f"{'IoU':>5} | {'TP':>4} {'FP':>4} {'FN':>4} | {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-"*44)
    for i in range(0, 21):
        thresh = i * 0.05
        TP, FP, FN, prec, rec, f1 = evaluate_at_thresh(thresh)
        print(f"{thresh:5.2f} | {TP:4d} {FP:4d} {FN:4d} | {prec:6.3f} {rec:6.3f} {f1:6.3f}")
        if f1 > best["F1"]:
            best.update({"thresh":thresh, "F1":f1})
    print("\n>> Best IoU threshold: {:.2f} with F1 = {:.3f}".format(
        best["thresh"], best["F1"]))
    
if __name__ == "__main__":
    find_best_threshold()



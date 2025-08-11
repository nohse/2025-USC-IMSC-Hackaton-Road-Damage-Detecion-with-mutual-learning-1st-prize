# 2025_USCIMSCHackaton-RoadDamageDetectionwithmutuallearning-1st_prize

For the best performing model, you can download our best performing model from [here](https://drive.google.com/drive/folders/1-MkWAZQ8RYX0kUeHmTsdwjRY4em67CCS?usp=sharing) and run it.

# Model Architecture
<p align="center"> <img src="assets/RDD-architecture.png" alt="Mutual Learning Architecture (Co-DETR ↔ RTMDet → KD to YOLOv10)" width="900"> </p>
Key idea (3 stages)

Stage 1 — Independent training: Train transformer-based Co-DETR and CNN-based RTMDet on the road-damage dataset (different inductive biases).

Stage 2 — Mutual learning: Exchange pseudo-labels and co-train so each model learns from the other’s strengths (classification and box regression guidance).

Stage 3 — Knowledge distillation: Distill the ensemble knowledge into a fast YOLOv10 student. (Optionally export to TensorRT for deployment.)

# Results
<p align="center"> <img src="assets/RDDg-result.jpg" alt="Mutual Learning result (Co-DETR ↔ RTMDet → KD to YOLOv10)" width="500"> </p>

We took first place with a substantial lead over the second-place team.

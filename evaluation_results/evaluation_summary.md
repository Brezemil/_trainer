# Strict Test Set Evaluation Summary (COCO Metrics)

This table aggregates the strict pycocotools AP/AR metrics computed on the test split.
Scores are reported as **Mean ± Standard Deviation** across the configured seeds.

| Model Variant | Runs | mAP@0.50:0.95 | mAP@0.50 | mAP@0.75 | AP (Small) | AP (Medium) | AP (Large) | AR@100 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| rtdetr-l | 3 | 0.00003 ± 0.00001 | 0.00014 ± 0.00011 | 0.00001 ± 0.00001 | 0.00000 ± 0.00000 | 0.00001 ± 0.00001 | 0.00007 ± 0.00002 | 0.00937 ± 0.00265 |
| yolo11s | 3 | 0.00019 ± 0.00004 | 0.00066 ± 0.00017 | 0.00006 ± 0.00001 | 0.00000 ± 0.00000 | 0.00006 ± 0.00004 | 0.00035 ± 0.00009 | 0.05917 ± 0.00463 |
| yolo26n | 1 | 0.00017 ± 0.00000 | 0.00117 ± 0.00000 | 0.00002 ± 0.00000 | 0.00000 ± 0.00000 | 0.00001 ± 0.00000 | 0.00033 ± 0.00000 | 0.03562 ± 0.00000 |
| yolo26s | 3 | 0.00018 ± 0.00005 | 0.00079 ± 0.00026 | 0.00005 ± 0.00001 | 0.00000 ± 0.00000 | 0.00017 ± 0.00019 | 0.00033 ± 0.00007 | 0.05500 ± 0.00627 |
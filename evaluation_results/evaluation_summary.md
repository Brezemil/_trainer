# Strict Test Set Evaluation Summary (COCO Metrics)

This table aggregates the strict pycocotools AP/AR metrics computed on the test split.
Scores are reported as **Mean ± Standard Deviation** across the configured seeds.

| Model Variant | Runs | mAP@0.50:0.95 | mAP@0.50 | mAP@0.75 | AP (Small) | AP (Medium) | AP (Large) | AR@100 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| rtdetr-l | 3 | 0.00011 ± 0.00009 | 0.00040 ± 0.00029 | 0.00003 ± 0.00004 | 0.00000 ± 0.00000 | 0.00007 ± 0.00004 | 0.00022 ± 0.00020 | 0.03125 ± 0.02244 |
| yolo11n | 3 | 0.00038 ± 0.00022 | 0.00123 ± 0.00065 | 0.00008 ± 0.00005 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00070 ± 0.00037 | 0.05292 ± 0.00390 |
| yolo26n | 3 | 0.00023 ± 0.00003 | 0.00060 ± 0.00019 | 0.00018 ± 0.00002 | 0.00000 ± 0.00000 | 0.00002 ± 0.00003 | 0.00043 ± 0.00005 | 0.04042 ± 0.00560 |
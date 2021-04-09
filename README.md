**VinBigData Chest X-ray Abnormalities Detection**

I suppose the content of this repository will be copied into "net_sergey_yolo" directory of the final solution.

**Yolo v5 training**

**Stage 1.** At first, we produce the “average” boxes from original
train.csv using WBF (weighted box fusion). Using these labels we train
Yolo v5 with the default resolution 640 pixels. Training was done using
5KFold. During inference besides an original image, we used a
horizontally flipped image and then combined their predictions (Test
Time Augmentation) using WBF. Predictions from folds are combined using
WBF too.

It was done in another script.

**Stage 2.** At this stage we excluded all images from ‘R8’, ‘R9’ and
‘R10’ radiologists. Every image with boxes was tripled using 3 variants
of boxes (there are 248 images, after tripling they become 248 \* 3 =
744 images). To speed up training, only a part of empty images was used
(namely, 6893 images). The learning rate was decreased from default 0.01
to 0.0005. Training was done using the same 5 folds as in Stage 1,
starting from weights obtained at the previous stage. Inference is the
same as at Stage 1.

```
python3 net_sergey_yolo/data/create_yolo_labels_separate_rads.py
python3 net_sergey_yolo/data/yolo_converter_stage2.py
python3 net_sergey_yolo/yolo_stage2_train.py --weights ./net_sergey_yolo/weights/stage1_fold0.pt --cfg ./net_sergey_yolo/yolo5/configYolo5/yolov5x_roma.yaml --data ./net_sergey_yolo/yolo5/configYolo5/vinbigdata0_stage2.yaml --hyp net_sergey_yolo/stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./net_sergey_yolo/runs/train --workers 2
python3 net_sergey_yolo/yolo_stage2_train.py --weights ./net_sergey_yolo/weights/stage1_fold1.pt --cfg ./net_sergey_yolo/yolo5/configYolo5/yolov5x_roma.yaml --data ./net_sergey_yolo/yolo5/configYolo5/vinbigdata1_stage2.yaml --hyp net_sergey_yolo/stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./net_sergey_yolo/runs/train --workers 2
python3 net_sergey_yolo/yolo_stage2_train.py --weights ./net_sergey_yolo/weights/stage1_fold2.pt --cfg ./net_sergey_yolo/yolo5/configYolo5/yolov5x_roma.yaml --data ./net_sergey_yolo/yolo5/configYolo5/vinbigdata2_stage2.yaml --hyp net_sergey_yolo/stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./net_sergey_yolo/runs/train --workers 2
python3 net_sergey_yolo/yolo_stage2_train.py --weights ./net_sergey_yolo/weights/stage1_fold3.pt --cfg ./net_sergey_yolo/yolo5/configYolo5/yolov5x_roma.yaml --data ./net_sergey_yolo/yolo5/configYolo5/vinbigdata3_stage2.yaml --hyp net_sergey_yolo/stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./net_sergey_yolo/runs/train --workers 2
python3 net_sergey_yolo/yolo_stage2_train.py --weights ./net_sergey_yolo/weights/stage1_fold4.pt --cfg ./net_sergey_yolo/yolo5/configYolo5/yolov5x_roma.yaml --data ./net_sergey_yolo/yolo5/configYolo5/vinbigdata4_stage2.yaml --hyp net_sergey_yolo/stage2_params.yaml --batch-size 6 --epochs 15 --img-size 640 --project ./net_sergey_yolo/runs/train --workers 2
```

Files “best.pt“ from folders “runs\\train\\exp\\weights”,
“runs\\train\\exp2\\weights” etc should be copied to “weights” folder.

**Yolo v5 inference**

To run inference on the test set for stage2 (stage 1 was done in another script):

```
./net_sergey_yolo/download_test_data.sh
python3 net_sergey_yolo/yolo_inf.py --stage 2
```

The weights should be in the “weights” folder.

After that it’s needed to run postprocessing:

```
python3 net_sergey_yolo/postprocess.py -f net_sergey_yolo/yolo_stage2_all_folds.csv
```
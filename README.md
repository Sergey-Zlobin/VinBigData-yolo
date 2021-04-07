**VinBigData Chest X-ray Abnormalities Detection**

**Yolo v5 training**

**Stage 1.** At first, we produce the “average” boxes from original
train.csv using WBF (weighted box fusion). Using these labels we train
Yolo v5 with the default resolution 640 pixels. Training was done using
5KFold. During inference besides an original image, we used a
horizontally flipped image and then combined their predictions (Test
Time Augmentation) using WBF. Predictions from folds are combined using
WBF too.

**Stage 2.** At this stage we excluded all images from ‘R8’, ‘R9’ and
‘R10’ radiologists. Every image with boxes was tripled using 3 variants
of boxes (there are 248 images, after tripling they become 248 \* 3 =
744 images). To speed up training, only a part of empty images was used
(namely, 6893 images). The learning rate was decreased from default 0.01
to 0.0005. Training was done using the same 5 folds as in Stage 1,
starting from weights obtained at the previous stage. Inference is the
same as at Stage 1.

**Yolo v5 inference**

To run inference on the test set,

python yolo\_inf.py --stage 1

python yolo\_inf.py --stage 2

The weights should be in the “weights” folder.

After that it’s needed to run postprocessing:

python postprocess.py -f yolo\_stage1\_all\_folds.csv

python postprocess.py -f yolo\_stage2\_all\_folds.csv

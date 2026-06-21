
total videos: 53 
total animals : 18 
G2C01 - died before third session only S1 and S2 available 


videos in the list for training datset dlc: 

G1C01_S1 : climbs on top 
G1C02_S2 : Stain in the beginning of the video
G1C03_S3: normal random
G2C05_S1 : spends all its time around (camera top shelter)
G2C06_S1 : merges into rim
G2C06_S2 : various thigmotaxis behaviour angles 
G3C02_S1 : distinct body end image (Quality and shape)
G3C05_S1 : under the bridge (camera bottom shelter) for a considerably long time
G4C02_S1 : bean shaped and one leg spread out 
G4C04_S3 : normal behaviour example


Total labelled frames:
Basler_G4_C02_S1_acA2040-90um_: 113
Basler_G3_C02_S1_acA2040-90um_: 751
Basler_G3_C05_S1_acA2040-90um_: 168
Basler_G1_C03_S3_acA2040-90um_: 279
Basler_G4_C04_S3_acA2040-90um_: 360
Basler_G2_C05_S1_acA2040-90um_: 1369
Basler_G2_C06_S2_acA2040-90um_: 30
Basler_G1_C01_S1_acA2040-90um_: 83
Basler_G1_C02_S2_acA2040-90um_: 102
Basler_G2_C06_S1_acA2040-90um_: 538

Total labeled frames: 3793



Phase 1 labeling and initial training (Shuffle 1)

Started with 10 videos selected across G1-G4 groups covering diverse behaviors including thigmotaxis, rim-merging, shelter approaches and normal open-field locomotion. Used k-means frame extraction (30 frames per video) giving an initial set of ~270 frames. Additionally extracted targeted frames from known shelter-visit timestamps provided manually, bringing the total to around 500 labeled frames. Body parts labeled: *head* and *body_end*.

Training was run on the server with weights initialized to (`snapshot-200.pt`) which is from the previous open arena odour preference experiment that tracked the same body parts (and more). Training ran for 200 epochs with batch size 8.

Final evaluation on Shuffle 1:

    train rmse: 3.00 px     test rmse: 3.02 px
    train mAP: 52.85        test mAP: 56.68
    train mAR: 67.05        test mAR: 64.23

No overfitting observed (train and test RMSE nearly identical). However mAP of ~57% was considered insufficient given that *shelter occlusion* frames are important to the boldness/shyness analysis and were very few in number in the training set (~60 occlusion frames out of 500 total, roughly 12%). and predictions were sort of 3 pixels away from actual points. 

Phase 2 labeling and retraining (Shuffle 2)

All shelter-visit timestamp windows provided manually were used to extract frames at *1 frame per 10 frames (0.4s at 25fps)*
3292 new frames  specifically showing full and partial occlusion under both circular shelters. Combined with the original labeled set of 500 = large occlusion focused dataset. A new training dataset was created as Shuffle 2 to keep Shuffle 1 intact for comparison.
network also has same weight initialization for shuffle 2 from the old open arena model. 


training stats: epoch 200 Epoch 200/200 (lr=1e-05), train loss 0.00090, valid loss 0.00119, GPU: 3838.0/45518.2 MiB
Model performance:

  metrics/test.rmse:           2.96
  metrics/test.rmse_pcutoff:   2.96
  metrics/test.mAP:           39.52
  metrics/test.mAR:           56.53


Running evaluate network step showed snapshot 190 is better
Evaluation results file: DLC_Resnet50_Open_Arena_Jun26_Shyness_BoldnessJun12shuffle2_snapshot_best-190-results.csv
Evaluation results for DLC_Resnet50_Open_Arena_Jun26_Shyness_BoldnessJun12shuffle2_snapshot_best-190-results.csv (pcutoff: 0.6):

train rmse             2.68
train rmse_pcutoff     2.68
train mAP             44.92
train mAR             61.82
test rmse              2.95
test rmse_pcutoff      2.95
test mAP              39.70
test mAR              56.68
Name: (0.95, 2, 190, -1, 0.6), dtype: float64


RMSE = root mean squared error 
This measures the average distance in pixels between where you clicked a bodypart during labeling and where the model predicted it to be. So test RMSE: 2.95 means on frames the model never saw during training, its predictions land on average ~3 pixels away from your labels. Lower is better.

mAP: For each predicted bodypart location, the model assigns a confidence score (0 to 1). mAP asks: across different confidence thresholds, how often is the model both finding the bodypart (recall) and being right about where it is (precision)?
A prediction "counts" as correct if it lands within a certain distance of the true label. mAP averages this correctness score across all bodyparts and all confidence thresholds

In this regard, for this shuffle, 

RMSE improved (2.95 vs 3.02) therefore the model is more spatially accurate. But mAP dropped. The mAP drop likely comes from the test set being only 190 frames (5% of ~3800), which are now drawn from a very different distribution than shuffle 1's 26 test frames. The test set changed between shuffles, making direct mAP comparison unreliable.
The RMSE improvement is notable, the model predicts locations ~0.07px more accurately on unseen frames.

Inspecting two analyzed and labelled videos with shuffle2 (snapshot-best-190, ResNet50) revealed that:

G3_C05_S1 - many instances of shelter occlusion: was accurate
G2_C05_S1 — the most heavily labelled video during annotation because it has the most number of occlusion instances, therefore contributes greatly to the training set for the network.  (~1369 labeled occlusion frames). Specifically, this video is chosen because the cockroach is blurry as well and it appears less darker than the other videos in the training dataset. Checking the labeled output confirmed that the model is indeed more accurate now and has gotten better at detecting cases where the animal is heavily occluded under the shelter (due to the focus issue), both body_parts : head and body_end appear to be stable and are correctly predicted even as the animals moves across the shelterl. Time spent outside the shelter is also accurately predicted. Therefore, I have decided to go on with analyzing the whole dataset and then labelling them with snapshot 190 from shuffle 2 on the cluster.


Here's the note for your repo, plus the formula breakdown.

---

**Arena/shelter pixel measurements

*Pixel-to-cm calibration*

Arena inner diameter measured via Hough circle detection (OpenCV) on a still frame, cross-validated across 3 videos from different groups/days (G1_C01_S1, G3_C02_S1, G4_C04_S3) to confirm camera stability. Measurements were highly consistent (std dev 0.74px across videos)

    Arena diameter (pixels): 930.27 px (mean across 3 videos)
    Arena diameter (real): 59 cm
    Conversion factor: 930.27 / 59 = 15.77 px/cm
    Inverse: 59 / 930.27 = 0.0634 cm/px

Shelter diameter measured for cross-validation: ~137px detected, converting to ~8.69cm against a stated physical diameter of 8cm 

*Thigmotaxis zone*

5cm band adjacent to the arena wall  measured inward from the arena boundary.

    5 cm × 15.77 px/cm = 78.85 px

So thigmotaxis zone = annular region between radius (465 - 79) = *386px* and *465px* from arena center, where arena center ≈ (499.5, 507) and outer radius ≈ 465px.

From a google search: how the cirular elements got their measurements 

Hough Circle Transform is a computer vision algorithm that detects circular shapes in an image. It works by:

1. Detecting edges in the image (boundaries where pixel brightness changes sharply — like the arena wall against the floor)
2. For every edge point, it considers all possible circles that could pass through that point at various radii
3. It accumulates "votes" in a parameter space (center x, center y, radius) — points that are consistent with the same circle vote for the same parameters
4. The circle with the most votes (the one most consistent with the actual edge pattern) is returned

In OpenCV, the call looks like:

```python
cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)
```

- `dp=1`: resolution of the accumulator matches the image resolution
- `minDist=500`: minimum distance between detected circle centers (prevents detecting near-duplicate circles)
- `param1=50`: edge detection sensitivity threshold
- `param2=30`: how many "votes" needed to count as a real circle — lower means more lenient detection
- `minRadius`/`maxRadius`: search bounds, set based on expected arena/shelter size in pixels

The output gives (x_center, y_center, radius) for the most confident circle found within the constraints.

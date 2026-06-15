
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

No overfitting observed (train and test RMSE nearly identical). However mAP of ~57% was considered insufficient given that *shelter occlusion* frames are central to the boldness/shyness analysis and were underrepresented in the training set (~60 occlusion frames out of 500 total, roughly 12%). and predictions were sort of 3 pixels away from actual points. 

Phase 2 labeling and retraining (Shuffle 2)

All shelter-visit timestamp windows provided manually were used to extract frames at *1 frame per 10 frames (0.4s at 25fps)*
3292 new frames  specifically showing full and partial occlusion under both circular shelters. Combined with the original labeled set of 500 = large occlusion focused dataset. A new training dataset was created as Shuffle 2 to keep Shuffle 1 intact for comparison.
network also has same weight initialization for shuffle 2 from the old open arena model. 

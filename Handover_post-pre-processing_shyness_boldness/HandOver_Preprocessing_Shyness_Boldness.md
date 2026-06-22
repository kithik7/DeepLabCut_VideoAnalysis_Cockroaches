# Open Arena Shyness-Boldness Dataset
## DeepLabCut Tracking Pipeline Output
**Project:** Open_Arena_Jun26_Shyness_Boldness
**Experiment dates:** May-June 2026
**Animals:** 18 cockroaches across 4 groups (G1-G4), 53 sessions total

---

##### Summary

| Parameter | Value |
|---|---|
| Total animals | 18 |
| Groups | G1, G2, G3, G4 |
| Total videos | 53 |
| Sessions per animal | 3 (except G2_C01: 2 sessions) |
| Video resolution | 1024 x 1024 px |
| Frame rate | 25 fps |
| Video duration | ~61 minutes per session |

---

##### Arena & Shelter Dimensions

 Pixel-to-cm Conversion

Conversion rate was obtained by detecting the arena boundary via Hough Circle Transform (OpenCV) across 3 videos from different groups and dates. 
Measurements were highly consistent (std dev 0.74 px), therefore, the camera was stable during data collection

| Parameter | Pixels | cm |
|---|---|---|
| Arena diameter | 930.27 px | 59 cm |
| Arena radius | 465 px | 29.5 cm |
| Arena center | (500, 507) px | — |
| Shelter diameter | ~137 px | 8 cm |
| Shelter radius | ~69 px | 4 cm |
| **Conversion factor** | **15.77 px/cm** | **0.0634 cm/px** |

### Shelter Coordinates (pixels) - origin top left (cartesian coordinates = bottom left, video coordinate = origin top left)

the mismatch dates back to cathode ray computers ( pretty interesting google rabbit hole if you are interested)


CRT Monitor Legacy: Early television and computer screens used an electron beam to scan the display. This beam always started sweeping from the top-left corner, moved horizontally to the right, and then stepped downward to the next line.
Reading Direction Alignment: Computer memory stores data sequentially. Aligning the pixel coordinate system with the standard Western text layout (top-to-bottom, left-to-right) made it intuitive to map consecutive memory addresses directly to screen pixels.
Matrix Representation: Computer graphics mathematically treat images as data matrices. In standard matrix notation, row indices always count downward from the top row, perfectly matching the inverted Y-axis used in image coordinates.


https://www.mathworks.com/matlabcentral/answers/521441-why-do-points-obtained-from-an-image-appear-to-be-flipped-in-the-vertical-direction-when-plotted#answer_428959
The flipping occurs because the coordinate system used by images is different from the conventional Cartesian coordinate system. 
For Cartesian system, the bottom left corner is chosen as the origin with the Y axis along the upward direction. But for images, the top left corner is chosen as the origin with the Y axis along the downward direction. 
The coordinates of the checkerboard points are obtained with respect to the coordinate system of the images. 
Hence when these coordinates are plotted in Cartesian coordinates, the points will appear to be flipped along the vertical direction

| Shelter | Center (x, y) | Radius |
|---|---|---|
| Top shelter | (490, 213) | 69 px |
| Bottom shelter | (502, 778) | 68 px |

##### Thigmotaxis Zone

Defined as the 5 cm band adjacent to the arena wall, measured inward from the arena boundary.

| Parameter | Value |
|---|---|
| Width | 5 cm = 79 px |
| Inner radius | 386 px from arena center |
| Outer radius | 465 px from arena center (arena wall) |

A cockroach is considered in the thigmotaxis zone when its distance from the arena center is between 386 px and 465 px.

See `arena_calibration_diagram.png` for a visual reference of all zones.

---

##### Tracked Body Parts

Two body parts were tracked per animal per frame using DeepLabCut (ResNet50, Shuffle 2, snapshot-best-190):

- **head** — anterior tip of the cockroach head
- **body_end** — posterior tip of the abdomen

---

##### Directory Structure

```
predictions/
  Group1/
    G1_C01/
      G1_C01_S1_25-05-26/
        Basler_..._DLC_...shuffle2_snapshot_best-190.h5
        Basler_..._DLC_...shuffle2_snapshot_best-190.csv
        Basler_..._DLC_...shuffle2_snapshot_best-190_meta.pickle
        Basler_..._DLC_...shuffle2_snapshot_best-190_full.pickle
        Basler_..._DLC_...shuffle2_snapshot_best-190_p60_labeled.mp4
        plot_poses/
          trajectory.png
          plot.png
          plot-likelihood.png
          hist.png
      G1_C01_S2_27-05-26/
        ...
    G1_C02/
      ...
  Group2/
    ...
  Group3/
    ...
  Group4/
    ...
```

---

**CSV/H5 File Format**

Each CSV/H5 file contains one row per frame. Column structure:

```
scorer        | DLC_Resnet50_..._snapshot_best-190
bodypart      | head                    | body_end
coordinate    | x      | y   | likelihood | x   | y   | likelihood
```

- x, y: pixel coordinates of the bodypart in that frame 
- likelihood: model confidence (0-1). Values below 0.6 are unreliable and should be analyzed separately as a filtered dataset
excluding the frames that fall below this threshold and one including all of them.

To read in Python:
```python
import pandas as pd
df = pd.read_hdf('path/to/file.h5')

# or from CSV
df = pd.read_csv('path/to/file.csv', header=[1,2], index_col=0)
```

---

##### Zone Classification

To classify a cockroach's position in each frame:

```python
import numpy as np

ARENA_CENTER = (500, 507)   # px
ARENA_RADIUS = 465          # px
THIGMO_INNER_RADIUS = 386   # px (5cm from wall)

SHELTER_TOP = {'center': (490, 213), 'radius': 69}
SHELTER_BOTTOM = {'center': (502, 778), 'radius': 68}

def dist(x, y, cx, cy):
    return np.sqrt((x - cx)**2 + (y - cy)**2)

def classify_position(x, y):
    d_arena = dist(x, y, *ARENA_CENTER)
    d_top = dist(x, y, *SHELTER_TOP['center'])
    d_bot = dist(x, y, *SHELTER_BOTTOM['center'])

    if d_top <= SHELTER_TOP['radius']:
        return 'under_shelter_top'
    elif d_bot <= SHELTER_BOTTOM['radius']:
        return 'under_shelter_bottom'
    elif d_arena >= THIGMO_INNER_RADIUS:
        return 'thigmotaxis'
    else:
        return 'open_arena'
```

---

##### Plot Poses Folde

Each `plot_poses/` subfolder contains 4 diagnostic plots generated by DLC:

| File | Contents |
|---|---|
| `trajectory.png` | 2D movement path of head and body_end across the full session |
| `plot.png` | X and Y coordinates over time for both body parts |
| `plot-likelihood.png` | Model confidence (likelihood) over time per body part |
| `hist.png` | Histogram of detected x/y positions across the session |

---

Model Details

| Parameter | Value |
|---|---|
| Architecture | ResNet50 |
| DLC version | 3.0.0 |
| Shuffle | 2 |
| Best snapshot | epoch 190 |
| Training frames | ~3800 (including targeted shelter-occlusion frames) |
| Test RMSE | 2.95 px (~0.19 cm) |
| Test mAP | 39.70% |
| pcutoff | 0.6 |

---

**Tracking Quality Assessment**

Quality metrics were computed across all 53 sessions from the DLC output coordinates. 
See `tracking_quality_assessment_shyness_boldness.csv` for per-session values and `tracking_quality_plots.png` for distributions.

Metrics Computed:

1. Fraction of frames above pcutoff

For each bodypart, per session of T frames:
quality = count(likelihood[t] >= 0.6 for t in 1..T) / T

| Metric | Mean | Min |
|---|---|---|
| head above pcutoff | 0.997 | 0.902 |
| body_end above pcutoff | 0.997 | 0.867 |

 99.7% of frames per session are tracked with high confidence. The 0.6 threshold was also evaluated to see if it holds for frame where it was known from the training dataset
 that the animal is under the shelter. The result of this validation was: 
 known shelter-visit frames showed 0.0% (head) and 0.1% (body_end) of under-shelter frames fall below 0.6. Therefore, the threshold holds.

2. Spatial plausibility

Fraction of frames where the predicted midpoint between head and body_end falls inside the arena boundary:


midpoint_x[t] = (head_x[t] + body_x[t]) / 2
midpoint_y[t] = (head_y[t] + body_y[t]) / 2
dist_from_center[t] = sqrt((midpoint_x[t] - 500)^2 + (midpoint_y[t] - 507)^2)
spatial_plausibility = count(dist_from_center[t] <= 465) / T

| Metric | Mean |
|---|---|
| Spatial plausibility | 0.999 |

less than 0.1% of frames per session are predicted outside the arena boundary.

3. Temporal smoothness (jump rate)

Fraction of consecutive frames where the predicted position jumps more than the maximum physically plausible displacement. 
At 25fps, a cockroach at maximum sprint speed (~1.5 m/s) covers at most 6 cm/frame = 95 px/frame: 

Robert J. Full, Michael S. Tu; Mechanics of A Rapid Running Insect: Two-, Four-and Six-Legged Locomotion. 
J Exp Biol 1 March 1991; 156 (1): 215–231. doi: https://doi.org/10.1242/jeb.156.1.215

jump[t] = sqrt((x[t] - x[t-1])^2 + (y[t] - y[t-1])^2)
jump_rate = count(jump[t] > 95) / T

| Metric | Mean | Max |
|---|---|---|
| Head jump rate | 0.0001 | 0.0013 |
| Body_end jump rate | 0.0001 | 0.0029 |
almost no unusual tracking jumps across all sessions. At worst, 0.3% of frames in any single session had this kind of a flux!


---

**Notes**

- G2_C01 has only 2 sessions (S1, S2) - S3 was not recorded
- Shelters are circular, dark grey, top-down view
- The cockroach is sometimes fully occluded under a shelter and due to the camera focus issue, the model was trained separately with 3000 frames approximately that only had frames where
the animal was under the shelter and occluded. The model prediction is naturally better now and the labelled videos are testament to this. 
- All coordinates are in pixel space (origin top-left). Use the conversion factor (15.77 px/cm) to convert to centimetres

Happy Analyzing! If you have any more questions don't hesitate to contact me over slack.

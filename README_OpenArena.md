## DeepLabCut_VideoAnalysis_Cockroaches

* Markdown file and some scripts that can be reused for video analysis with DLC. 
* The pipeline requires one to label videos on the GUI, upload the directory on the cluster, train the model on the cluster, run analyze videos and generate trajectory plots+ raw coordinate csvs. 
* Following this, for just cockroach data, theres mov_analysis and shelter analysis scripts.

### Movement and Shelter Preference Analysis PM1 - Project Documentation 
The paths in the config.yaml files are for the AGM-NAS cluster (nawrot lab), the folders specifically have local paths for the linux desktop in the same lab. This is subject to change of course relative to the project and my computational apparatus. 


**Git Working Directory Path:** `/home/keerthie/Desktop/Keerthi/Project_Mod1/Movement_Analysis_Reusable_CompPipeline/`

***

### Project Directory & Repository Links

* [Data_Movement_Analysis/](Data_Movement_Analysis/)
  Main directory containing core tracking datasets.

  * [Data_Movement_Analysis/Movement_Analysis_Group/](Data_Movement_Analysis/Movement_Analysis_Group/)
    Group trial tracking spreadsheet data sheets (all frames & tracking metrics).

  * [Data_Movement_Analysis/Movement_Analysis_Solo/](Data_Movement_Analysis/Movement_Analysis_Solo/)
    Solo trial tracking spreadsheet data sheets (all frames & tracking metrics).


* [Pipeline/](Pipeline/)
  Main directory containing configuration files, notes, and execution scripts.

  * [Pipeline/config_cluster.yaml](Pipeline/config_cluster.yaml)
    Configuration file containing cluster specific server paths.

  * [Pipeline/config_group_cluster.yaml](Pipeline/config_group_cluster.yaml)
    Group configuration file containing cluster specific server paths.

  * [Pipeline/bodyparts.txt](Pipeline/bodyparts.txt)
    Reference file listing tracked body parts.

  * [Pipeline/imp_notes.txt](Pipeline/imp_notes.txt)
    Important pointers and reference notes regarding tmux usage and cluster environments.

  * Pipeline/Generate_Shelter_Data/ Subfolder containing scripts to make data frames with time spent under            shelter parameters.
    
  * [Pipeline/Generate_Movement_Data/](Pipeline/Generate_Movement_Data/)
    Subfolder containing scripts to extract tracking datasets.

    * [Pipeline/Generate_Movement_Data/movement_analysis_all_frames.py](Pipeline/Generate_Movement_Data/movement_analysis_all_frames.py)
      Script for extracting all frames data.

    * [Pipeline/Generate_Movement_Data/movement_analysis_tracking_efficiency_GROUP.py](Pipeline/Generate_Movement_Data/movement_analysis_tracking_efficiency_GROUP.py)
      Script for measuring tracking efficiency in group trials.

    * [Pipeline/Generate_Movement_Data/movement_analysis_tracking_efficiency.py](Pipeline/Generate_Movement_Data/movement_analysis_tracking_efficiency.py)
      Script for tracking efficiency (*Note: group_all_frames component is missing*).


  * `Pipeline/Pytorch_Models_Old/` 
    DeepLabCut PyTorch models from September 2025 with separate test and train folders.


  * [Pipeline/Post_Training_Scripts/](Pipeline/Post_Training_Scripts/)
    Subfolder containing evaluation and video visualization scripts.

    * [Pipeline/Post_Training_Scripts/analyze_all_videos.py](Pipeline/Post_Training_Scripts/analyze_all_videos.py)
      Script used to run inference and analyze all videos.

    * [Pipeline/Post_Training_Scripts/labelled_videos_trajectoryplots.py](Pipeline/Post_Training_Scripts/labelled_videos_trajectoryplots.py)
      Script used to generate labeled videos and plot trajectory graphs.


### Server Reference & Cluster Execution

#### Environment Initialization
```bash
# Load module infrastructure and activate environment
module load python3/anaconda3
source /rhomes/kkesavan/miniforge3/etc/profile.d/conda.sh
conda activate /rhomes/kkesavan/miniforge3/envs/dlc_pytorch_gpu
```

#### Path - cluster
```text
/rhomes/kkesavan/miniforge3/envs/DEEPLABCUT/bin/python
```

#### Tmux attach
```bash
# Attach to active terminal multiplexer session
cd dlcv_project
tmux attach -t 1
```

***

### Experimental Spatial Arena Metrics

* **Total Arena Video Resolution:** `1507 x 1076` pixels
* **New Left Shelter Region Coordinates:** `x=[100 - 700], y=[300 - 900]`
* **New Right Shelter Region Coordinates:** `x=[1300 - 1900], y=[300 - 900]`

#### ImageJ Coordinate Workspace Matrix

| Region Location | Area (px) | Mean Intensity | X-Start | Y-Start | Width | Height |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Left Arena Border** | 41,064 | 136.447 | 237 | 403 | 537 | 236 |
| **Right Arena Border** | 42,672 | 158.546 | 1195 | 509 | 254 | 168 |
| **Central Open Area** | 1,442,392 | 147.010 | 357 | 1 | 1204 | 1198 |

***
Locomotion Analysis

Locomotion parameters were calculated by converting the pixel coordinates into centimeters using the pixel-to-cm conversion value (0.032413). The parameters that were evaluated are mathematically and operationally tabulated below:

| Metric Name | Formula | Units | Description |
| :--- | :--- | :--- | :--- |
| **Total Frames** | $N$ | frames | Number of frames in video |
| **Total Time** | $\frac{N}{\text{fps}}$ | seconds | Duration of video |
| **Total Distance** | $\sum_{i=1}^{N-1} \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}$ | cm | Cumulative path length traveled |
| **Mean Speed** | $\frac{1}{N-1} \sum_{i=1}^{N-1} \frac{\sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}}{\text{frame duration}}$ | cm/s | Speed averaged over all frames |
| **Max Speed** | $\max \left( \frac{\sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}}{\text{frame duration}} \right)$ | cm/s | Maximum instantaneous frame-to-frame speed |
| **Net Displacement** | $\sqrt{(x_{\text{final}} - x_{\text{initial}})^2 + (y_{\text{final}} - y_{\text{initial}})^2}$ | cm | Straight-line distance start to end |
| **Straightness Index** | $\frac{\text{net displacement}}{\text{total distance}}$ | ratio | Path efficiency (0–1); higher is straighter |

### Arena Quadrant Division

The arena is divided into an X-shaped configuration based on angular thresholds calculated from the central coordinate $(X_{\text{center}}, Y_{\text{center}})$. 


| Quadrant | Angular Range | Description |
| :--- | :--- | :--- |
| **TOP** | $45^\circ < \theta \leq 135^\circ$ | Upper quadrant section |
| **LEFT** | $135^\circ < \theta \leq 225^\circ$ | Left quadrant section |
| **BOTTOM** | $225^\circ < \theta \leq 315^\circ$ | Lower quadrant section |
| **RIGHT** | $315^\circ < \theta \leq 45^\circ$ (or $\geq 0^\circ \cup \leq 45^\circ$) | Right quadrant section |

*Key Parameters:*
* **Arena Diameter:** $59\text{ cm}$
* **Origin Point:** $\text{Center } (X_{\text{center}}, Y_{\text{center}})$


#### Video Frame Parameters
* **Frame rate:** `2.0 FPS`
* **Target Run Duration:** `600.50 seconds` (10.01 minutes)
* **Standard Frame Allocation:** `1201 frames` is verified as accurate (`100.08%` ratio)

#### Tracking Bodypart Likelihood Profile (DLC ResNet50)
* **Confidence/ Likelihood Threshold :** `0.6` (Frames below threshold reduce Total Valid Time)
* **`body_end`**: Mean `0.753` | 7.1% failure rate (*Recommended baseline tracking target*)
* **`left_antennae_tip`**: Mean `0.304` | 100.0% tracking failure rate (*Do not utilize*)
* **`right_antennae_tip`**: Mean `0.361` | 99.9% tracking failure rate (*Do not utilize*)

***

### DeepLabCut PyTorch & Cluster Troubleshooting

#### 1. Out of Memory / Large Video File Frame Analysis Failures
* **Issue**: Error when processing large inference sequences exceeding 300,000 frames using legacy TensorFlow backends.
* **Resolution**: Force DeepLabCut engine execution within PyTorch native deployment wrappers. Pure TensorFlow allocations fail to scale over long frame periods. Reference engine verification context via [DeepLabCut Issue #2644](https://github.com).
* Always train on Cluster with GPU! 
* DLC models - pytorch works, the other dlc models folder is invalid 
* dlc changed to pytorch from tensor flow and the dlc models folder is an empty folder, thence. 


#### 2. Tracking Quality Drop & Decreased "Total Valid Time"
* **Issue**: The script calculates a `Total Valid Time` lower than your standard video window duration of `600.5 seconds`.
* **Resolution**: Verify your confidence filtering configuration thresholds. If data frames possess an evaluation likelihood score lower than `< 0.6`, they drop out of tracking runtime limits. 
* **Actionable Correction**:
  * Swap unstable tracking nodes (`left_antennae_tip` at 100% failure rate / `right_antennae_tip` at 99.9% failure rate) out of distance-mapping arrays.
  * Point your tracking pipeline execution loops exclusively at the stable `body_end` node (Mean `0.753` likelihood with only a `7.1%` threshold failure rate).

#### 3. PyTorch CUDA / GPU Initialization Mismatch on Cluster
* **Issue**: Python defaults to CPU execution inside tmux or fails to fetch active GPU nodes.
* **Resolution**: Always ensure environmental shell variables and paths load accurately, and in proper sequence, prior to execution. Run these explicit module reset lines in Tmux
  ```bash
  module purge
  module load python3/anaconda3
  source /rhomes/kkesavan/miniforge3/etc/profile.d/conda.sh
  conda activate /rhomes/kkesavan/miniforge3/envs/dlc_pytorch_gpu
  ```



cluster analysis issue from sep 2025 when DLC updated to pytorch from tensorflow: <https://github.com/DeepLabCut/DeepLabCut/issues/2644 - Pytorch with dlc pytorch only tensorflow cannot analyze all 300000 frames #!/bin/bash>


Date: 2026-05-31
Windows (Alienware)

- OS: Windows, GPU: NVIDIA RTX 2080 Ti (11GB VRAM)
- Miniconda installed, conda 26.3.2
- Environment: DEEPLABCUT (Python 3.10)
- Packages: pytables==3.8.0, torch/torchvision (CUDA 12.4), deeplabcut[gui] 3.0.0
- torch.cuda.is_available() prints True


to open gui: python -c "import deeplabcut; deeplabcut.launch_dlc()"

python  import deeplabcut
  deeplabcut.launch_dlc()
Cluster (agmn-srv-1)

- OS: Ubuntu 24.04, login node
- Miniconda at /rhomes/kkesavan/miniconda3/
- Environment: DLC_OpenArena (Python 3.10)
- Packages: pytables==3.8.0, torch/torchvision (CUDA 12.4), deeplabcut 3.0.0 (headless)
- torch.cuda.is_available() → True
- Passwordless SSH exists now
- Training targets: agmn-srv-3 / agmn-srv-4 (A6000 GPUs) via SLURM
- Cluster access through VScode, separate open arena environment exists installed with Remote SSh
  extension

How to: 
##### Activate environment for windows
conda activate DEEPLABCUT

##### Activate environment on Cluster
source ~/miniconda3/bin/activate
conda activate DLC_OpenArena

---
##### Cluster Directory 

/rhomes/kkesavan/
├── miniconda3/                          # conda installation (shared across projects)
├── Open_Arena_Jun26_Shyness_Boldness/   # current project
└── DLC_Open_Arena_Aug25_OdourPref/      # archived old project
    ├── dlc_project/
    ├── dlc_project_group/
    └── conda-envs/
- Old miniforge3 (~30GB) deleted from archive to free disk space



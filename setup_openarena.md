Date: 2026-05-31
Windows (Alienware)

OS: Windows, GPU: NVIDIA RTX 2080 Ti (11GB VRAM)
Miniconda installed, conda 26.3.2
Environment: DEEPLABCUT (Python 3.10)
Packages: pytables==3.8.0, torch/torchvision (CUDA 12.4), deeplabcut[gui] 3.0.0
torch.cuda.is_available() prints True


to open gui: python -c "import deeplabcut; deeplabcut.launch_dlc()"

python  import deeplabcut
  deeplabcut.launch_dlc()
Cluster (agmn-srv-1)

OS: Ubuntu 24.04, login node
Miniconda at /rhomes/kkesavan/miniconda3/
Environment: DLC_OpenArena (Python 3.10)
Packages: pytables==3.8.0, torch/torchvision (CUDA 12.4), deeplabcut 3.0.0 (headless)
torch.cuda.is_available() → True
Passwordless SSH exists now
Training targets: agmn-srv-3 / agmn-srv-4 (A6000 GPUs) via SLURM

Cluster access through VScode, separate open arena environment exists 

How to: 
Activate environments
bash# Windows (VS Code terminal)
conda activate DEEPLABCUT

# Cluster
source ~/miniconda3/bin/activate
conda activate DLC_OpenArena



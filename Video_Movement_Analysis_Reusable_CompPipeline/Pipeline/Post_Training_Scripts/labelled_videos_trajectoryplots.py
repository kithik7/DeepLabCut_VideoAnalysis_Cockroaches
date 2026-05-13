#!/bin/bash
#SBATCH --job-name=dlc_viz_no_crop
#SBATCH --output=dlc_viz_no_crop.out
#SBATCH --error=dlc_viz_no_crop.err
#SBATCH --partition=gpu_a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

module load python3/anaconda3
source /rhomes/kkesavan/miniforge3/etc/profile.d/conda.sh
conda activate /rhomes/kkesavan/miniforge3/envs/dlc_pytorch_gpu

python -c "
import deeplabcut
import glob
import os

print('=== CREATING LABELED VIDEOS (NO CROPPING) ===')
video_files = glob.glob('./videos/**/*.mp4', recursive=True)
video_files = [os.path.abspath(v) for v in video_files]
print(f'Found {len(video_files)} videos')

success_count = 0
for i, video_path in enumerate(video_files, 1):
    print(f'\\n[{i}/{len(video_files)}] Creating labeled video: {os.path.basename(video_path)}')
    
    try:
        deeplabcut.create_labeled_video(
            config='config_cluster.yaml',
            videos=[video_path],
            shuffle=1,
            trainingsetindex=0,
            filtered=False,
            displaycropped=False,  # NO CROPPING - FULL FRAME
            draw_skeleton=True,
            trailpoints=30,
            dotsize=10,
            colormap='rainbow',
            alphavalue=0.7,
            fastmode=True,
            save_frames=False
        )
        success_count += 1
        print('  ✅ Labeled video created (full frame)')
    except Exception as e:
        print(f'  ❌ Failed: {e}')

print(f'\\n✅ Labeled Videos: {success_count}/{len(video_files)}')

print('\\n=== CREATING TRAJECTORY PLOTS ===')
plot_success = 0
for i, video_path in enumerate(video_files, 1):
    print(f'[{i}/{len(video_files)}] Creating trajectory plot: {os.path.basename(video_path)}')
    try:
        deeplabcut.plot_trajectories(
            'config_cluster.yaml',
            [video_path],
            shuffle=1,
            trainingsetindex=0,
            filtered=False,
            showfigures=False
        )
        plot_success += 1
        print('  ✅ Trajectory plot created')
    except Exception as e:
        print(f'  ❌ Plot failed: {e}')

print(f'\\n🎉 VISUALIZATION COMPLETE!')
print(f'   - Labeled Videos: {success_count}/{len(video_files)}')
print(f'   - Trajectory Plots: {plot_success}/{len(video_files)}')
"

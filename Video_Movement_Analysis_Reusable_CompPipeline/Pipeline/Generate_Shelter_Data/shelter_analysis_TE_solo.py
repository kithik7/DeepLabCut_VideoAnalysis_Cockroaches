#!/bin/bash
#SBATCH --job-name=zone_analysis_fixed
#SBATCH --output=zone_analysis_fixed.out
#SBATCH --error=zone_analysis_fixed.err
#SBATCH --partition=gpu_a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

module load python3/anaconda3
source /rhomes/kkesavan/miniforge3/etc/profile.d/conda.sh
conda activate /rhomes/kkesavan/miniforge3/envs/dlc_pytorch_gpu

python -c "
import pandas as pd
import numpy as np
import glob
import os



# Only shelter definitions - open area is everything else
ZONES = {
    'left_shelter': {'x_range': (403, 639), 'y_range': (537, 711)},
    'right_shelter': {'x_range': (1195, 1449), 'y_range': (509, 677)}
}

print('Zone coordinates:')
for zone, coords in ZONES.items():
    print(f'  {zone}: X{coords[\"x_range\"]}, Y{coords[\"y_range\"]}')

def point_in_zone(x, y, zone):
    x_min, x_max = ZONES[zone]['x_range']
    y_min, y_max = ZONES[zone]['y_range']
    return x_min <= x <= x_max and y_min <= y <= y_max

def analyze_video_zone_preference(csv_file, fps=2.0):
    try:
        print(f'    Reading CSV: {os.path.basename(csv_file)}')
        # Read DeepLabCut CSV file
        df = pd.read_csv(csv_file, header=[0, 1, 2])
        
        # Look for body_end tracking point
        body_end_cols = [col for col in df.columns if 'body_end' in str(col[1]).lower()]
        
        if not body_end_cols:
            print(f'    Using thorax boundaries as fallback')
            left_thorax_cols = [col for col in df.columns if 'left_pmthorax_boundary' in str(col[1]).lower() and col[2] == 'x']
            right_thorax_cols = [col for col in df.columns if 'right_pmthorax_boundary' in str(col[1]).lower() and col[2] == 'x']
            
            if left_thorax_cols and right_thorax_cols:
                left_x = df[left_thorax_cols[0]].values
                left_y = df[(left_thorax_cols[0][0], left_thorax_cols[0][1], 'y')].values
                right_x = df[right_thorax_cols[0]].values
                right_y = df[(right_thorax_cols[0][0], right_thorax_cols[0][1], 'y')].values
                
                x_coords = (left_x + right_x) / 2
                y_coords = (left_y + right_y) / 2
                likelihood = np.minimum(
                    df[(left_thorax_cols[0][0], left_thorax_cols[0][1], 'likelihood')].values,
                    df[(right_thorax_cols[0][0], right_thorax_cols[0][1], 'likelihood')].values
                )
            else:
                return None
        else:
            x_col = body_end_cols[0]
            y_col = (x_col[0], x_col[1], 'y')
            likelihood_col = (x_col[0], x_col[1], 'likelihood')
            
            x_coords = df[x_col].values
            y_coords = df[y_col].values
            likelihood = df[likelihood_col].values
        
        # Filter by likelihood
        confidence_threshold = 0.6
        valid_frames = likelihood >= confidence_threshold
        
        # Calculate time in each zone
        total_frames = len(x_coords)
        frame_duration = 1.0 / fps
        
        time_in_zones = {}
        for zone_name in ZONES.keys():
            zone_frames = 0
            for i in range(total_frames):
                if valid_frames[i]:
                    if point_in_zone(x_coords[i], y_coords[i], zone_name):
                        zone_frames += 1
            time_in_zones[zone_name] = zone_frames * frame_duration
        
        # Calculate total shelter time and open area time
        total_valid_frames = np.sum(valid_frames)
        total_valid_time = total_valid_frames * frame_duration
        
        total_shelter_time = sum(time_in_zones.values())
        time_open_area = total_valid_time - total_shelter_time
        
        # Calculate percentages
        if total_valid_time > 0:
            percent_left = (time_in_zones['left_shelter'] / total_valid_time) * 100
            percent_right = (time_in_zones['right_shelter'] / total_valid_time) * 100
            percent_open = (time_open_area / total_valid_time) * 100
        else:
            percent_left = percent_right = percent_open = 0
        
        # Determine preferred zone
        left_time = time_in_zones['left_shelter']
        right_time = time_in_zones['right_shelter']
        
        if left_time > right_time:
            preferred_zone = 'left_shelter'
        elif right_time > left_time:
            preferred_zone = 'right_shelter'
        else:
            preferred_zone = 'open_area'
        
        return {
            'time_left_shelter': time_in_zones['left_shelter'],
            'time_right_shelter': time_in_zones['right_shelter'],
            'time_open_area': time_open_area,
            'percent_left': percent_left,
            'percent_right': percent_right,
            'percent_open': percent_open,
            'preferred_zone': preferred_zone,
            'total_valid_time': total_valid_time,
            'total_valid_frames': total_valid_frames,
            'total_frames': total_frames,
            'tracking_point': 'body_end' if body_end_cols else 'thorax_midpoint'
        }
    
    except Exception as e:
        print(f'    Error analyzing CSV: {e}')
        return None

# Load experimental design
exp_data = pd.read_csv('solo_part_1.csv')
print(f'Loaded data for {len(exp_data)} experiments')

# Process each video in original order
results = []

for idx, row in exp_data.iterrows():
    cockroach_id = row['cockroach_id']
    video_name = row['video']
    print(f'\\n[{idx+1}/{len(exp_data)}] Analyzing: {cockroach_id}')
    print(f'  Video: {video_name}')
    
    # Convert H5 filename to actual CSV filename pattern
    csv_pattern = video_name.replace('snapshot_020.h5', 'snapshot_200.csv')\
                           .replace('snapshot_best-110.h5', 'snapshot_200.csv')
    
    # Search in exact folder structure
    search_paths = [
        f'videos/August_2025/*/*/{csv_pattern}',
        f'videos/August_2025/*/{csv_pattern}',
        f'videos/*/{csv_pattern}',
        f'videos/{csv_pattern}'
    ]
    
    csv_file = None
    for path in search_paths:
        files = glob.glob(path)
        if files:
            csv_file = files[0]
            break
    
    if not csv_file:
        print(f'   CSV file not found for pattern: {csv_pattern}')
        # Try broader search with just the base name
        base_name = video_name.replace('DLC_Resnet50_single_k7Sep2shuffle1_snapshot_020.h5', '')\
                              .replace('DLC_Resnet50_single_k7Sep2shuffle1_snapshot_best-110.h5', '')
        broader_search = f'videos/**/*{base_name}*snapshot_200.csv'
        files = glob.glob(broader_search, recursive=True)
        if files:
            csv_file = files[0]
            print(f'  Found via broader search: {os.path.basename(csv_file)}')
    
    if not csv_file:
        print(f'  CSV file not found')
        result = {
            'date': row['date'],
            'group': row['group'],
            'cockroach_id': row['cockroach_id'],
            'video': row['video'],
            'experiment_type': row['experiment_type'],
            'left_stimulus': row['left_stimulus'],
            'right_stimulus': row['right_stimulus'],
            'time_left_shelter': None,
            'time_right_shelter': None,
            'time_open_area': None,
            'percent_left': None,
            'percent_right': None,
            'percent_open': None,
            'preferred_zone': None,
            'preferred_stimulus': None,
            'total_valid_time': None,
            'valid_frames': None,
            'total_frames': None,
            'tracking_point': None,
            'error': 'CSV file not found'
        }
    else:
        print(f'  Found: {csv_file}')
        analysis = analyze_video_zone_preference(csv_file, fps=2.0)
        
        if analysis:
            # Determine preferred stimulus
            preferred_zone = analysis['preferred_zone']
            if preferred_zone == 'left_shelter':
                preferred_stimulus = row['left_stimulus']
            elif preferred_zone == 'right_shelter':
                preferred_stimulus = row['right_stimulus']
            else:
                preferred_stimulus = 'open_area'
            
            result = {
                'date': row['date'],
                'group': row['group'],
                'cockroach_id': row['cockroach_id'],
                'video': row['video'],
                'experiment_type': row['experiment_type'],
                'left_stimulus': row['left_stimulus'],
                'right_stimulus': row['right_stimulus'],
                'time_left_shelter': analysis['time_left_shelter'],
                'time_right_shelter': analysis['time_right_shelter'],
                'time_open_area': analysis['time_open_area'],
                'percent_left': analysis['percent_left'],
                'percent_right': analysis['percent_right'],
                'percent_open': analysis['percent_open'],
                'preferred_zone': preferred_zone,
                'preferred_stimulus': preferred_stimulus,
                'total_valid_time': analysis['total_valid_time'],
                'valid_frames': analysis['total_valid_frames'],
                'total_frames': analysis['total_frames'],
                'tracking_point': analysis['tracking_point'],
                'error': None
            }
            print(f'  Preferred {preferred_zone} (stimulus: {preferred_stimulus})')
            print(f'  Tracking: {analysis[\"total_valid_frames\"]}/{analysis[\"total_frames\"]} frames ({analysis[\"total_valid_time\"]:.1f}s valid)')
            print(f'  Times - Left: {analysis[\"time_left_shelter\"]:.1f}s, Right: {analysis[\"time_right_shelter\"]:.1f}s, Open: {analysis[\"time_open_area\"]:.1f}s')
        else:
            result = {
                'date': row['date'],
                'group': row['group'],
                'cockroach_id': row['cockroach_id'],
                'video': row['video'],
                'experiment_type': row['experiment_type'],
                'left_stimulus': row['left_stimulus'],
                'right_stimulus': row['right_stimulus'],
                'time_left_shelter': None,
                'time_right_shelter': None,
                'time_open_area': None,
                'percent_left': None,
                'percent_right': None,
                'percent_open': None,
                'preferred_zone': None,
                'preferred_stimulus': None,
                'total_valid_time': None,
                'valid_frames': None,
                'total_frames': None,
                'tracking_point': None,
                'error': 'Analysis failed'
            }
            print(f'   Analysis failed')
    
    results.append(result)

# Save results
results_df = pd.DataFrame(results)
output_file = 'shelter_preference_analysis_corrected.csv'
results_df.to_csv(output_file, index=False)

successful = len(results_df[results_df['error'].isna()])
print(f'\\n Job Done')
print(f' Saved to: {output_file}')

"

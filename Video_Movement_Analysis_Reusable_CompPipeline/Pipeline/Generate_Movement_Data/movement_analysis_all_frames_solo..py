#!/bin/bash
#SBATCH --job-name=movement_analysis_all_frames
#SBATCH --output=movement_analysis_all_frames.out
#SBATCH --error=movement_analysis_all_frames.err
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
from scipy import stats

print('Comprehensive movement analysis (all frames)')

# Constants for speed calculation
FPS = 2.0  # frames per second <adjust according to your setup>
PIXELS_PER_CM = 28.0  #adjust according to your setup
TOTAL_VIDEO_LENGTH = 600.5  # seconds

# Shelter definitions
ZONES = {
    'left_shelter': {'x_range': (403, 639), 'y_range': (537, 711)},
    'right_shelter': {'x_range': (1195, 1449), 'y_range': (509, 677)}
}

def point_in_zone(x, y, zone):
    x_min, x_max = ZONES[zone]['x_range']
    y_min, y_max = ZONES[zone]['y_range']
    return x_min <= x <= x_max and y_min <= y <= y_max

def calculate_movement_metrics(x_coords, y_coords, fps=FPS, pixels_per_cm=PIXELS_PER_CM):
    \"\"\"Calculate speed, velocity, and movement patterns\"\"\"
    if len(x_coords) < 2:
        return None
    
    # Remove any NaN values
    valid_indices = ~(np.isnan(x_coords) | np.isnan(y_coords))
    x_clean = x_coords[valid_indices]
    y_clean = y_coords[valid_indices]
    
    if len(x_clean) < 2:
        return None
    
    # Calculate displacements between frames
    dx = np.diff(x_clean)  # x displacement (pixels/frame)
    dy = np.diff(y_clean)  # y displacement (pixels/frame)
    
    # Distance traveled per frame (pixels/frame)
    distances_pixels = np.sqrt(dx**2 + dy**2)
    
    # Convert to cm/s
    frame_interval = 1.0 / fps
    speeds_cm_s = (distances_pixels / pixels_per_cm) / frame_interval
    
    # Velocity components (cm/s)
    velocity_x = (dx / pixels_per_cm) / frame_interval
    velocity_y = (dy / pixels_per_cm) / frame_interval
    
    # Total displacement from start to end
    total_displacement = np.sqrt((x_clean[-1] - x_clean[0])**2 + (y_clean[-1] - y_clean[0])**2) / pixels_per_cm
    
    # Total path length (distance traveled)
    total_path_length = np.sum(distances_pixels) / pixels_per_cm
    
    # Movement metrics
    movement_data = {
        'mean_speed_cm_s': np.mean(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'max_speed_cm_s': np.max(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'std_speed_cm_s': np.std(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'median_speed_cm_s': np.median(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'total_distance_cm': total_path_length,
        'net_displacement_cm': total_displacement,
        'straightness_index': total_displacement / total_path_length if total_path_length > 0 else 0,
        'mean_velocity_x': np.mean(velocity_x) if len(velocity_x) > 0 else 0,
        'mean_velocity_y': np.mean(velocity_y) if len(velocity_y) > 0 else 0,
        'movement_frames': len(speeds_cm_s),
        'stationary_frames': np.sum(speeds_cm_s < 0.1) if len(speeds_cm_s) > 0 else 0,  # < 0.1 cm/s = stationary
        'active_frames': np.sum(speeds_cm_s >= 0.1) if len(speeds_cm_s) > 0 else 0,
    }
    
    return movement_data, speeds_cm_s, velocity_x, velocity_y

def analyze_video_comprehensive(csv_file):
    \"\"\"Analyze both zone preference and movement metrics\"\"\"
    try:
        print(f'    Reading CSV: {os.path.basename(csv_file)}')
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
            else:
                return None, None
        else:
            x_col = body_end_cols[0]
            y_col = (x_col[0], x_col[1], 'y')
            
            x_coords = df[x_col].values
            y_coords = df[y_col].values
        
        # preference for zones
        total_frames = len(x_coords)
        frame_duration = 1.0 / FPS
        
        time_in_zones = {}
        zone_frames = {zone: [] for zone in ZONES.keys()}
        
        for zone_name in ZONES.keys():
            zone_frames_count = 0
            for i in range(total_frames):
                if point_in_zone(x_coords[i], y_coords[i], zone_name):
                    zone_frames_count += 1
                    zone_frames[zone_name].append(i)
            time_in_zones[zone_name] = zone_frames_count * frame_duration
        
        total_shelter_time = sum(time_in_zones.values())
        time_open_area = TOTAL_VIDEO_LENGTH - total_shelter_time
        
        percent_left = (time_in_zones['left_shelter'] / TOTAL_VIDEO_LENGTH) * 100
        percent_right = (time_in_zones['right_shelter'] / TOTAL_VIDEO_LENGTH) * 100
        percent_open = (time_open_area / TOTAL_VIDEO_LENGTH) * 100
        
        left_time = time_in_zones['left_shelter']
        right_time = time_in_zones['right_shelter']
        
        if left_time > right_time:
            preferred_zone = 'left_shelter'
        elif right_time > left_time:
            preferred_zone = 'right_shelter'
        else:
            preferred_zone = 'open_area'
        
        zone_preference = {
            'time_left_shelter': time_in_zones['left_shelter'],
            'time_right_shelter': time_in_zones['right_shelter'],
            'time_open_area': time_open_area,
            'percent_left': percent_left,
            'percent_right': percent_right,
            'percent_open': percent_open,
            'preferred_zone': preferred_zone,
            'total_video_time': TOTAL_VIDEO_LENGTH,
            'total_frames': total_frames,
            'tracking_point': 'body_end' if body_end_cols else 'thorax_midpoint'
        }
        
        # movement analysis
        movement_results = calculate_movement_metrics(x_coords, y_coords)
        
        if movement_results:
            movement_data, all_speeds, all_velocity_x, all_velocity_y = movement_results
            
            # Calculate speed in each zone
            zone_speeds = {}
            for zone_name, frame_indices in zone_frames.items():
                if len(frame_indices) > 1:
                    # Get speeds when in this zone (need consecutive frames in same zone)
                    zone_speed_values = []
                    for i in range(1, len(frame_indices)):
                        if frame_indices[i] - frame_indices[i-1] == 1:  # consecutive frames
                            idx = frame_indices[i] - 1  # -1 because speeds array is one shorter
                            if idx < len(all_speeds):
                                zone_speed_values.append(all_speeds[idx])
                    
                    if zone_speed_values:
                        zone_speeds[f'mean_speed_{zone_name}'] = np.mean(zone_speed_values)
                        zone_speeds[f'max_speed_{zone_name}'] = np.max(zone_speed_values)
                    else:
                        zone_speeds[f'mean_speed_{zone_name}'] = 0
                        zone_speeds[f'max_speed_{zone_name}'] = 0
                else:
                    zone_speeds[f'mean_speed_{zone_name}'] = 0
                    zone_speeds[f'max_speed_{zone_name}'] = 0
            
            # Calculate open area speed (frames not in any shelter)
            open_frames = [i for i in range(total_frames) 
                          if not any(point_in_zone(x_coords[i], y_coords[i], zone) for zone in ZONES.keys())]
            
            open_speed_values = []
            for i in range(1, len(open_frames)):
                if open_frames[i] - open_frames[i-1] == 1:  # consecutive frames
                    idx = open_frames[i] - 1
                    if idx < len(all_speeds):
                        open_speed_values.append(all_speeds[idx])
            
            if open_speed_values:
                zone_speeds['mean_speed_open_area'] = np.mean(open_speed_values)
                zone_speeds['max_speed_open_area'] = np.max(open_speed_values)
            else:
                zone_speeds['mean_speed_open_area'] = 0
                zone_speeds['max_speed_open_area'] = 0
            
            # Combine all movement data
            comprehensive_movement = {**movement_data, **zone_speeds}
            
        else:
            comprehensive_movement = None
        
        return zone_preference, comprehensive_movement
    
    except Exception as e:
        print(f'    Error analyzing CSV: {e}')
        return None, None

# Load experimental design
exp_data = pd.read_csv('solo_part_1.csv')
print(f'Loaded data for {len(exp_data)} experiments')

# Process each video
preference_results = []
movement_results = []

for idx, row in exp_data.iterrows():
    cockroach_id = row['cockroach_id']
    video_name = row['video']
    print(f'\\n[{idx+1}/{len(exp_data)}] Analyzing: {cockroach_id}')
    print(f'  Video: {video_name}')
    
    # Convert H5 filename to CSV filename pattern
    csv_pattern = video_name.replace('snapshot_020.h5', 'snapshot_200.csv')\
                           .replace('snapshot_best-110.h5', 'snapshot_200.csv')
    
    # Search for CSV file
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
        # Try broader search
        base_name = video_name.replace('DLC_Resnet50_single_k7Sep2shuffle1_snapshot_020.h5', '')\
                              .replace('DLC_Resnet50_single_k7Sep2shuffle1_snapshot_best-110.h5', '')
        broader_search = f'videos/**/*{base_name}*snapshot_200.csv'
        files = glob.glob(broader_search, recursive=True)
        if files:
            csv_file = files[0]
            print(f'  Found via broader search: {os.path.basename(csv_file)}')
    
    if not csv_file:
        print(f'  CSV file not found')
        # Create empty results for failed analyses
        pref_result = {
            'date': row['date'], 'group': row['group'], 'cockroach_id': row['cockroach_id'],
            'video': row['video'], 'experiment_type': row['experiment_type'],
            'left_stimulus': row['left_stimulus'], 'right_stimulus': row['right_stimulus'],
            'time_left_shelter': None, 'time_right_shelter': None, 'time_open_area': None,
            'percent_left': None, 'percent_right': None, 'percent_open': None,
            'preferred_zone': None, 'preferred_stimulus': None,
            'total_video_time': None, 'total_frames': None, 'tracking_point': None,
            'error': 'CSV file not found'
        }
        move_result = {
            'date': row['date'], 'group': row['group'], 'cockroach_id': row['cockroach_id'],
            'video': row['video'], 'experiment_type': row['experiment_type'],
            'left_stimulus': row['left_stimulus'], 'right_stimulus': row['right_stimulus'],
            'error': 'CSV file not found'
        }
    else:
        print(f'   Found: {csv_file}')
        zone_pref, movement_data = analyze_video_comprehensive(csv_file)
        
        if zone_pref:
            # Determine preferred stimulus
            preferred_zone = zone_pref['preferred_zone']
            if preferred_zone == 'left_shelter':
                preferred_stimulus = row['left_stimulus']
            elif preferred_zone == 'right_shelter':
                preferred_stimulus = row['right_stimulus']
            else:
                preferred_stimulus = 'open_area'
            
            pref_result = {
                'date': row['date'], 'group': row['group'], 'cockroach_id': row['cockroach_id'],
                'video': row['video'], 'experiment_type': row['experiment_type'],
                'left_stimulus': row['left_stimulus'], 'right_stimulus': row['right_stimulus'],
                'time_left_shelter': zone_pref['time_left_shelter'],
                'time_right_shelter': zone_pref['time_right_shelter'],
                'time_open_area': zone_pref['time_open_area'],
                'percent_left': zone_pref['percent_left'],
                'percent_right': zone_pref['percent_right'],
                'percent_open': zone_pref['percent_open'],
                'preferred_zone': preferred_zone,
                'preferred_stimulus': preferred_stimulus,
                'total_video_time': zone_pref['total_video_time'],
                'total_frames': zone_pref['total_frames'],
                'tracking_point': zone_pref['tracking_point'],
                'error': None
            }
            print(f'   Preferred {preferred_zone} (stimulus: {preferred_stimulus})')
            
            if movement_data:
                move_result = {
                    'date': row['date'], 'group': row['group'], 'cockroach_id': row['cockroach_id'],
                    'video': row['video'], 'experiment_type': row['experiment_type'],
                    'left_stimulus': row['left_stimulus'], 'right_stimulus': row['right_stimulus'],
                    **movement_data,
                    'error': None
                }
                print(f'   Movement: {movement_data[\"mean_speed_cm_s\"]:.2f} cm/s mean speed')
            else:
                move_result = {
                    'date': row['date'], 'group': row['group'], 'cockroach_id': row['cockroach_id'],
                    'video': row['video'], 'experiment_type': row['experiment_type'],
                    'left_stimulus': row['left_stimulus'], 'right_stimulus': row['right_stimulus'],
                    'error': 'Movement analysis failed'
                }
        else:
            pref_result = {
                'date': row['date'], 'group': row['group'], 'cockroach_id': row['cockroach_id'],
                'video': row['video'], 'experiment_type': row['experiment_type'],
                'left_stimulus': row['left_stimulus'], 'right_stimulus': row['right_stimulus'],
                'error': 'Analysis failed'
            }
            move_result = {
                'date': row['date'], 'group': row['group'], 'cockroach_id': row['cockroach_id'],
                'video': row['video'], 'experiment_type': row['experiment_type'],
                'left_stimulus': row['left_stimulus'], 'right_stimulus': row['right_stimulus'],
                'error': 'Analysis failed'
            }
            print(f'   Analysis failed')
    
    preference_results.append(pref_result)
    movement_results.append(move_result)

# Save both result files
pref_df = pd.DataFrame(preference_results)
move_df = pd.DataFrame(movement_results)

pref_output = 'shelter_preference_all_frames.csv'
move_output = 'movement_analysis_all_frames.csv'

pref_df.to_csv(pref_output, index=False)
move_df.to_csv(move_output, index=False)

successful_pref = len(pref_df[pref_df['error'].isna()])
successful_move = len(move_df[move_df['error'].isna()])

print(f'\\n job done!')
print(f' Shelter preference saved to: {pref_output}')
print(f' Movement analysis saved to: {move_output}')
print(f' Successful preferences: {successful_pref}/{len(pref_df)}')
print(f' Successful movement: {successful_move}/{len(move_df)}')

# Print summary statistics
if successful_move > 0:
    successful_moves = move_df[move_df['error'].isna()]
    print(f'\\n mov summary:')
    print(f'   Mean speed: {successful_moves[\"mean_speed_cm_s\"].mean():.2f} ± {successful_moves[\"mean_speed_cm_s\"].std():.2f} cm/s')
    print(f'   Max speed: {successful_moves[\"max_speed_cm_s\"].mean():.2f} ± {successful_moves[\"max_speed_cm_s\"].std():.2f} cm/s')
    print(f'   Total distance: {successful_moves[\"total_distance_cm\"].mean():.1f} ± {successful_moves[\"total_distance_cm\"].std():.1f} cm')
    print(f'   Straightness: {successful_moves[\"straightness_index\"].mean():.3f} ± {successful_moves[\"straightness_index\"].std():.3f}')
"

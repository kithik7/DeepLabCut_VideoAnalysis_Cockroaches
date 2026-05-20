#!/bin/bash
#SBATCH --job-name=group_movement_tracking
#SBATCH --output=group_movement_tracking.out
#SBATCH --error=group_movement_tracking.err
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
import re
from scipy import stats

print('group movement analysis with tracking efficiency')

# Experimental design from CSV hardcoded in
EXPERIMENT_DESIGN = {
    '2025-08-20': {'batch': 2, 'type': 'Iso/Nothing', 'left': 'Iso', 'right': 'Nothing'},
    '2025-08-21': {'batch': 1, 'type': 'Iso/Nothing', 'left': 'Iso', 'right': 'Nothing'},
    '2025-08-22': {'batch': 4, 'type': 'Feces/Nothing', 'left': 'Feces', 'right': 'Nothing'},
    '2025-08-23': {'batch': 3, 'type': 'Feces/Nothing', 'left': 'Nothing', 'right': 'Feces'},
    '2025-08-26': {'batch': 5, 'type': 'Iso/Feces', 'left': 'Iso', 'right': 'Feces'},
    '2025-08-27': {'batch': 6, 'type': 'Iso/Feces', 'left': 'Feces', 'right': 'Iso'},
    '2025-08-28': {'batch': 7, 'type': 'Iso/Nothing', 'left': 'Iso', 'right': 'Nothing'},
    '2025-08-29': {'batch': 8, 'type': 'Iso/Nothing', 'left': 'Iso', 'right': 'Nothing'},
    '2025-08-30': {'batch': 9, 'type': 'Feces/Nothing', 'left': 'Nothing', 'right': 'Feces'},
    '2025-08-31': {'batch': 10, 'type': 'Feces/Nothing', 'left': 'Feces', 'right': 'Nothing'}
}

# Constants for movement analysis
FPS = 2.0
PIXELS_PER_CM = 28.0
TOTAL_VIDEO_LENGTH = 600.5

# Shelter definitions
ZONES = {
    'left_shelter': {'x_range': (403, 639), 'y_range': (537, 711)},
    'right_shelter': {'x_range': (1195, 1449), 'y_range': (509, 677)}
}

def extract_date_from_filename(filename):
    \"\"\"Extract date from video filename\"\"\"
    date_pattern = r'202508(\d{2})'
    match = re.search(date_pattern, filename)
    if match:
        day = match.group(1)
        return f'2025-08-{day}'
    
    alt_pattern = r'2025-08-(\d{2})'
    match = re.search(alt_pattern, filename)
    if match:
        day = match.group(1)
        return f'2025-08-{day}'
    
    batch_match = re.search(r'[Bb]atch[_\s]*(\d+)', filename)
    if batch_match:
        batch_num = int(batch_match.group(1))
        for date, info in EXPERIMENT_DESIGN.items():
            if info['batch'] == batch_num:
                return date
    
    return None

def point_in_zone(x, y, zone):
    x_min, x_max = ZONES[zone]['x_range']
    y_min, y_max = ZONES[zone]['y_range']
    return x_min <= x <= x_max and y_min <= y <= y_max

def calculate_tracking_quality_metrics(likelihood, confidence_threshold=0.6):
    \"\"\"Calculate comprehensive tracking quality metrics\"\"\"
    valid_frames = likelihood >= confidence_threshold
    valid_frames_count = np.sum(valid_frames)
    total_frames = len(likelihood)
    tracking_efficiency = (valid_frames_count / total_frames) * 100
    
    # Calculate tracking segments (consecutive valid frames)
    segments = []
    current_segment = 0
    for i in range(len(valid_frames)):
        if valid_frames[i]:
            current_segment += 1
        else:
            if current_segment > 0:
                segments.append(current_segment)
                current_segment = 0
    if current_segment > 0:
        segments.append(current_segment)
    
    if segments:
        avg_segment_length = np.mean(segments)
        max_segment_length = np.max(segments)
        min_segment_length = np.min(segments)
    else:
        avg_segment_length = max_segment_length = min_segment_length = 0
    
    # Determine tracking quality category
    if tracking_efficiency >= 90:
        tracking_quality = 'High'
    elif tracking_efficiency >= 70:
        tracking_quality = 'Medium'
    elif tracking_efficiency >= 50:
        tracking_quality = 'Low'
    else:
        tracking_quality = 'Poor'
    
    return {
        'tracking_efficiency': tracking_efficiency,
        'avg_likelihood': np.mean(likelihood[valid_frames]) if np.sum(valid_frames) > 0 else 0,
        'valid_frames': valid_frames_count,
        'total_frames': total_frames,
        'percent_frames_analyzed': (valid_frames_count / total_frames) * 100,
        'avg_segment_length': avg_segment_length,
        'max_segment_length': max_segment_length,
        'min_segment_length': min_segment_length,
        'num_segments': len(segments),
        'tracking_quality': tracking_quality
    }

def calculate_movement_metrics_with_tracking(x_coords, y_coords, likelihood, fps=FPS, pixels_per_cm=PIXELS_PER_CM):
    \"\"\"Calculate movement metrics with tracking efficiency parameters\"\"\"
    if len(x_coords) < 2:
        return None
    
    # First calculate tracking quality metrics
    tracking_metrics = calculate_tracking_quality_metrics(likelihood)
    
    # Filter by likelihood for movement analysis
    confidence_threshold = 0.6
    valid_frames = likelihood >= confidence_threshold
    x_clean = x_coords[valid_frames]
    y_clean = y_coords[valid_frames]
    
    if len(x_clean) < 2:
        return {
            **tracking_metrics,
            'error': 'Insufficient valid frames for movement analysis'
        }
    
    # Calculate displacements
    dx = np.diff(x_clean)
    dy = np.diff(y_clean)
    
    # Distance traveled per frame
    distances_pixels = np.sqrt(dx**2 + dy**2)
    
    # Convert to cm/s
    frame_interval = 1.0 / fps
    speeds_cm_s = (distances_pixels / pixels_per_cm) / frame_interval
    
    # Velocity components
    velocity_x = (dx / pixels_per_cm) / frame_interval
    velocity_y = (dy / pixels_per_cm) / frame_interval
    
    # Total displacement and path length
    total_displacement = np.sqrt((x_clean[-1] - x_clean[0])**2 + (y_clean[-1] - y_clean[0])**2) / pixels_per_cm
    total_path_length = np.sum(distances_pixels) / pixels_per_cm
    
    # Movement classification
    stationary_threshold = 0.1
    slow_threshold = 1.0
    medium_threshold = 3.0
    
    stationary_frames = np.sum(speeds_cm_s < stationary_threshold) if len(speeds_cm_s) > 0 else 0
    slow_frames = np.sum((speeds_cm_s >= stationary_threshold) & (speeds_cm_s < slow_threshold))
    medium_frames = np.sum((speeds_cm_s >= slow_threshold) & (speeds_cm_s < medium_threshold))
    fast_frames = np.sum(speeds_cm_s >= medium_threshold)
    total_movement_frames = len(speeds_cm_s)
    
    # Activity ratio (proportion of time moving)
    activity_ratio = (total_movement_frames - stationary_frames) / total_movement_frames if total_movement_frames > 0 else 0
    
    # Movement metrics with tracking efficiency
    movement_data = {
        **tracking_metrics,
        
        # Basic speed metrics
        'mean_speed_cm_s': np.mean(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'max_speed_cm_s': np.max(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'min_speed_cm_s': np.min(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'median_speed_cm_s': np.median(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        'std_speed_cm_s': np.std(speeds_cm_s) if len(speeds_cm_s) > 0 else 0,
        
        # Distance metrics
        'total_distance_cm': total_path_length,
        'net_displacement_cm': total_displacement,
        'straightness_index': total_displacement / total_path_length if total_path_length > 0 else 0,
        'mean_step_length_cm': np.mean(distances_pixels) / pixels_per_cm if len(distances_pixels) > 0 else 0,
        
        # Velocity metrics
        'mean_velocity_x': np.mean(velocity_x) if len(velocity_x) > 0 else 0,
        'mean_velocity_y': np.mean(velocity_y) if len(velocity_y) > 0 else 0,
        'std_velocity_x': np.std(velocity_x) if len(velocity_x) > 0 else 0,
        'std_velocity_y': np.std(velocity_y) if len(velocity_y) > 0 else 0,
        
        # Movement classification
        'stationary_frames': stationary_frames,
        'slow_frames': slow_frames,
        'medium_frames': medium_frames,
        'fast_frames': fast_frames,
        'total_movement_frames': total_movement_frames,
        'percent_stationary': (stationary_frames / total_movement_frames * 100) if total_movement_frames > 0 else 0,
        'percent_slow': (slow_frames / total_movement_frames * 100) if total_movement_frames > 0 else 0,
        'percent_medium': (medium_frames / total_movement_frames * 100) if total_movement_frames > 0 else 0,
        'percent_fast': (fast_frames / total_movement_frames * 100) if total_movement_frames > 0 else 0,
        'activity_ratio': activity_ratio,
        
        'error': None
    }
    
    return movement_data

def analyze_group_movement_with_tracking(csv_file):
    \"\"\"Analyze movement with tracking efficiency for all animals in group\"\"\"
    try:
        print(f'    Reading CSV: {os.path.basename(csv_file)}')
        
        df = pd.read_csv(csv_file, header=[0, 1, 2, 3])
        print(f'    DataFrame shape: {df.shape}')
        
        # Find individual animals
        individuals = set()
        for col in df.columns:
            if len(col) == 4:
                individual_name = col[1]
                if 'individual_c' in str(individual_name):
                    individuals.add(individual_name)
        
        print(f'    Found {len(individuals)} animals: {list(individuals)}')
        
        if not individuals:
            return None
        
        # Analyze each animal
        animal_results = {}
        
        for animal in individuals:
            print(f'    Analyzing {animal}...')
            
            # Find body_end coordinates and likelihood
            x_coords = None
            y_coords = None
            likelihood = None
            
            for col in df.columns:
                if len(col) == 4:
                    scorer, individual, bodypart, coord_type = col
                    
                    if (individual == animal and 
                        'body_end' in str(bodypart).lower() and 
                        coord_type == 'x'):
                        
                        x_coords = df[col].values
                        y_col = (scorer, individual, bodypart, 'y')
                        likelihood_col = (scorer, individual, bodypart, 'likelihood')
                        
                        if y_col in df.columns:
                            y_coords = df[y_col].values
                        if likelihood_col in df.columns:
                            likelihood = df[likelihood_col].values
                        break
            
            if x_coords is None or y_coords is None or likelihood is None:
                print(f'      Could not find complete tracking data for {animal}')
                continue
            
            # Convert to numeric
            x_coords = pd.to_numeric(x_coords, errors='coerce')
            y_coords = pd.to_numeric(y_coords, errors='coerce')
            likelihood = pd.to_numeric(likelihood, errors='coerce')
            
            # Calculate movement metrics with tracking efficiency
            movement_data = calculate_movement_metrics_with_tracking(x_coords, y_coords, likelihood)
            
            animal_results[animal] = movement_data
            
            if movement_data and movement_data['error'] is None:
                print(f'      {animal}: {movement_data[\"tracking_efficiency\"]:.1f}% efficiency, {movement_data[\"mean_speed_cm_s\"]:.2f} cm/s')
                print(f'      Quality: {movement_data[\"tracking_quality\"]}, Segments: {movement_data[\"num_segments\"]}')
            else:
                print(f'      {animal}: {movement_data.get(\"error\", \"failed\")}')
        
        return animal_results
    
    except Exception as e:
        print(f'    Error analyzing CSV: {e}')
        import traceback
        traceback.print_exc()
        return None

# Find all group CSV files
print('Searching for group analysis CSV files...')
csv_files = glob.glob('videos/August_2025_Group/*_el.csv')
print(f'Found {len(csv_files)} group CSV files')

# Process each group video
all_movement_results = []

for csv_file in csv_files:
    video_name = os.path.basename(csv_file)
    print(f'\\nAnalyzing: {video_name}')
    
    # Extract date and match with experimental design
    date = extract_date_from_filename(video_name)
    
    if date and date in EXPERIMENT_DESIGN:
        exp_info = EXPERIMENT_DESIGN[date]
        print(f'  Matched with experiment: {date} - {exp_info[\"type\"]}')
    else:
        print(f'   Could not match with experimental design')
        batch_match = re.search(r'[Bb]atch[_\s]*(\d+)', video_name)
        if batch_match:
            batch_num = int(batch_match.group(1))
            for d, info in EXPERIMENT_DESIGN.items():
                if info['batch'] == batch_num:
                    date = d
                    exp_info = info
                    print(f'   Matched via batch {batch_num}: {date} - {exp_info[\"type\"]}')
                    break
            else:
                exp_info = {'batch': 'unknown', 'type': 'unknown', 'left': 'unknown', 'right': 'unknown'}
        else:
            exp_info = {'batch': 'unknown', 'type': 'unknown', 'left': 'unknown', 'right': 'unknown'}
    
    # Analyze the video
    analysis = analyze_group_movement_with_tracking(csv_file)
    
    if analysis:
        for animal, movement_data in analysis.items():
            result = {
                'video': video_name,
                'date': date,
                'batch': exp_info['batch'],
                'experiment_type': exp_info['type'],
                'left_shelter_odor': exp_info['left'],
                'right_shelter_odor': exp_info['right'],
                'animal_id': animal,
                **movement_data
            }
            all_movement_results.append(result)
    else:
        print(f'    failed for {video_name}')
        result = {
            'video': video_name,
            'date': date,
            'batch': exp_info['batch'],
            'experiment_type': exp_info['type'],
            'left_shelter_odor': exp_info['left'],
            'right_shelter_odor': exp_info['right'],
            'animal_id': 'unknown',
            'error': 'Analysis failed'
        }
        all_movement_results.append(result)

# Save movement results
if all_movement_results:
    move_df = pd.DataFrame(all_movement_results)
    move_output = 'group_movement_analysis_tracking.csv'
    move_df.to_csv(move_output, index=False)
    
    successful_moves = len(move_df[move_df['error'].isna()])
    
    print(f'\\job done!')
    print(f'Saved to: {move_output}')
    print(f'Total animals: {len(move_df)}')
    print(f'Successful: {successful_moves}/{len(move_df)}')
    
    # Summary statistics
    if successful_moves > 0:
        successful_data = move_df[move_df['error'].isna()]
        print(f'\\n TE summary:')
        print(f'   Average tracking efficiency: {successful_data[\"tracking_efficiency\"].mean():.1f}%')
        print(f'   Average likelihood: {successful_data[\"avg_likelihood\"].mean():.3f}')
        print(f'   Tracking quality distribution:')
        quality_counts = successful_data['tracking_quality'].value_counts()
        for quality, count in quality_counts.items():
            print(f'     {quality}: {count} animals ({count/len(successful_data)*100:.1f}%)')
        
        print(f'\\n mov summary:')
        print(f'   Average speed: {successful_data[\"mean_speed_cm_s\"].mean():.2f} cm/s')
        print(f'   Average distance: {successful_data[\"total_distance_cm\"].mean():.1f} cm')
        print(f'   Average straightness: {successful_data[\"straightness_index\"].mean():.3f}')
        
        print(f'\\n columns in output file:')
        print(f'   Total columns: {len(move_df.columns)}')
        print(f'   Key tracking columns: tracking_efficiency, avg_likelihood, tracking_quality, avg_segment_length')
        print(f'   Key movement columns: mean_speed_cm_s, total_distance_cm, straightness_index, activity_ratio')
else:
    print('\\n could not generate movement results')

print(f'\\n Analysis complete.')
"

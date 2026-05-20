import deeplabcut as dlc

config_path = "/rhomes/kkesavan/dlc_project_group/config_group.yaml"

video_paths = [
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250831_150526766.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250830_144838281.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250829_132827504.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250828_130547622.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250827_152835726.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250826_154726773.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250823_174448601.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250822_173238702.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250821_162405597.mp4",
    "/rhomes/kkesavan/dlc_project_group/August_2025_Group/Basler acA2040-90um (24827095)_20250820_161513132.mp4"
]

print("Starting video analysis..")
dlc.analyze_videos(config_path, video_paths, save_as_csv=True)
print("Video analysis complete!")

print("Creating labeled videos...")
dlc.create_labeled_video(config_path, video_paths, draw_skeleton=True)
print("Labelled videos processed")

print("Plotting trajectories...")
dlc.plot_trajectories(config_path, video_paths)
print("Trajectory plots generated")

print("ALL PROCESSING COMPLETE!")

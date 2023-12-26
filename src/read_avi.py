# Importing necessary libraries
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Directory where AVI files are stored (update this with the actual directory path)
avi_files_directory = "RGB"
avi_files_suffix = "s1_t1_color.avi"

class VideoProcessor:
    def __init__(self, directory, filesuffix):
        self.directory = directory
        self.filesuffix = filesuffix

    @staticmethod
    def get_label_from_filename(filename):
        parts = filename.split('_')
        action_number = int(parts[0][1:])  # Extracts the action number (removing 'a' and converting to int)
        subject_number = int(parts[1][1:])  # Extracts the subject number (removing 's' and converting to int)
        trial_number = int(parts[2][1:])  # Extracts the trial number (removing 't' and converting to int)
        return filename

    @staticmethod
    def get_minimum_frame_count(videos):
        return min(len(frames) for frames in videos)

    @staticmethod
    def evenly_sampled_frames(frames, target_count):
        frame_indices = np.round(np.linspace(0, len(frames) - 1, target_count)).astype(int)
        return [frames[i] for i in frame_indices]

    def process_avi_files(self):
        temp_video_data = []

        # First pass: read all videos to determine the minimum frame count
        for filename in os.listdir(self.directory):
            if filename.endswith(self.filesuffix):
                filepath = os.path.abspath(os.path.join(self.directory, filename))
                cap = cv2.VideoCapture(filepath)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        break
                cap.release()
                temp_video_data.append(frames)

        # Find the minimum frame count among all videos
        min_frame_count = self.get_minimum_frame_count(temp_video_data)

        video_data = []
        labels = []

        # Second pass: Sample frames from each video
        for filename, frames in zip(os.listdir(self.directory), temp_video_data):
            sampled_frames = self.evenly_sampled_frames(frames, min_frame_count)
            video_data.append(sampled_frames)
            label = self.get_label_from_filename(filename)  # Implement this as needed
            labels.append(label)

        print(len(video_data[0]), len(video_data[1]), len(video_data[1]))
        # Convert to NumPy arrays
        video_data = np.array(video_data)
        labels = np.array(labels)

        return video_data, labels

    @staticmethod
    def save_video(video_frames, output_file, frame_rate=15, resolution=(640, 480)):
        # Define the codec and initialize the video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Adjust based on desired file format
        out = cv2.VideoWriter(output_file, fourcc, frame_rate, resolution)

        # Write each frame to the video
        for frame in video_frames:
            # Ensure the frame size matches the specified resolution
            resized_frame = cv2.resize(frame, resolution)
            out.write(resized_frame)

        # Release the video writer
        out.release()


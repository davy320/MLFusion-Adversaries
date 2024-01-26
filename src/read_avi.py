# Importing necessary libraries
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Directory where AVI files are stored (update this with the actual directory path)
avi_files_directory = "RGB"
avi_files_suffix = "t1_color.avi"


def get_label_from_filename(filename):
    parts = filename.split('_')
    action_number = int(parts[0][1:])  # Extracts the action number (removing 'a' and converting to int)
    subject_number = int(parts[1][1:])  # Extracts the subject number (removing 's' and converting to int)
    trial_number = int(parts[2][1:])  # Extracts the trial number (removing 't' and converting to int)
    return action_number


def get_minimum_frame_count(videos):
    return min(len(frames) for frames in videos)


def evenly_sampled_frames(frames, target_count):
    frame_indices = np.round(np.linspace(0, len(frames) - 1, target_count)).astype(int)
    return [frames[i] for i in frame_indices]


def process_avi_files(directory, filesuffix, scale_factor=1, min_frame_count=None, resnet=False):
    video_frames_dict = {}

    # Process each file and store frames in a dictionary
    for filename in os.listdir(directory):
        if filename.endswith(filesuffix):
            filepath = os.path.abspath(os.path.join(directory, filename))
            cap = cv2.VideoCapture(filepath)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = frame
                    if resnet:
                        gray_frame = cv2.resize(gray_frame,
                                                (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))
                    else:
                        gray_frame = cv2.resize(gray_frame, (224, 224))
                    frames.append(gray_frame)
                else:
                    break
            cap.release()
            video_frames_dict[filename] = frames

    if min_frame_count is None:
        min_frame_count = get_minimum_frame_count(list(video_frames_dict.values()))

    video_data = []
    labels = []

    # Process each item in the dictionary
    for filename, frames in video_frames_dict.items():
        sampled_frames = evenly_sampled_frames(frames, min_frame_count)
        video_data.append(sampled_frames)
        label = get_label_from_filename(filename)
        labels.append(label)

    print("Total frames/RGBMPG: ", min_frame_count)

    video_data = np.array(video_data)
    labels = np.array(labels)

    return video_data, labels


def save_video(video_frames, output_file, frame_rate=15):
    if len(video_frames[0].shape) == 2:
        is_grayscale = True
        temp_frame = cv2.cvtColor(video_frames[0], cv2.COLOR_GRAY2BGR)
        resolution = (temp_frame.shape[1], temp_frame.shape[0])
    else:
        is_grayscale = False
        resolution = (video_frames[0].shape[1], video_frames[0].shape[0])

    fourcc = cv2.VideoWriter.fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, resolution, not is_grayscale)

    for frame in video_frames:
        if is_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()


def save_frames_as_images(video_frames, output_directory, image_prefix='frame', image_format='jpg'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, frame in enumerate(video_frames):
        image_path = os.path.join(output_directory, f"{image_prefix}_{i}.{image_format}")
        cv2.imwrite(image_path, frame)

    print(f"Saved {len(video_frames)} frames as images in '{output_directory}'")

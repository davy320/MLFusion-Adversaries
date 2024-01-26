from read_avi import *


# Function to split the data into training and testing sets
def split_data(video_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(video_data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Function to prepare data for machine learning training
def prepare_data_for_ml(directory, filesuffix):
    video_processor = VideoProcessor(directory, filesuffix)
    video_data, labels = video_processor.process_avi_files(20)
    print("Preprocessed data shape: ", video_data.shape)
    X_train, X_test, y_train, y_test = split_data(video_data, labels)
    return X_train, X_test, y_train, y_test


# Prepare data for ML training
X_train, X_test, y_train, y_test = prepare_data_for_ml(avi_files_directory, avi_files_suffix)


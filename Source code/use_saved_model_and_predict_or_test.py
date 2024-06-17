import csv

import os
import sys
import tqdm

from matplotlib import pyplot as plt
import itertools

import numpy as np
import pandas as pd
import keras.models

from sklearn.metrics import classification_report, confusion_matrix


def preprocess_data(data_folder, csv_out_path):
    """
        Preprocessing Data and generate CSV file
        regarding the point with the highest score as the body keypoint for each heatmap
        generate CSV file to store x, y coordinates and scores of keypoints
        using the CSV file on csv_out_path as input data for pose classification

    :param data_folder: folder directory of the original dataset (the "actions" folder)
    :param csv_out_path: file path of the output CSV file
    :no returns

    """
    # Create the output CSV file
    with open(csv_out_path, 'w', newline='') as csv_out_file:
        # Add header to the output CSV file
        header_list = []
        header_names = [['bodypart' + str(i) + '_x', 'bodypart' + str(i) + '_y',
                         'bodypart' + str(i) + '_score'] for i in range(1, 18)]
        for column in header_names:
            header_list += column
        header_list = ['file_name'] + header_list + ['class_no', 'class_name']
        csv_out_dict_writer = csv.DictWriter(csv_out_file,
                                             delimiter=',',
                                             fieldnames=header_list)
        csv_out_dict_writer.writeheader()
        # Add content to output CSV file
        csv_out_writer = csv.writer(csv_out_file,
                                    delimiter=',',
                                    quoting=csv.QUOTE_MINIMAL)

        # List all pose class names
        pose_class_names = sorted([n for n in os.listdir(data_folder) if not n.startswith('.')])

        # Numbering pose classes
        no_class = 0
        no_class -= 1

        # Loop through classes
        for pose_class_name in pose_class_names:

            # Numbering classes
            no_class += 1

            # Prompt processing progress
            print('Processing', pose_class_name, file=sys.stderr)

            # Obtain pose class path
            pose_class_path = os.path.join(data_folder, pose_class_name)

            """
             a body heatmap file is a numpy array file with the shape of [1, 17, 64, 48] which 
             corresponds to 17 body keypoints 
            """

            # List all body heatmap filenames in the class
            body_heatmap_names = sorted(
                [n for n in os.listdir(pose_class_path) if not n.startswith('.')])

            # Loop through body heatmap numpy files
            for body_heatmap_name in tqdm.tqdm(body_heatmap_names):
                body_heatmap_path = os.path.join(pose_class_path, body_heatmap_name)

                # Load body heatmap
                body_heatmap = np.load(body_heatmap_path)

                # Create a list for storing coordinates and scores
                landmarks = []

                # Loop through keypoint heatmaps
                for no_heatmap in range(len(body_heatmap[0])):
                    # Process keypoint heatmap
                    # taking point with the highest score as body keypoint and store its score and coordinates
                    heatmap = body_heatmap[0][no_heatmap]
                    score = np.amax(heatmap)
                    coordinates = np.where(heatmap == score)
                    coordinates = [coordinates[1][0] + 1, coordinates[0][0] + 1]

                    landmarks = landmarks + coordinates + [score]

                # Writer out a row on CSV file on each body heatmap
                csv_out_writer.writerow([body_heatmap_name] + landmarks + [no_class, pose_class_name])


def load_pose_landmarks(csv_path):
    """
        Load a CSV created by preprocess_data
        reformat the data in CSV for training and testing DNN models

    :param csv_path: file path of the CSV file
    :returns:
        X: Landmark coordinates and scores of shape (N, 17 * 3)
        y: Ground truth labels of shape (N, label_count)
        classes: The list of all class names found in the dataset

    """
    # Load the CSV file
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()

    # Drop the file_name columns becasue we don't need it during training.
    df_to_process.drop(columns=['file_name'], inplace=True)

    # Extract the list of class names
    classes = df_to_process.pop('class_name').unique()

    # Extract the labels
    y = df_to_process.pop('class_no')

    # Convert the input features and labels into the correct format for training.
    X = df_to_process.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def convert_body_heatmap_to_landmarks(body_heatmap_path):
    """Convert input body heatmap to landmarks that can be used by model to predict

    :param body_heatmap_path: file path of input body heatmap
    :return: landmarks: a list containing landmark coordinates and scores of shape [17 * 3]

    """
    body_heatmap = np.load(body_heatmap_path)
    landmarks = []
    for no_heatmap in range(len(body_heatmap[0])):

        heatmap = body_heatmap[0][no_heatmap]
        score = np.amax(heatmap)
        coordinates = np.where(heatmap == score)

        # regulate the coordinates to [x, y]
        coordinates = [float(coordinates[1][0] + 1), float(coordinates[0][0] + 1)]
        landmarks = landmarks + coordinates + [float(score)]

    return landmarks


# Predict on single body heatmap
# Create a numpy array for storing class names
class_names = np.array(['action_down', 'action_inside', 'action_new', 'action_outside',
                        'action_remove_block', 'action_select_block', 'action_switch',
                        'action_up', 'block_events_ran', 'category_control',
                        'category_events', 'category_looks', 'category_motion',
                        'category_sound', 'dummy', 'select_avatar', 'select_backdrop'])

# Set load model path
load_model_path = './weights.best.hdf5'
# Set body heatmap path
body_heatmap_path = './actions/actions/action_up/action_up_57_0.npy'

# Load model and generate landmarks
model = keras.models.load_model(load_model_path)
landmarks = convert_body_heatmap_to_landmarks(body_heatmap_path)
landmarks = np.expand_dims(landmarks, axis=0)

# Make a prediction
predictions = model.predict(landmarks)

# Convert the prediction result to class name
predictions_label = [class_names[i] for i in np.argmax(predictions, axis=1)]
print(predictions_label)


# Testing model in new dataset
# Set new dataset folder directory
test_data_folder = os.path.join(os.getcwd(), 'actions\\actions')
# Set csv output path
csv_out_path = os.path.join(os.getcwd(), 'testdata_experiment.csv')

# Preprocess data and generate csv
preprocess_data(test_data_folder, csv_out_path)

# Process csv and generate landmarks
X_test, y_test, class_names = load_pose_landmarks('testdata_experiment.csv')

# Plot the confusion matrix
# Classify pose in the TEST dataset using the trained model
y_pred = model.predict(X_test)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix for better understand the performance of the model
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

plot_confusion_matrix(cm,
                      class_names,
                      title ='Confusion Matrix of Pose Classification Model')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_true_label,
                                                          y_pred_label))

# Print total loss and accuracy on test dataset
loss, accuracy = model.evaluate(X_test, y_test)
print("On test dataset \nTotal Loss: ", loss, "\nTotal Accuracy: ", accuracy)


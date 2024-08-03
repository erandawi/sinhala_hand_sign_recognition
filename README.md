# Sinhala Hand Sign Recognition

This project aims to create a hand sign recognition system for the Sinhala sign alphabet. The current version of the project recognizes three Sinhala characters: අ , ආ , and ඉ . The ultimate goal is to extend the system to recognize the entire Sinhala sign alphabet.

## Table of Contents

- [Introduction](#introduction)
- [Libraries Used](#libraries-used)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)
- [References](#references)

## Introduction

The project uses a custom Convolutional Neural Network (CNN) model to recognize hand signs. The system captures real-time video feed from a webcam, processes the images to detect hand landmarks using MediaPipe, and uses the trained model to predict the corresponding Sinhala letter.

## Libraries Used

- **TensorFlow**: An open-source library developed by Google for deep learning applications. It is used here to build and train the custom CNN model.
- **NumPy**: A fundamental package for scientific computing in Python. It is used for numerical operations and data manipulation.
- **OpenCV**: An open-source computer vision and machine learning software library. It is used to capture and process the video feed.
- **MediaPipe**: A cross-platform framework by Google for building multimodal applied ML pipelines. It is used to detect hand landmarks.
- **Flask**: A micro web framework written in Python. It is used to create a web application to display the video feed and predicted letters.
- **scikit-learn**: A machine learning library for Python. It is used for label encoding in this project.

### Why These Libraries?

- **TensorFlow**: Chosen for its extensive support for deep learning models and ease of use.
- **NumPy**: Essential for efficient numerical computations and data manipulation.
- **OpenCV**: Provides powerful tools for real-time computer vision applications, making it ideal for capturing and processing video feeds.
- **MediaPipe**: Offers a robust and efficient way to detect and track hand landmarks, which is crucial for this project.
- **Flask**: Allows quick development of web applications to showcase the model's predictions.
- **scikit-learn**: Provides simple and efficient tools for data analysis and preprocessing.

## System Architecture

1. **Data Collection**: Capture hand sign images using a webcam and store the corresponding landmarks and labels.
2. **Preprocessing**: Process the collected images to extract and flatten the hand landmarks.
3. **Model Training**: Train a custom CNN model using the preprocessed data.
4. **Real-time Prediction**: Use the trained model to predict hand signs from a live webcam feed and display the predictions on a web application.

## Model Architecture

The custom CNN model used in this project consists of the following layers:

1. **Input Layer**: Accepts input of shape (21, 3, 1), representing 21 hand landmarks each with 3 coordinates (x, y, z).
2. **Convolutional Layer 1**: 32 filters of size (3, 3), activation function ReLU.
3. **MaxPooling Layer 1**: Pool size (2, 2).
4. **Dropout Layer 1**: Dropout rate of 0.25 to prevent overfitting.
5. **Flatten Layer**: Flattens the input.
6. **Dense Layer 1**: 128 units, activation function ReLU.
7. **Dropout Layer 2**: Dropout rate of 0.5.
8. **Output Layer**: Number of units equal to the number of classes (3 in the current setup), activation function Softmax.

### Why a Custom CNN?

A custom CNN is used to tailor the architecture to the specific needs of the hand sign recognition task. The model is designed to capture spatial features from the hand landmarks effectively and make accurate predictions based on those features.

## Usage

### Prerequisites

- Python 3.7+
- [requirements.txt](requirements.txt) to install the necessary libraries.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sinhala_hand_sign.git
   cd sinhala_hand_sign
   ```

2. Install the required libraries::

   ```pip install -r requirements.txt

   ```

### Data Collection

To collect data for additional Sinhala letters, update the sinhala_letters dictionary in main.py and follow the existing data collection process.

### Future Work

- Extend the model to recognize the entire Sinhala sign alphabet.
- Improve the accuracy and efficiency of the model.
- Implement a more user-friendly interface for the web application.
- Explore the use of other deep learning architectures for better performance.
- Integrate additional preprocessing steps to enhance the model's robustness.
- Implement a mobile application version for broader accessibility.
- License
- This project is licensed under the MIT License. See the LICENSE file for more details.

### References

1. NumPy Documentation
2. OpenCV Documentation
3. MediaPipe Documentation
4. scikit-learn Documentation
5. TensorFlow Documentation
6. Flask Documentation

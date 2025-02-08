This project builds a deep learning model using ResNet50 (Transfer Learning) to classify facial expressions into seven emotions:
😠 Angry, 🤢 Disgust, 😨 Fear, 😀 Happy, 😐 Neutral, 😢 Sad, and 😲 Surprise.

Dataset: FER-2013
The model is trained on the Facial Expression Recognition 2013 (FER-2013) dataset.
It consists of 35,887 grayscale images of faces, each labeled with one of 7 emotions.
Images are 48x48 pixels, and are fed into a custom-built Residual Neural Network

Model: Residual Neural Network
-> Around 12 million parameters
-> Image data generation using ImageDataGenerator class from Tensorflow
->Batch Normalization (for stable training)
-> Dropout (0.4) (to reduce overfitting)
-> Adam Optimizer (learning rate = 1e-4) (adaptive learning)

📂 Emotion-Classifier
 ├── train.py                # Training script for the model
 ├── model.h5                # Saved trained model
 ├── notebook.ipynb          # Full Colab notebook (if applicable)
 ├── dataset/                # Dataset folder (not uploaded to GitHub)
 ├── README.md               # Project documentation (this file)
 ├── LICENSE                 # MIT License for open-source usage


Results:
Training Accuracy - ~70-80%
Validation Accuracy - ~50-60%

Learnings:
1. Need more data for optimzation
2. Handling grayscale images result in less feature extraction

I hope you enjoy reading this project, this was a fun experience.

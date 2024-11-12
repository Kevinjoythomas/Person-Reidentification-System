# Person Re-Identification System

This Person Re-Identification (ReID) system is designed to detect and match individuals across frames in a video. It leverages feature extraction to identify the first and last occurrence of a person in the video, which can be useful for tracking individuals in surveillance applications. The system is built using a Flask API and a ResNet50 model, trained with open-source weights from the CrowdHuman dataset.

## Features

### Feature Extraction
- The system processes video frames to extract feature embeddings for each detected person.
- ResNet50, fine-tuned on CrowdHuman, is used for extracting feature vectors.
- Features are stored in a designated folder for quick access and retrieval.

### Matching API
- A Flask API accepts a person’s image and compares it against the stored features to identify the person’s first and last appearance in the video.
- This matching process involves cosine similarity to find the most similar feature vectors in the stored dataset.
- Returns the frame timestamps of the first and last appearances.

## How It Works

### Setup and Inference
- Video frames are processed in batches to detect persons and extract their feature embeddings.
- Feature vectors are saved as separate files (e.g., in `.pkl` format) for each frame in the designated `features` folder.

### API Usage
- The Flask API endpoint `/predict` allows users to upload a person's image.
- The API searches through saved features, comparing the uploaded image's embedding with stored embeddings.
- Upon finding the closest matches, it returns the timestamps of the first and last frames where the person appears.


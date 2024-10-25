from flask import Flask, request, jsonify, render_template
import os
import threading
from reid import REID
import pickle
import uuid 
import numpy as np  # Assuming you'll use numpy for distance calculation
from datetime import datetime
import cv2
import io
import base64
from PIL import Image
app = Flask(__name__)
reid = REID()

FEATURE_FOLDER = './features'
if not os.path.exists(FEATURE_FOLDER):
    os.makedirs(FEATURE_FOLDER)
DISTANCE_THRESHOLD = 360
task_results = {}

app.config['FEATURE_FOLDER'] = FEATURE_FOLDER


# def extract_datetime_from_filename(filename):
#     try:
#         # Split filename by underscore and hyphen
#         parts = filename.split('_')
#         date_str = parts[1]
#         time_str = parts[2].split('.')[0]  # Get the time part without the extension

#         # Combine date and time strings
#         datetime_str = f"{date_str} {time_str}"
#         return datetime.strptime(datetime_str, "%Y-%m-%d %H-%M-%S-%f")  # Include milliseconds
#     except Exception as e:
#         print(f"Error extracting date/time from filename: {filename}, {str(e)}")
#         return None


def extract_datetime_from_timestamp(timestamp):
    """Converts a timestamp string into a datetime object."""
    try:
        datetime_str = timestamp.replace('_', ' ', 1)
        # Expected timestamp format: "YYYY-MM-DD HH-MM-SS-fff"
        return datetime.strptime(datetime_str, "%Y-%m-%d %H-%M-%S-%f")
    except ValueError as e:
        print(f"Error parsing timestamp: {timestamp}, {str(e)}")
        return None

def load_features_from_pickle(pickle_path):
    """Load features and timestamps from a pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)  # [(features, timestamp), ...]
    except Exception as e:
        print(f"Error reading pickle file {pickle_path}: {str(e)}")
        return []
    
def handle_feature_extraction(task_id, pictures, data, event):
    """Thread function to extract features and store results in a shared dictionary."""
    try:
        with app.app_context():  # Ensure Flask context within the thread

            # Extract features from pictures
            input_features = reid._features(pictures)

            # Parse and validate input data
            start_epoch = int(data.get('start_epoch'))
            end_epoch = int(data.get('end_epoch'))
            camera_ids = data.get('camera_id')
            person_id = data.get('person_id')

            if not isinstance(camera_ids, list):
                camera_ids = [camera_ids]

            start_datetime = datetime.fromtimestamp(start_epoch)
            end_datetime = datetime.fromtimestamp(end_epoch)

            if start_datetime > end_datetime:
                raise ValueError("Start time must be before end time.")

            final_result = []

            # Process each camera ID
            for camera_id in camera_ids:
                print(f"Processing features for camera ID: {camera_id}")

                camera_result = {
                    "first_match": None,
                    "last_match": None,
                    "person_id": person_id,
                    "camera_id": camera_id
                }

                video_folder = os.path.join(FEATURE_FOLDER, str(camera_id))
                if not os.path.exists(video_folder):
                    final_result.append({
                        "error": f"Camera ID {camera_id} not found",
                        "camera_id": camera_id
                    })
                    continue

                # Process each track file
                for track_id in os.listdir(video_folder):
                    track_pickle_file = os.path.join(video_folder, track_id)

                    if os.path.exists(track_pickle_file):
                        print(f"Checking track folder: {track_pickle_file}")
                        stored_data = load_features_from_pickle(track_pickle_file)
                        matches = 0
                        # Check timestamps and compute feature distances
                        for stored_feature, timestamp in stored_data:
                            feature_datetime = extract_datetime_from_timestamp(timestamp)

                            if start_datetime <= feature_datetime <= end_datetime:
                                if(matches<12):
                                    distance = float(np.mean(reid.compute_distance(input_features, stored_feature)))
                                else:
                                    print("skipping found enough matches before")
                                    distance = 0
                                if distance < DISTANCE_THRESHOLD:
                                    print(f"Match found for camera {camera_id}, track {track_id}")
                                    matches += 1
                                    
                                        
                                    if not camera_result["first_match"]:
                                        camera_result["first_match"] = feature_datetime
                                    camera_result["last_match"] = feature_datetime

                # Add message if no match was found
                if camera_result["first_match"] is None:
                    camera_result["message"] = "No matching features found within the specified distance."

                final_result.append(camera_result)

            print("Returning results for all cameras.")

            # Store the result in the shared dictionary
            task_results[task_id] = {
                "status": "completed",
                "result": final_result
            }

    except Exception as e:
        # Handle any errors encountered during processing
        print(f"Error during feature extraction: {str(e)}")
        task_results[task_id] = {
            "status": "failed",
            "error": str(e)
        }

    finally:
        # Signal that the task is complete
        event.set()

# Flask route to receive requests and spawn a thread for each request
@app.route("/extract_features", methods=["POST"])
def extract_features():
    print("Received request to extract features")

    pictures = []
    if 'images' in request.files:
        images = request.files.getlist("images")
        for im in images:
            file_bytes = np.frombuffer(im.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                pictures.append(image)
    elif 'images' in request.json:
        try:
            base64_images = request.json['images']
            for base64_image in base64_images:
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                pictures.append(cv_image)
        except Exception as e:
            print(f"Error decoding base64 image: {str(e)}")
            return jsonify({"error": f"Error decoding base64 image: {str(e)}"}), 400

    if not pictures:
        return jsonify({"error": "No images found in the request."}), 400

    data = request.json if request.is_json else request.form

    task_id = str(uuid.uuid4())

        # Create an Event to signal when the task is complete
    event = threading.Event()
    # Start a new thread to handle the request
    thread = threading.Thread(target=handle_feature_extraction, args=(task_id, pictures, data, event))
    thread.start()

    event.wait(timeout=200)  # Adjust timeout as needed

    # Fetch the result from the shared dictionary
    result = task_results.pop(task_id, None)

    if not result:
        return jsonify({"error": "Task did not complete in time."}), 504

    return jsonify(result), 200



if __name__ == "__main__":
    app.run(debug=True, threaded=True)

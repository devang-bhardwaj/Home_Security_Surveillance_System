# Real-Time Face Recognition System

This is a Python-based Real-Time Face Recognition system using OpenCV, dlib, and face_recognition libraries. The system detects and recognizes faces from live webcam input and logs the recognized faces with their corresponding timestamps in an Excel file.

## Features

- **Real-Time Face Recognition**: Detects and recognizes faces from webcam input in real-time.
- **Face Registration**: Automatically saves and encodes new faces for recognition.
- **Excel Logging**: Logs the recognized face with timestamp and image path in an Excel file.
- **Face Deletion**: Allows the user to delete saved known faces from the system.
- **Conditional Formatting in Excel**: Marks unrecognized faces in red and creates hyperlinks to saved images.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/face-recognition-system.git
   cd face-recognition-system
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The required libraries include:
   - `opencv-python`
   - `dlib`
   - `face_recognition`
   - `numpy`
   - `pandas`
   - `openpyxl`
   - `tqdm`
3. Download the necessary dlib models:

   - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

   Extract these files and place them in the project directory.

## Usage

### Step 1: Encode Faces

1. Place images of the faces you want to recognize in the `Images/` folder. Ensure the file name is the person's name (e.g., `Devang.jpg`).
2. Run the script:
   ```bash
   python main.py
   ```
3. The system will automatically encode the new faces and add them to the database.

### Step 2: Real-Time Face Recognition

1. The system will start the webcam and begin recognizing faces in real-time.
2. When a known face is recognized, it displays the name and recognition confidence on the screen.
3. Press `s` to save the photo and log the event in the Excel file.
4. Press `q` to quit.

### Step 3: Deleting Known Faces

Before starting the face recognition, you will be prompted if you want to delete any previously registered faces. Follow the instructions to remove unwanted faces.

## Face Recognition Accuracy

- The system uses a distance-based recognition model.
- If the distance between known face encodings and the captured face encoding is below a threshold (`0.47`), it considers the face recognized.
- You can adjust the threshold value in the `real_time_face_recognition` function for more/less strict recognition.

## Excel Logging and Formatting

- The Excel file (`recognition_log.xlsx`) logs every recognized face with the name, timestamp, and image path.
- Conditional formatting is applied:
  - Known faces are displayed in **black**.
  - Unknown faces are displayed in **red**.
  - Image paths are hyperlinked for easy access to captured photos.

## Contributing

Feel free to submit issues and pull requests for improvements.

## Author

- **Devang Sharma** (devangsharma.developer@gmail.com)

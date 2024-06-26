# FOD-streamlit

# Real-time Object Detection and GPS Localization

This Streamlit-based web application performs real-time object detection and GPS localization using a pre-trained YOLOv8 model. The application supports various input sources, including webcam feeds, IP camera streams, video URLs, and .mp4 files.

## Features
- Real-time object detection using YOLOv8
- GPS localization via perspective transformation
- Interactive map displaying detected objects
- Video feed display
- List of detected objects with details

## Installation

1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Place the YOLOv8 model (`yolov8.pt`) in the appropriate path.

## Usage

1. Run the Streamlit application:
   ```sh
   streamlit run main.py
   ```

2. Open your web browser and navigate to the displayed URL (usually `http://localhost:8501`).

3. On the start page, select the input source:
   - **Webcam**: Uses your default webcam.
   - **IP Camera**: Enter the URL of the IP camera stream.
   - **Video URL**: Enter the URL of the video.
   - **MP4 File**: Enter the file path of the .mp4 video.

4. Click the **Start** button to begin.

5. The main page will display:
   - A blue semi-transparent title bar at the top with the text "Object Detection".
   - An interactive map on the left (80% of the width).
   - Video feed at the bottom-right (30% of the remaining width).
   - A list of detected objects on the right.

## Code Structure

### main.py
- **capture_frame_from_camera**: Captures frames from the provided URL.
- **get_frames**: Generator function to read video frames from the selected input source.
- **update_ui**: Updates the UI with the map, video feed, and detected objects.

### utils.py
- **load_model**: Loads the YOLOv8 model.
- **detect_objects**: Detects objects in a frame using the YOLOv8 model.
- **compute_perspective_transform**: Computes the perspective transformation matrix.
- **transform_to_gps**: Transforms object coordinates to GPS coordinates.

### Layout
- The main page layout includes a title bar, an interactive map, a video feed, and a list of detected objects.
- The map occupies 80% of the page's width on the left side.
- The video feed occupies 30% of the remaining width at the bottom-right.
- The detected object information occupies the remaining space on the right.

## Notes
- Ensure you have a stable internet connection if using IP camera or video URLs.
- Adjust the model path in `main.py` to the location of your YOLOv8 model file.

---

This summary should give a clear overview of your project and help users understand how to install and use it.

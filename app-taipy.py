import cv2
import numpy as np
import requests
import taipy as tp
from utils import load_model, detect_objects, compute_perspective_transform, transform_to_gps

def capture_frame_from_camera(url):
    try:
        response = requests.get(url)
        frame = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        return frame, None
    except Exception as e:
        return None, str(e)

def get_frames(input_source, url_input):
    if input_source == "Webcam":
        cap = cv2.VideoCapture(0)
    elif input_source == "IP Camera":
        cap = cv2.VideoCapture(url_input)
    elif input_source == "Video URL":
        cap = cv2.VideoCapture(url_input)
    elif input_source == "MP4 File":
        cap = cv2.VideoCapture(url_input)
    else:
        print("Invalid source")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def update_ui(input_source, url_input, model, perspective_matrix):
    for frame in get_frames(input_source, url_input):
        detections = detect_objects(model, frame)
        transformed_detections = transform_to_gps(detections, perspective_matrix)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update map with detections
        map_html = tp.Map(
            lat=10.0, lon=10.0, zoom=2,
            markers=[{"lat": det['gps']['lat'], "lon": det['gps']['lon'], "popup": f"{det['name']} ({det['confidence']})"} for det in transformed_detections]
        ).to_html()

        # Update video feed
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        video_feed_html = f'<img src="data:image/jpeg;base64,{frame_bytes}" width="240" height="180">'

        # Update detected objects info
        object_info_html = '<br>'.join([f"ID: {det['id']}, Name: {det['name']}, Confidence: {det['confidence']}" for det in transformed_detections])

        tp.update_data(
            map_html=map_html,
            video_feed_html=video_feed_html,
            object_info_html=object_info_html
        )

model_path = "path/to/yolov8.pt"
model = load_model(model_path)
perspective_matrix = compute_perspective_transform(
    [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],  # Example GPS points
    [(100, 100), (200, 100), (200, 200), (100, 200)]  # Example image points
)

tp.run(
    layout=[
        tp.HTML(name="title", value="<h1>Object Detection</h1>"),
        tp.Map(name="map", lat=10.0, lon=10.0, zoom=2),
        tp.HTML(name="video_feed", value=""),
        tp.HTML(name="object_info", value=""),
        tp.Button(name="start", label="Start", on_click=lambda: update_ui("Webcam", "", model, perspective_matrix))
    ],
    data=dict(map_html="", video_feed_html="", object_info_html="")
)

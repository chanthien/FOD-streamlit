import streamlit as st
import cv2
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from utils import load_model, detect_objects, compute_perspective_transform, transform_to_gps

# Function to get video frames
def capture_frame_from_camera(url):
    try:
        response = requests.get(url)
        frame = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        return frame, None
    except Exception as e:
        return None, str(e)

# Initialize the Streamlit app
st.set_page_config(layout="wide")

# Start Page
if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:
    st.title("Real-time Object Detection and GPS Localization")
    st.subheader("Select Input Source")
    input_source = st.selectbox("Select Input Source", ["Webcam", "IP Camera", "Video URL", "MP4 File"])

    url_input = ""
    if input_source in ["IP Camera", "Video URL", "MP4 File"]:
        url_input = st.text_input("Enter URL or Path")

    start_button = st.button("Start")
    if start_button:
        st.session_state.start = True
        st.session_state.input_source = input_source
        st.session_state.url_input = url_input
else:
    # Load the model and perspective matrix only once
    model_path = "path/to/yolov8.pt"
    model = load_model(model_path)
    perspective_matrix = compute_perspective_transform(
        [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],  # Example GPS points
        [(100, 100), (200, 100), (200, 200), (100, 200)]  # Example image points
    )

    # Main Page
    st.markdown(
        """
        <style>
        .title-bar {
            background-color: rgba(0, 0, 255, 0.5);
            color: red;
            font-size: 24px;
            height: 100px;
            line-height: 100px;
            text-align: center;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="title-bar">Object Detection</div>', unsafe_allow_html=True)

    # Create layout with containers
    col1, col2 = st.columns([4, 1])

    with col1:
        map_container = st.container()

    with col2:
        video_container = st.container()
        object_container = st.container()

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
            st.error("Invalid source")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

    def update_ui():
        for frame in get_frames(st.session_state.input_source, st.session_state.url_input):
            detections = detect_objects(model, frame)
            transformed_detections = transform_to_gps(detections, perspective_matrix)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with map_container:
                m = folium.Map(location=[10.0, 10.0], zoom_start=2)
                for det in transformed_detections:
                    folium.Marker(
                        location=[det['gps']['lat'], det['gps']['lon']],
                        popup=f"{det['name']} ({det['confidence']})"
                    ).add_to(m)
                st_folium(m, width=700, height=500)

            with video_container:
                st.image(frame, use_column_width=True)

            with object_container:
                st.write(transformed_detections)

    update_ui()

import streamlit as st
import tempfile
import estimated_water_height_process as ewhp
import os

st.title("YOLOv8 Video Processor")

video_file = st.file_uploader("Upload a video", type=["mp4"])

curr_dir = os.path.dirname(os.path.abspath(__file__))

if video_file is not None:
    # Simpan video sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
    output_path = f"hasil/{video_file.name}"

    st.video(video_file)  # Preview video input

    st.subheader("Set Confident Threshold Parameters")
    # Input threshold
    wg_threshold = st.slider("WATER GAUGE Threshold", 0.0, 1.0, 0.6)
    n_threshold = st.slider("NUMBER OBJECT DETECTION Threshold", 0.0, 1.0, 0.6)
    ws_threshold = st.slider("WATER SEGMENTATION Threshold", 0.0, 1.0, 0.7)

    # Tombol untuk mulai proses
    if st.button("Process Video"):
        st.write("Processing video...")
        ewhp.detect(video_path, output_path, wg_threshold, n_threshold, ws_threshold)
        st.success("Processing complete!")
        st.write(f"video saved to {output_path}")

        if os.path.exists(output_path):
            video_file_output = open(output_path, "rb")
            video_bytes = video_file_output.read()
            st.video(video_bytes)
        else:
            st.error("Processed video not found!")

    tfile.close()
# Estimated Water Height Detection

This project provides a comprehensive solution for estimating water levels from video and image feeds using advanced computer vision techniques, specifically object detection and segmentation. It features an interactive web interface built with Streamlit, enabling users to upload media, configure detection parameters, and visualize the estimated water height in real-time or from pre-recorded content.

## Features

*   **Multi-Model Object Detection**: Leverages YOLO models for a multi-stage detection process:
    *   **Water Gauge Detection**: Identifies the water gauge within the input media.
    *   **Number Object Detection**: Detects numerical markings on the identified water gauge.
    *   **Water Segmentation**: Precisely segments the water body to determine its level relative to the gauge.
*   **Real-time Water Height Estimation**: Calculates and displays the estimated water height in centimeters based on the combined output of the detection and segmentation models.
*   **Interactive Streamlit User Interface**: A user-friendly web application for intuitive interaction, allowing easy file uploads and parameter adjustments.
*   **Configurable Confidence Thresholds**: Users can dynamically adjust confidence thresholds for each detection model (water gauge, number, water segmentation) to fine-tune performance and adapt to different environmental conditions.
*   **Support for Various Water Gauge Heights**: Accommodates different physical water gauge heights (e.g., 400 cm, 450 cm), which can be selected by the user.
*   **Video Processing Capabilities**: Processes video files frame by frame, displaying progress, and generating an output video with overlaid annotations and water height estimations. It also provides the average water height over the entire video.
*   **Image Processing Capabilities**: Processes individual image files, offering options to apply specific models for isolated analysis or run the full pipeline to estimate water height.
*   **Scalable Architecture**: Designed with modularity using a `config.py` file for centralized parameter adjustments and `estimated_water_height_process.py` for encapsulating core logic.
*   **Batch Processing Utility**: Includes a separate script (`estimate_test_dataset.py`) for automated processing and evaluation on a dataset of test frames, useful for development and performance testing.

## Installation

To set up and run the Estimated Water Height Detection project, follow these steps:

1.  **Clone the Repository**:
    First, clone the project repository from GitHub to your local machine:
    ```bash
    git clone https://github.com/galang006/water_level_measurement.git
    cd water_level_measurement
    ```

2.  **Create a Virtual Environment (Recommended)**:
    It's highly recommended to use a virtual environment to manage project dependencies:
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    Install the necessary Python libraries using pip.
    ```bash
    pip install streamlit numpy opencv-python ultralytics supervision
    ```
    *Note: The `opencv-python` package provides the `cv2` module.*

4.  **Install FFmpeg**:
    The project uses `ffmpeg` for video post-processing to ensure compatibility of the output video files. Please ensure `ffmpeg` is installed and accessible in your system's PATH.
    *   **On Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install ffmpeg`
    *   **On macOS (using Homebrew)**: `brew install ffmpeg`
    *   **On Windows**: Download from the official FFmpeg website and add it to your system's PATH.

5.  **Download Pre-trained Models**:
    The project relies on pre-trained YOLO models for its detection and segmentation tasks. You need to download these model files (`.pt` extension) and place them in a directory named `models/` within your project root.
    Create the `models` directory:
    ```bash
    mkdir models
    ```
    Then, download the following `.pt` files and place them inside the `models/` directory:
    *   `water_gauge.pt`
    *   `number.pt`
    *   `water_seg.pt`
    *(Please refer to the original project source or contact the developer for links to download these pre-trained models.)*

## Usage

The project can be used via its interactive Streamlit web interface or by running specific Python scripts for batch processing.

### Running the Streamlit Application

1.  **Start the Application**:
    Navigate to the project's root directory in your terminal and execute:
    ```bash
    streamlit run app.py
    ```
    This command will launch the Streamlit application, which typically opens in your default web browser at `http://localhost:8501`.

2.  **Upload a File**:
    On the Streamlit interface, use the "Upload a video or image" file uploader to select an input file. Supported formats include `.mp4` for videos and `.jpg`, `.jpeg`, `.png` for images.

3.  **Processing Videos**:
    If you upload a video file:
    *   The uploaded video will be displayed.
    *   **Select Water Gauge Height**: Choose the appropriate height for your water gauge (e.g., "400 cm" or "450 cm").
    *   **Set Confidence Threshold Parameters**: Adjust the sliders for "WATER GAUGE Threshold", "NUMBER OBJECT DETECTION Threshold", and "WATER SEGMENTATION Threshold" to control the sensitivity of each model.
    *   Click the "Process Video" button. A progress bar will indicate the processing status.
    *   Upon completion, the processed video with visual annotations and the "Average Water Height" calculated over the video will be displayed.

4.  **Processing Images**:
    If you upload an image file:
    *   The uploaded image will be displayed.
    *   **Select Model**: You have options to process the image:
        *   **"All"**: Applies all three models (Water Gauge, Number Object Detection, Water Segmentation) sequentially to estimate water height. This option requires selecting the water gauge height and setting individual confidence thresholds for all models.
        *   **"Water Gauge"**, **"Number Object Detection"**, or **"Water Segmentation"**: Applies only the selected model, allowing you to set a single confidence threshold for that specific model.
    *   Click the "Process Image" button (ensure you click the one relevant to your selected mode).
    *   The annotated image showing detection/segmentation results will be displayed. If "All" models were selected, the estimated water height will also be presented.

### Running Test Dataset Processing

The `estimate_test_dataset.py` script is provided for batch processing images from a local dataset. This is particularly useful for evaluating the models' performance on a predefined set of test cases.

1.  **Prepare Test Frames**:
    Ensure your test images are organized within the `dataset/test frames/` directory, following the naming conventions expected by the script (e.g., `malam 0_frame_01.jpg`).

2.  **Execute the Script**:
    From the project's root directory, run the script:
    ```bash
    python estimate_test_dataset.py
    ```
    The script will print its progress to the console and save the processed images, complete with annotations, to the `dataset/test frames/result/` directory.

## Code Structure

The project is organized into a clear and modular structure to facilitate understanding, maintenance, and extension:

```
.
├── app.py
├── config.py
├── estimated_water_height_process.py
├── estimate_test_dataset.py
├── .gitignore
├── dataset/
│   ├── video/
│   └── test frames/
│       └── result/ (Generated output for test frames)
├── models/
│   ├── water_gauge.pt
│   ├── number.pt
│   └── water_seg.pt
└── hasil/ (Generated output for app.py)
```

*   **`app.py`**:
    This is the primary entry point for the Streamlit web application. It manages the user interface elements, handles file uploads, processes user selections for models and thresholds, and orchestrates calls to the core processing functions defined in `estimated_water_height_process.py`.
*   **`config.py`**:
    A dedicated configuration file that centralizes global parameters for the project. This includes default confidence thresholds for the object detection and segmentation models, file paths to the pre-trained models, and default input/output video paths. This file allows for easy modification of key settings without altering the core logic.
*   **`estimated_water_height_process.py`**:
    This file contains the core logic for the water height estimation pipeline. It includes:
    *   Functions for loading and initializing YOLO models.
    *   Helper functions (`get_y_coor_number`, `get_number_bounding_box_height`, `get_roi_length`, `get_water_height`, `get_pixel_scale`) responsible for extracting specific data points and measurements from the detection results.
    *   `calculate_estimated_water_height`: The central function that computes the final estimated water level in centimeters by combining information from water gauge, number, and water segmentation detections.
    *   `detect`: The main function for processing video streams. It iterates through frames, applies the detection and segmentation models, annotates the frames, saves the processed video, and calculates the average water height over the entire video sequence.
*   **`estimate_test_dataset.py`**:
    A utility script designed for developers and testers. It automates the batch processing of images located in a specified test dataset (`dataset/test frames`). The script loads the necessary models, applies the full detection and estimation pipeline to each image, and saves the annotated results to a designated output directory (`dataset/test frames/result/`).
*   **`.gitignore`**:
    Specifies files and directories that Git should ignore, preventing them from being committed to the repository. This includes temporary files, cache directories (`.env`, `__pycache__`), and potentially large generated output folders (`dataset`, `extract_dataset`, `hasil`, `models`, `view_cctv`, `sub`). Note that `dataset` and `models` are listed in `.gitignore`, implying they are either generated or expected to be downloaded separately and not tracked by Git.
*   **`dataset/`**:
    This directory serves as a placeholder for input media. It typically contains sample videos (`video/`) and collections of test image frames (`test frames/`) used for development and evaluation.
*   **`models/`**:
    This directory is where the project expects to find the pre-trained YOLO model files (`.pt` extension) required for water gauge detection, number recognition, and water segmentation.
*   **`hasil/`**:
    This directory is designated for storing the output of processed videos and images generated by the `app.py` and `estimated_water_height_process.py` scripts.
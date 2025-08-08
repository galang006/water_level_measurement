# Estimated Water Height Detection

An AI-powered system designed to accurately estimate water levels from video and image inputs. This project leverages advanced deep learning models (YOLOv8) for the precise detection of water gauges, recognition of numerical markings, and segmentation of the water body, providing a comprehensive solution for automated water level monitoring. The system is presented as a user-friendly web application built with Streamlit.

## Features
-   **Multi-Modal Input Support**: Process water level estimations from both video files (`.mp4`) and static images (`.jpg`, `.jpeg`, `.png`).
-   **YOLOv8 Integration**: Utilizes three distinct YOLO models for:
    -   **Water Gauge Detection**: Identifying the water gauge in the scene.
    -   **Number Object Detection**: Recognizing numerical markings on the water gauge within a detected ROI.
    -   **Water Segmentation**: Segmenting the water body to determine its height relative to the water gauge.
-   **Configurable Parameters**: Users can adjust confidence thresholds for each detection model and specify the total height of the water gauge (e.g., 350 cm or 400 cm) via the Streamlit interface.
-   **Automated Height Calculation**: Employs a sophisticated algorithm to convert pixel distances from detections into estimated water heights in centimeters.
-   **Visual Output**: Generates annotated video or image outputs, displaying detected objects, segmentation masks, and the calculated water height directly on the visual feed.
-   **Average Water Height for Videos**: For video inputs, the system calculates and displays the average estimated water height across all processed frames.
-   **FFmpeg Integration**: Ensures compatibility and proper encoding of output video files for wide playback support.

## Prerequisites
-   **Python 3.9**: The project is developed in Python.
-   **Streamlit**: For the web-based user interface.
-   **Ultralytics**: For YOLO model inference.
-   **Supervision**: For advanced annotation and visualization of detections.
-   **OpenCV (`cv2`)**: For image and video processing.
-   **NumPy**: For numerical operations.
-   **python-dotenv**: To load environment variables from a `.env` file.
-   **FFmpeg**: An essential command-line tool for video processing and encoding, particularly for saving the output video in a widely compatible format.

You will also need pre-trained YOLO models for water gauge detection, number object detection, and water segmentation. The paths to these models are expected to be configured in a `.env` file.

## Installation

Follow these steps to set up and run the project locally:

1.  **Clone the Repository (if applicable):**
    If you have the repository, clone it:
    ```bash
    git clone https://github.com/galang006/water_level_measurement.git
    cd water_level_measurement
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Install the required Python libraries using pip:
    ```bash
    pip install streamlit ultralytics supervision opencv-python numpy python-dotenv
    ```

4.  **Install FFmpeg:**
    FFmpeg is necessary for video processing.
    -   **On Ubuntu/Debian:**
        ```bash
        sudo apt update
        sudo apt install ffmpeg
        ```
    -   **On macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```
    -   **On Windows:**
        Download the executables from the [official FFmpeg website](https://ffmpeg.org/download.html) and add them to your system's PATH environment variable.

5.  **Obtain Pre-trained YOLO Models:**
    You will need three pre-trained YOLOv8 models. These are typically `.pt` files.
    -   `MODEL_WATER_GAUGE.pt` (for water gauge detection)
    -   `MODEL_NUMBER.pt` (for number object detection)
    -   `MODEL_WATER_SEGMENTATION.pt` (for water segmentation)
    Place these models in a directory, e.g., `models/`.

6.  **Create and Configure `.env` File:**
    Create a file named `.env` in the root directory of your project and populate it with the paths to your models and desired default thresholds.

    ```dotenv
    # Example .env file
    WATER_GAUGE_THRESHOLD=0.6
    NUMBER_OBJECT_DETECTION_THRESHOLD=0.6
    WATER_SEGMENTATION_THRESHOLD=0.6
    WATER_GAUGE_HEIGHT=400

    MODEL_WATER_GAUGE=models/water_gauge_model.pt
    MODEL_NUMBER=models/number_model.pt
    MODEL_WATER_SEGMENTATION=models/water_segmentation_model.pt

    # Optional: Default paths for `estimated_water_height_process.py` standalone execution
    VIDEO_PATH=input_videos/sample.mp4
    OUTPUT_VIDEO_PATH=hasil/output_video.mp4
    ```
    *Adjust model paths (`models/your_model_name.pt`) to reflect your actual file structure.*

7.  **Create Output Directory:**
    Ensure a directory named `hasil` exists in the project root to store processed output videos and images.
    ```bash
    mkdir hasil
    ```

## Usage

To run the Streamlit application:

1.  **Start the Application:**
    Navigate to the project's root directory in your terminal and execute:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser (usually at `http://localhost:8501`).

2.  **Upload File:**
    On the Streamlit interface, use the "Upload a video or image" section to upload your desired media file.

3.  **Configure Parameters:**
    -   **Video Processing**:
        -   Select the "Water Gauge Height" (e.g., "400 cm" or "450 cm") relevant to your water gauge.
        -   Adjust the "Confident Threshold Parameters" using the sliders for Water Gauge, Number Object Detection, and Water Segmentation.
        -   Click "Process Video" to start the analysis. A progress bar will indicate the processing status.
    -   **Image Processing**:
        -   Select the model you want to use for detection ("All", "Water Gauge", "Number Object Detection", "Water Segmentation").
        -   If "All" is selected, configure water gauge height and all three confidence thresholds.
        -   If a specific model is selected, adjust its individual threshold.
        -   Click "Process Image" to perform the detection.

4.  **View Results:**
    -   For videos, the processed video with annotations and the average estimated water height will be displayed.
    -   For images, annotated images showing the detection results at different stages (if "All" models are selected) or for the specific chosen model will be shown, along with the estimated water height.

## Code Structure

The project is organized into several key files and directories:

-   `app.py`:
    -   The main Streamlit application file.
    -   Handles the user interface, file uploads (video/image).
    -   Manages user input for parameters like water gauge height and confidence thresholds.
    -   Orchestrates calls to `estimated_water_height_process.py` for actual detection and calculation.
    -   Displays the processed outputs.

-   `estimated_water_height_process.py`:
    -   Contains the core logic for water level estimation.
    -   Initializes and loads the YOLO models for water gauge, number, and water segmentation.
    -   Includes functions to:
        -   `get_y_coor_number`: Extract Y-coordinates of detected numbers.
        -   `get_number_bounding_box_height`: Get bounding box height of numbers.
        -   `get_roi_length`: Get length of water gauge region of interest.
        -   `get_water_height`: Get the Y-coordinate of the water segmentation's top.
        -   `get_pixel_scale`: Calculate the pixel-to-centimeter scale.
        -   `calculate_estimated_water_height`: Implement the main algorithm for water height calculation.
    -   `detect`: The primary function for processing videos frame by frame, performing detections, annotating frames, and saving the output video.

-   `estimate_test_dataset.py`:
    -   A standalone script used for testing and evaluating the water height estimation models on a predefined dataset of test frames.
    -   It iterates through frames, applies the detection models, calculates water heights, and saves the annotated results to a specified output folder.

-   `.env` (implied):
    -   Configuration file for environment variables.
    -   Stores paths to the YOLO models and default confidence thresholds for various detections.

-   `.gitignore`:
    -   Specifies files and directories that should be ignored by Git.
    -   Includes entries like `__pycache__/`, `dataset/`, `hasil/`, `models/`, and `.env` to prevent sensitive or generated files from being tracked.

-   `hasil/`:
    -   (Not explicitly in codebase, but implied by `.gitignore` and `app.py`)
    -   This directory is intended to store the output videos and images generated by the application after processing.

-   `models/`:
    -   This directory is expected to house the pre-trained YOLO model files (`.pt`) used for water gauge detection, number recognition, and water segmentation.

-   `dataset/`:
    -   (Implied by `.gitignore` and `estimate_test_dataset.py`)
    -   Likely contains the test frames and related data used for model training or evaluation.

## Result
![malam cerah_result_frame_23](https://github.com/user-attachments/assets/ac419e21-ab58-4af7-8615-1b45c6f8c258)


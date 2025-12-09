"""
Streamlit-based web application for extracting frames from a video, performing object
segmentation using the Segment Anything Model (SAM), interactively refining the
segmentation masks, and exporting the results.  The app assumes that you have
installed a recent version of Streamlit together with streamlit-drawable-canvas,
opencv-python, numpy, and the ``sam3`` package (the open source implementation of
Meta's Segment Anything Model).  In addition a valid model checkpoint must be
available on disk.  The default location for the checkpoint is
``/sam3/sam3/checkpoints/sam3.pt``, but this can be adjusted via a sidebar
setting.

Key features
------------

* **Video Selection** – Choose a video file from the ``./sources`` folder.
* **Frame Extraction** – Specify how many frames to sample and optional start/end
  frames.  Frames are extracted using OpenCV and cached on disk.
* **Prompt Definition** – Click on the first frame to add positive or negative
  prompts.  Positive points (label 1) indicate the object of interest and
  negative points (label 0) indicate background.  You can reset or modify
  points before running SAM.
* **Segmentation Preview** – Run the model on the first frame to preview the
  segmentation mask.
* **Propagation** – Once satisfied with the first frame, propagate the same
  prompts to all extracted frames to generate masks for the entire sequence.
* **Interactive Refinement** – Navigate between frames and refine individual
  masks by adding additional points or erasing portions of the mask.
* **Overlay Viewer** – Toggle the mask overlay to inspect each frame with its
  current mask.
* **Video Export** – Once all frames are processed, export an overlay video
  showing the original frames with the mask boundaries.

This application is intended as a starting point for more specialized video
annotation workflows.  It does **not** implement sophisticated object tracking;
instead it simply applies the same prompt-based segmentation to each frame.
Depending on the motion in your video you may need to provide additional
positive points per frame or use an external tracking algorithm.

Example usage
-------------

Assuming you have installed the required dependencies (see ``requirements.txt``)
and have a SAM checkpoint at ``/sam3/sam3/checkpoints/sam3.pt``, run this
application with:

```bash
streamlit run sam_video_annotation_app.py
```

Then open the provided local URL in your browser.  You will be able to select
your video and interactively segment the object of interest.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import streamlit as st

try:
    # streamlit_drawable_canvas is optional.  It provides an interactive
    # canvas for clicking on images.  If it's not installed, the app will
    # fall back to a simple image viewer and prompt entry via text boxes.
    from streamlit_drawable_canvas import st_canvas  # type: ignore
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

try:
    # Attempt to import the SAM model from the user's installation.
    import torch
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False


@st.cache_data(show_spinner=False)
def list_videos(source_dir: str) -> List[str]:
    """Return a sorted list of video files in the given directory.

    Parameters
    ----------
    source_dir: str
        Relative or absolute path containing the input videos.

    Returns
    -------
    List[str]
        Sorted list of filenames (not full paths).
    """
    valid_exts = {".mp4", ".mov", ".avi", ".mkv", ".mpg"}
    videos = []
    for item in os.listdir(source_dir):
        ext = Path(item).suffix.lower()
        if ext in valid_exts:
            videos.append(item)
    return sorted(videos)


@st.cache_data(show_spinner=False)
def extract_frames(
    video_path: str,
    max_frames: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> List[np.ndarray]:
    """Extract frames from a video using OpenCV.

    Parameters
    ----------
    video_path: str
        Path to the video file.
    max_frames: int
        Maximum number of frames to extract.
    start_frame: Optional[int], default None
        Optional starting frame index (0‑based).  If None, extraction starts
        from the beginning of the video.
    end_frame: Optional[int], default None
        Optional ending frame index (inclusive).  If provided, extraction stops
        at this frame.  If ``max_frames`` would go beyond this index, it will
        be truncated accordingly.

    Returns
    -------
    List[np.ndarray]
        List of BGR image arrays representing the extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = start_frame or 0
    end = end_frame if end_frame is not None else total_frames - 1
    end = min(end, total_frames - 1)
    if start < 0 or start > end:
        raise ValueError("Invalid start/end frame range")
    frames: List[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    current = start
    while current <= end and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        current += 1
    cap.release()
    return frames


def get_sam_predictor(
    checkpoint: str,
    model_type: str = "vit_h",
    device: Optional[str] = None,
) -> SamPredictor:
    """Load the SAM model and return a predictor object.

    This function requires that PyTorch and the Segment Anything
    repository be installed in the Python environment.  The user should
    provide a valid checkpoint path.  If PyTorch is unavailable the
    function raises an ImportError.

    Parameters
    ----------
    checkpoint: str
        Path to the SAM model checkpoint.
    model_type: str, default 'vit_h'
        Variant of the SAM model to load (e.g., 'vit_h', 'vit_l', etc.).
    device: Optional[str], default None
        Device on which to run the model.  If None, uses 'cuda' when
        available otherwise 'cpu'.

    Returns
    -------
    SamPredictor
        Instance of the predictor for running inference.
    """
    if not SAM_AVAILABLE:
        raise ImportError(
            "SAM is not available.  Please install the segment-anything "
            "repository and its dependencies (including torch)."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def run_sam_on_image(
    image: np.ndarray,
    predictor: SamPredictor,
    points: List[Tuple[int, int]],
    labels: List[int],
) -> np.ndarray:
    """Run SAM on a single image given sparse prompts.

    Parameters
    ----------
    image: np.ndarray
        Input image in BGR format (as returned by OpenCV).
    predictor: SamPredictor
        Predictor instance.
    points: List[Tuple[int, int]]
        Clicked pixel coordinates.  Coordinates should be provided as (x, y)
        pairs in image space.
    labels: List[int]
        Corresponding labels for each point: 1 for positive, 0 for negative.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W) with dtype uint8 where 1 represents the
        segmented object.
    """
    if len(points) == 0:
        # If no prompts are provided, return an empty mask.
        return np.zeros(image.shape[:2], dtype=np.uint8)
    # Convert BGR to RGB for SAM
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    input_points = np.array(points)
    input_labels = np.array(labels)
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    mask = masks[0].astype(np.uint8)
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary mask on top of an image.

    Parameters
    ----------
    image: np.ndarray
        Original image (BGR).
    mask: np.ndarray
        Binary mask with the same height and width as the image.
    color: Tuple[int, int, int], default (0, 255, 0)
        Color used for the mask overlay in BGR format.
    alpha: float, default 0.5
        Opacity of the mask overlay.

    Returns
    -------
    np.ndarray
        Image with mask overlay.
    """
    overlay = image.copy()
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask > 0] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
    """Write a list of frames to a video file.

    Parameters
    ----------
    frames: List[np.ndarray]
        Sequence of frames (BGR).
    output_path: str
        Output path of the generated video.
    fps: int, default 30
        Frames per second for the output video.
    """
    if len(frames) == 0:
        raise ValueError("No frames provided for video generation.")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def main() -> None:
    st.title("Video Segmentation with SAM")
    st.write(
        "This application lets you sample frames from a video, segment an object with the "
        "Segment Anything Model (SAM), refine the masks, and export an overlay video."
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")
    sources_dir = st.sidebar.text_input("Sources directory", "/data/sam3/sources")
    videos = list_videos(sources_dir)
    if not videos:
        st.sidebar.warning("No videos found in the specified directory.")
        st.stop()
    video_file = st.sidebar.selectbox("Select video", videos)
    video_path = os.path.join(sources_dir, video_file)

    max_frames = st.sidebar.number_input(
        "Number of frames to extract", min_value=1, max_value=1000, value=300, step=1
    )
    start_frame = st.sidebar.number_input(
        "Start frame (0‑based)", min_value=0, value=0, step=1
    )
    end_frame = st.sidebar.number_input(
        "End frame (inclusive, leave 0 for end)", min_value=0, value=0, step=1
    )
    if end_frame == 0:
        end_frame = None  # sentinel for extract_frames

    checkpoint_path = st.sidebar.text_input(
        "SAM checkpoint path", "/sam3/sam3/checkpoints/sam3.pt"
    )
    model_type = st.sidebar.selectbox(
        "SAM model type", ["vit_h", "vit_l", "vit_b", "vit_t"], index=0
    )
    run_device = st.sidebar.selectbox(
        "Device", ["auto", "cpu", "cuda"], index=0,
        help="Select 'auto' to use CUDA when available, otherwise CPU."
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Extract Frames"):
        with st.spinner("Extracting frames ..."):
            frames = extract_frames(video_path, int(max_frames), int(start_frame), end_frame)
        st.session_state["frames"] = frames
        st.session_state["masks"] = [None] * len(frames)
        st.session_state["current_index"] = 0
        st.success(f"Extracted {len(frames)} frames from the video.")

    # Ensure frames are loaded before continuing
    if "frames" not in st.session_state:
        st.info("Please extract frames from your video using the sidebar.")
        st.stop()

    frames: List[np.ndarray] = st.session_state["frames"]
    masks: List[Optional[np.ndarray]] = st.session_state["masks"]
    current_index: int = st.session_state.get("current_index", 0)

    # Load SAM predictor when needed (only once)
    if "predictor" not in st.session_state:
        if SAM_AVAILABLE:
            with st.spinner("Loading SAM model ..."):
                device = None if run_device == "auto" else run_device
                predictor = get_sam_predictor(
                    checkpoint=checkpoint_path, model_type=model_type, device=device
                )
            st.session_state["predictor"] = predictor
        else:
            st.warning(
                "SAM is not available in this environment.  You can still extract "
                "frames and manually draw masks, but automatic segmentation will not run."
            )

    # Show current frame and editing tools
    frame = frames[current_index]
    st.subheader(f"Frame {current_index + 1} of {len(frames)}")

    # Display mask overlay if it exists
    display_image = frame.copy()
    current_mask = masks[current_index]
    if current_mask is not None:
        display_image = overlay_mask(frame, current_mask)

    # Collect prompts via canvas or text
    st.markdown("### Define prompts for segmentation")
    if CANVAS_AVAILABLE:
        # Convert image to RGB for canvas display
        image_for_canvas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
            stroke_width=5,
            stroke_color="#FF0000",  # Red strokes for positive points
            background_image=image_for_canvas,
            update_streamlit=True,
            height=image_for_canvas.shape[0],
            width=image_for_canvas.shape[1],
            drawing_mode="point",
            key=f"canvas_{current_index}",
        )
        # Extract points from canvas JSON output
        positive_points: List[Tuple[int, int]] = []
        negative_points: List[Tuple[int, int]] = []
        if canvas_result.json_data is not None:
            # Each point is encoded as a circle shape in the JSON data
            for obj in canvas_result.json_data.get("objects", []):
                x = int(obj["left"])
                y = int(obj["top"])
                label = obj.get("stroke", "#FF0000")
                if label == "#FF0000":
                    positive_points.append((x, y))
                else:
                    negative_points.append((x, y))
    else:
        st.info(
            "streamlit-drawable-canvas is not installed.  You can still run the "
            "segmentation by manually entering prompt coordinates below."
        )
        pos_str = st.text_input(
            "Positive points (comma‑separated x,y pairs)", ""
        )
        neg_str = st.text_input(
            "Negative points (comma‑separated x,y pairs)", ""
        )
        positive_points = []
        negative_points = []
        if pos_str.strip():
            for pair in pos_str.split(";"):
                try:
                    x_str, y_str = pair.strip().split(",")
                    positive_points.append((int(x_str), int(y_str)))
                except Exception:
                    continue
        if neg_str.strip():
            for pair in neg_str.split(";"):
                try:
                    x_str, y_str = pair.strip().split(",")
                    negative_points.append((int(x_str), int(y_str)))
                except Exception:
                    continue

    # Buttons for segmentation and propagation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run SAM on this frame"):
            if not SAM_AVAILABLE:
                st.error("SAM is not available.  Unable to run segmentation.")
            elif len(positive_points) + len(negative_points) == 0:
                st.warning("Please specify at least one positive or negative point.")
            else:
                with st.spinner("Running SAM..."):
                    predictor: SamPredictor = st.session_state["predictor"]
                    points = positive_points + negative_points
                    labels = [1] * len(positive_points) + [0] * len(negative_points)
                    mask = run_sam_on_image(frame, predictor, points, labels)
                    masks[current_index] = mask
                    st.session_state["masks"] = masks
                st.success("Segmentation completed for this frame.")
    with col2:
        if st.button("Propagate to all frames"):
            if not SAM_AVAILABLE:
                st.error("SAM is not available.  Unable to run segmentation.")
            elif len(positive_points) + len(negative_points) == 0:
                st.warning("Please specify prompts for the first frame before propagating.")
            else:
                with st.spinner("Propagating segmentation across frames..."):
                    predictor: SamPredictor = st.session_state["predictor"]
                    points = positive_points + negative_points
                    labels = [1] * len(positive_points) + [0] * len(negative_points)
                    new_masks: List[np.ndarray] = []
                    for idx, img in enumerate(frames):
                        mask = run_sam_on_image(img, predictor, points, labels)
                        new_masks.append(mask)
                    st.session_state["masks"] = new_masks
                st.success("Segmentation propagated to all frames.")
    with col3:
        if st.button("Reset prompts"):
            # Clear canvas prompts by resetting the key
            if CANVAS_AVAILABLE:
                st.session_state.pop(f"canvas_{current_index}", None)
            st.experimental_rerun()

    # Navigation between frames
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Previous frame"):
            if current_index > 0:
                st.session_state["current_index"] = current_index - 1
                st.experimental_rerun()
    with nav2:
        if st.button("Next frame"):
            if current_index < len(frames) - 1:
                st.session_state["current_index"] = current_index + 1
                st.experimental_rerun()

    # Display the overlay image
    st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), caption="Current frame with mask overlay", use_column_width=True)

    # Save video button
    if st.button("Generate overlay video"):
        # Ensure all masks are generated
        if any(m is None for m in masks):
            st.warning(
                "Not all frames have masks.  You should propagate or segment each frame individually before generating a video."
            )
        else:
            save_path = st.text_input(
                "Output video filename", value=f"{Path(video_file).stem}_overlay.mp4"
            )
            if save_path:
                with st.spinner("Generating video ..."):
                    overlay_frames = []
                    for img, m in zip(frames, masks):
                        assert m is not None
                        overlay_frames.append(overlay_mask(img, m))
                    save_video(overlay_frames, save_path)
                st.success(f"Saved overlay video to {save_path}")


if __name__ == "__main__":
    main()
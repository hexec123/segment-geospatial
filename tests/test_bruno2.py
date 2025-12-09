# %pip install "segment-geospatial[samgeo3]"

import os
import pytest
import torch
from samgeo import SamGeo3Video, download_file

# This example requires CUDA-enabled GPUs because SAM3's video predictor uses
# multi-GPU collective initialization (NCCL). If CUDA is not available in the
# test environment (CI or developer machine), skip the example to avoid
# "Default process group has not been initialized" errors.
if not torch.cuda.is_available():
	pytest.skip("Skipping SAM3 video example: CUDA is not available in this environment")

sam = SamGeo3Video()

url = "https://github.com/opengeos/datasets/releases/download/videos/cars.mp4"
video_path = download_file(url)

sam.set_video(video_path)

sam.show_video(video_path)

# Segment all car in the video
sam.generate_masks("car")

# Show the first frame with masks
sam.show_frame(0, axis="on")

# Show multiple frames in a grid
sam.show_frames(frame_stride=20, ncols=3)


import numpy as np
import pytest

from samgeo.samgeo3 import SamGeo3Video


class MockPredictor:
    def __init__(self):
        self.stream_called = False
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        t = request.get("type")
        if t == "start_session":
            return {"session_id": "neg"}
        if t == "add_prompt":
            # Return a simple add_prompt-style response
            return {
                "outputs": {
                    "object_ids": [1],
                    "masks": [np.ones((4, 4), dtype=np.uint8)],
                }
            }
        if t == "reset_session":
            return {}
        if t == "close_session":
            return {}
        return {}

    def handle_stream_request(self, request):
        # Mark that a propagation stream was requested; yield no items
        self.stream_called = True
        if False:
            yield None


def make_video_obj():
    """Create a SamGeo3Video-like instance without calling __init__.

    This avoids importing or initializing heavy SAM3 machinery in tests.
    """
    obj = object.__new__(SamGeo3Video)
    obj.predictor = MockPredictor()
    obj.session_id = "sess"
    obj.video_frames = ["/tmp/frame0.png"]
    obj.video_path = "/tmp"
    obj._frame_cache = {}
    obj.outputs_per_frame = None
    obj.frame_width = 10
    obj.frame_height = 10
    return obj


def test_generate_masks_no_propagate_only_add_prompt_called():
    sam = make_video_obj()

    out = sam.generate_masks("test-object", frame_idx=0, propagate=False)

    # Ensure stream was not called (no propagation)
    assert sam.predictor.stream_called is False

    # outputs_per_frame should be set and only contain the single frame
    assert isinstance(out, dict)
    assert 0 in out


def test_negative_prompt_propagate_false_uses_only_add_prompt():
    sam = make_video_obj()

    out = sam.generate_masks(
        "pos-object", frame_idx=0, propagate=False, negative_prompt="neg"
    )

    # Negative propagation should not trigger stream requests
    assert sam.predictor.stream_called is False

    # outputs_per_frame should be a dict (possibly cleaned)
    assert isinstance(out, dict)
    # It should contain the frame 0 key (single-frame behavior)
    assert 0 in out

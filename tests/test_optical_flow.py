import pytest as pt
import numpy as np
from neuflowv2 import OpticalFlow
import os

def test_optical_flow():
    of = OpticalFlow(model_path="models/neuflow_sintel.onnx")

    height, width = 5, 5
    image0 = np.zeros((height, width, 3), dtype=np.float32)
    image1 = np.zeros((height, width, 3), dtype=np.float32)

    # Create a simple pattern: a 2x2 block of ones
    image0[1:3, 1:3, :] = 1.0
    # Shift the block right by 1 pixel
    image1[1:3, 2:4, :] = 1.0

    result = of.update(image0)
    assert result == 0

    result = of.update(image1)
    ms_sum, ms_mean, ms_median = of.compute_movement_scores(result)
    assert 3712.3428 == pt.approx(ms_sum)
    assert 148.49371 == pt.approx(ms_mean)
    assert 153.50700 == pt.approx(ms_median)

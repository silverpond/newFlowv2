import unittest
import cv2
import numpy as np
from neuflowv2 import OpticalFlow
import os

class TestOpticalFlow(unittest.TestCase):
    def setUp(self):
        # Initialize the OpticalFlow class with default parameters
        self.optical_flow = OpticalFlow(model_path="models/neuflow_sintel.onnx")
        # Path to the test video
        self.test_video_path = "vendor/optical_flow_measure/inputs/test.mp4"

    def test_optical_flow_initialization(self):
        # Test that the OpticalFlow class initializes correctly
        self.assertIsNotNone(self.optical_flow)
        self.assertIsNotNone(self.optical_flow.estimator)
        
    def test_reset_method(self):
        # Test that the reset method works correctly
        # First update with an image
        cap = cv2.VideoCapture(self.test_video_path)
        ret, frame = cap.read()
        cap.release()
     
        self.optical_flow.update(frame) # update the first frame
        self.assertIsNotNone(self.optical_flow.prev_frame)
         
        # Now reset and check prev_frame is None
        self.optical_flow.reset()
        self.assertIsNone(self.optical_flow.prev_frame)
        
    def test_optical_flow_calculation(self):
        # Test that the optical flow calculation produces expected results
        cap = cv2.VideoCapture(self.test_video_path)
        
        # Read first frame
        ret, frame1 = cap.read() 
        self.assertTrue(ret, "Failed to read first frame")
        
        # First update should return 0 since there will be  no previous frame
        score1 = self.optical_flow.update(frame1)
        self.assertEqual(score1, 0.0, "First frame should return a score of 0.0")
        
        # Read second frame
        ret, frame2 = cap.read()
        self.assertTrue(ret, "Failed to read second frame")
        
        # Second update should return a non-zero score
        score2 = self.optical_flow.update(frame2)
        self.assertGreater(score2, 0.0, "Second frame should return a non-zero score")
        
        # Read third frame
        ret, frame3 = cap.read()
        self.assertTrue(ret, "Failed to read third frame")
        
        # Third update should also return a non-zero score
        score3 = self.optical_flow.update(frame3)
        self.assertGreater(score3, 0.0, "Third frame should return a non-zero score")
        
        # Check the expected output
        
        # Clean up
        cap.release()
        self.optical_flow.reset()
        
    def test_identical_frames(self):
        # Test with identical frames - should return very low movement score
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # expected to be gray image
        
        # First update should return 0
        score1 = self.optical_flow.update(test_image)
        self.assertEqual(score1, 0.0, "First frame should return a score of 0.0")
        
        # Second update with identical frame should return very low score
        score2 = self.optical_flow.update(test_image.copy())
        self.assertAlmostEqual(score2, 0.0, delta=1.0, 
                              msg="Identical frames should return a score close to 0.0")
        
if __name__ == "__main__":
    unittest.main()



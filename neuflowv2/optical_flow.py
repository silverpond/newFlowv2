from typing import Tuple
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from .neuflowv2 import NeuFlowV2

class OpticalFlow:
    """
    A standalone optical flow class that calculates a movement score 
    based on optical flow between consecutive frames.
    """
    
    def __init__(self, model_path:str="neuflow_sintel.onnx", use_gpu:bool=True, blur_kernel:tuple=(3, 3)) -> None:
        """
        Initialize the OpticalFlow class.
        
        Args:
            model_path (str): Path to the optical flow model file.
            use_gpu (bool): Whether to use GPU for preprocessing.
            blur_kernel (tuple): Size of Gaussian blur kernel for preprocessing.
        Returns:
            None
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.blur_kernel = blur_kernel  
        self.estimator = self._load_model()  #load the model from the model_path
        self.prev_frame = None
        
        # Image preprocessing transform pipeline
        self.preprocess_frame = T.Compose([
            T.ToTensor(), # Convert image to tensor
            T.ConvertImageDtype(torch.float32), # Convert image to float32 
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize image
        ])
        
        print(f"Initialized OpticalFlow on device: {self.device}")
    
    def _load_model(self) -> NeuFlowV2:
        """
        Load the NeuFlowV2 optical flow model.
        
        Returns:
            The loaded NeuFlowV2 model instance.
        """
        return NeuFlowV2(self.model_path)
    
    def update(self, image) -> float:
        """
        Update the algorithm with a new image and return the latest movement score.
        
        The movement score is calculated as the sum of the magnitudes of all optical 
        flow vectors between the current image and the previous image.
        
        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV).
            
        Returns:
            float: Movement score indicating the amount of motion between frames.
        """
        # Apply Gaussian blur to reduce noise
        filtered_frame = cv2.GaussianBlur(image, self.blur_kernel, 0)
        
        # Assume RGB
        # frame_tensor is CHW
        frame_tensor = self.preprocess_frame(filtered_frame).to(self.device)

        # If this is the first frame, store it and return 0
        if self.prev_frame is None:
            self.prev_frame = frame_tensor
            return 0.0  # Return 0 if it's the first frame 
        
        # Convert tensors back to numpy for OpticalFlow estimation
        # convert from CHW back to HWC
        prev_frame_np = (self.prev_frame.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        curr_frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        
        # Calculate optical flow
        flow_vectors = self.estimator(prev_frame_np, curr_frame_np) 
        
        # Store current frame for next comparison
        self.prev_frame = frame_tensor
        
        return flow_vectors
    
    
    def compute_movement_scores(self, flow_vectors:np.ndarray) -> Tuple[float, float, float]:
        """Computes the movement score from the flow vectors"""
        u, v = flow_vectors[..., 0], flow_vectors[..., 1]
        magnitude = np.sqrt(u**2 + v**2)
        movement_sum = np.sum(magnitude)
        movement_mean = np.mean(magnitude)
        movement_median = np.median(magnitude)

        return movement_sum, movement_mean, movement_median


    def reset(self):
        """
        Reset the optical flow state.
        """
        self.prev_frame = None 


    @staticmethod
    def update_flow_vector_overlay():
        pass
     
     

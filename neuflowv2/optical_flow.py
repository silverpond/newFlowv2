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
        
        # Convert BGR to RGB and preprocess
        frame_tensor = self.preprocess_frame(
            cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
        ).to(self.device)
        
        # If this is the first frame, store it and return 0
        if self.prev_frame is None:
            self.prev_frame = frame_tensor
            return 0.0  # Return 0 if it's the first frame 
        
        # Convert tensors back to numpy for OpticalFlow estimation
        prev_frame_np = (self.prev_frame.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        curr_frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        
        # Calculate optical flow
        flow = self.estimator(prev_frame_np, curr_frame_np)
        
        # Calculate magnitude of flow vectors
        u, v = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(u**2 + v**2)
        
        # Calculate movement score (sum of all magnitudes)
        movement_score = np.sum(magnitude)
        
        # Store current frame for next comparison
        self.prev_frame = frame_tensor
        
        return movement_score
    
    def reset(self):
        """
        Reset the optical flow state.
        """
        self.prev_frame = None 
        
     
     
     
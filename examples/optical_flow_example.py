import cv2
import numpy as np
from neuflowv2 import OpticalFlow

def main():
    # Initialize the OpticalFlow class
    # Change the model path as needed for your environment
    flow = OpticalFlow(model_path="models/neuflow_sintel.onnx")
    
    # Open a video capture (0 for webcam, or a video file path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    print("Press 'q' to quit...")
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        
        # Calculate movement score
        movement_score = flow.update(frame)
        
        # Display the movement score on the frame
        cv2.putText(
            frame, 
            f"Movement: {movement_score:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Display the frame
        cv2.imshow("Optical Flow Movement", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
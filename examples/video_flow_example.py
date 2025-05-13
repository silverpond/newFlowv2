import cv2
import numpy as np
from neuflowv2 import OpticalFlow

def main():
    # Initialize the OpticalFlow class
    # Change the model path as needed for your environment
    flow = OpticalFlow(model_path="models/neuflow_sintel.onnx")
    
    # Path to the input video file
    video_path = "vendor/optical_flow_measure/inputs/test.mp4"
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video loaded: {frame_width}x{frame_height} at {fps} FPS")
    print("Press 'q' to quit, 'p' to pause/resume...")
    
    paused = False
    
    while True:
        if not paused:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            
            # Calculate movement score
            movement_score = flow.update(frame)
            
            # Print the movement score to the console
            print(f"Movement score: {movement_score:.2f}")
            
     #        # Display the movement score on the frame
     #        cv2.putText(
     #            frame, 
     #            f"Movement: {movement_score:.2f}", 
     #            (10, 30), 
     #            cv2.FONT_HERSHEY_SIMPLEX, 
     #            1, 
     #            (0, 255, 0), 
     #            2
     #        )
        
     #    # Display the frame
     #    cv2.imshow("Optical Flow Movement", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Video paused" if paused else "Video resumed")
    
    # Release resources
    cap.release()
#     cv2.destroyAllWindows()
    flow.reset()

if __name__ == "__main__":
    main() 
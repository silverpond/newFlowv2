import time
import cv2
import numpy as np
from neuflowv2 import OpticalFlow
import json

def main():
    # Initialize the OpticalFlow class
    # Change the model path as needed for your environment
    flow = OpticalFlow(model_path="models/neuflow_sintel.onnx")
    
    #movement score
    movement_scores: list[float] = []
    
    
    # Path to the input video file
    video_path = "vendor/optical_flow_measure/inputs/cold-chickens-combined.mp4"
    
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
    
    # list to store all the flow vectors and movement scores and inference times 
    frame_data: list[dict] = []
    
    while True:
        if not paused:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            #time before inference
            start = time.perf_counter()
            
            # Calculate flow vector
            flow_vectors = flow.update(frame)
            
            #inference time for model to compute the flow vector of two consecutive frames
            inference_time = (time.perf_counter() - start) * 1000 #milisecs
            
            # Print the flow vector to the console
            print(f"Flow vector: {flow_vectors}")
            
            # Calculate movement score
            # check if the flow vector is a numpy array
            if isinstance(flow_vectors, np.ndarray):
                movement_score = flow.compute_movement_score(flow_vectors)
            else: 
                # the situation where the first frame then no prev_frame then no movement score or sometime invalid value
                movement_score = 0
                      
            print(f"Movement score: {movement_score}")
            movement_scores.append(float(movement_score))       
            
            #Store data for each process
            data = {
                "inference_time": inference_time,
                "flow_vectors": flow_vectors.tolist(),
                "movement_scores": movement_scores
            }
            
            #save data to the list
            frame_data.append(data)
            
            
     #        # Display the flow vector on the frame
     #        cv2.putText(
     #            frame, 
     #            f"Flow vector: {flow_vectors}", 
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
            
    #save data to json file
    with open("flow_vectors_cold_chicken.json", "w") as f:
        json.dump(frame_data, f)
    
    # Release resources
    cap.release()
    #cv2.destroyAllWindows()
    flow.reset()

if __name__ == "__main__":
    main() 
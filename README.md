# NeuFlow v2: Dockerized Video Processing Code

This repository provides a Dockerized solution for running the NeuFlow v2 video processing pipeline. Follow the instructions below to set up and run the pipeline.

## Getting Started
### Step 1: Clone the repository
```bash
git clone https://github.com/dmanzanoa/newFlowv2.git
```
### Step 2: Build the Docker Image inside the repository
Execute the following command to build the Docker image:

```bash
docker build -t neuflowv2 .
```
### Step 3: Run the Docker Container

Once the image is built, run the container using:

```bash
docker run --gpus all -it neuflowv2
```
### Step 4: Move to vendor/optical_flow_measure/ folder

```bash
cd vendor/optical_flow_measure/
```

### Step 5: Excute the agent

```bash
hl agent run agents/OpticalFlowAgent.json -f inputs/test.mp4
```

## Using the OpticalFlow Class

The repository includes a standalone `OpticalFlow` class that calculates movement scores from video frames.

### Basic Usage

```python
from neuflowv2 import OpticalFlow

# Initialize with default parameters
flow = OpticalFlow(model_path="models/neuflow_sintel.onnx")

# Process a single frame and get movement score
frame = your_image_processing_function()  # OpenCV BGR image
movement_score = flow.update(frame)

# Reset internal state if needed
flow.reset()
```

### Example

Run the included example script to see the movement score calculated from webcam input:

```bash
python examples/optical_flow_example.py
```

See the `examples/optical_flow_example.py` file for a complete implementation.

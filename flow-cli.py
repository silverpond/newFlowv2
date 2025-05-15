import click
from neuflowv2 import OpticalFlow
from highlighter.agent.capabilities.sources import VideoFrameIterator
import json

DEFAULT_MODEL_PATH = "models/neuflow_sintel.onnx"

@click.group()
def flow_group():
    pass

@flow_group.command("compute")
@click.argument("video_path", type=str)
@click.option("--model-path", type=str, required=False, default=DEFAULT_MODEL_PATH)
@click.option("--fps", type=int, required=False, default=0)
@click.option("--max-frames", type=int, required=False, default=0)
def compute(video_path, model_path, max_frames, fps):
    if not max_frames:
        max_frames = float("inf")

    of = OpticalFlow(model_path=model_path)
    video_frames = VideoFrameIterator(
            source_urls=[video_path]
            )
    prev_frame = next(video_frames).content
    of.update(prev_frame)

    frame_number = 0
    scores: list[dict] = []
    while True:
        try:
            cur_frame = next(video_frames).content
        except StopIteration:
            print(f"End of video, frame: {frame_number}")
            break

        flow = of.update(cur_frame)
        f_sum, f_mean, f_median = of.compute_movement_scores(flow)
        move_score = {
                "sum": float(f_sum),
                "mean": float(f_mean),
                "median": float(f_median)
                }

        scores.append(move_score)

        if not (frame_number % 10):
            print(f"Sum:{int(f_sum)}, Mean:{f_mean:0.2f}, Med:{f_median:0.2f}")

        prev_frame = cur_frame
        frame_number += 1

        if frame_number >= max_frames:
            print(f"Exiting at max_frames: {max_frames}")
            break

    with open("flow_vector.json", "w") as f:
        json.dump(scores, f, indent=4)

@flow_group.command("overlay")
def overlay():
    pass


if __name__ == "__main__":
    flow_group()


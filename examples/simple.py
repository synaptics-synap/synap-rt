"""A simple object detection pipeline example."""

import json
import sys

from synapRT.pipelines import pipeline


def handle_inference_result(results, inference_time=None):
    # simple handler that writes the results to stdout
    sys.stdout.write("\033[H\033[J")
    formatted_json = json.dumps(results, indent=4)
    print(formatted_json, flush=True)
    sys.stdout.flush()
    if inference_time:
        print(f"Avg. inference time: {inference_time} ms", flush=True)
    sys.stdout.flush()

# define pipeline
pipe = pipeline(
    task="object-detection",
    model="/usr/share/synap/models/object_detection/coco/model/yolov8s-640x384/model.synap",
    profile=True, # enable inference time profiling
    handler=handle_inference_result,
)

# run pipeline with inputs
pipe(sys.argv[1])

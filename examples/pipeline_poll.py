import argparse
import json
import time
import sys
import threading

from synapRT.pipelines import pipeline


def handle_inference_result(results, inference_time=None):
    sys.stdout.write("\033[H\033[J")
    formatted_json = json.dumps(results, indent=4)
    print(formatted_json, flush=True)
    sys.stdout.flush()
    if inference_time:
        print(f"Avg. inference time: {inference_time} ms", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SyNAP inference pipeline")
    parser.add_argument(
        "-t", "--task",
        type=str,
        required=True,
        choices=["image-classification", "object-detection"],
        help="Task to perform"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to SyNAP model file"
    )
    parser.add_argument(
        "-i", "--inputs",
        type=str,
        required=True,
        nargs="+",
        help="Input sources for the pipeline"
    )
    parser.add_argument(
        "--iter-results",
        action="store_true",
        help="Iterate over inference results instead of polling"
    )
    parser.add_argument(
        "--no-threading",
        action="store_true",
        help="Run the pipeline in the main thread (not recommended for video input)"
    )
    parser.add_argument(
        "--infer-params",
        type=str,
        nargs="*",
        default=[],
        help="Additional inference arguments"
    )
    args = parser.parse_args()

    infer_params = {}
    for arg in args.infer_params:
        key, value = arg.split("=", 1)
        infer_params[key] = value

    pipe = pipeline(
        task=args.task,
        model=args.model,
        **infer_params
    )

    pipe_thread = threading.Thread(target=pipe, args=args.inputs)
    pipe_thread.start()

    try:
        if args.iter_results:
            for _, res in pipe:
                if res:
                    handle_inference_result(res, pipe.inference_time)
                time.sleep(0.001) # for stability
        else:
            while True:
                _, res = pipe.poll()
                if res:
                    handle_inference_result(res, pipe.inference_time)
                if pipe.error:
                    raise pipe.error
                if pipe.finished:
                    break
                time.sleep(0.001) # for stability
    except KeyboardInterrupt:
        sys.exit(0)

import subprocess
from pprint import pprint

import ffmpeg
import numpy as np
from PIL import Image


def run_nvidia_smi():
    result = (
        subprocess.Popen("nvidia-smi", shell=True, stdout=subprocess.PIPE)
        .stdout.read()
        .decode("unicode_escape")
    )
    print(result)


def load_frames(video_path, iframes_only=True, fps=1):

    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        # pprint(video_stream)

        if iframes_only:
            out, _ = (
                ffmpeg.input(video_path)
                .filter_("select", "eq(pict_type,I)")
                .output("pipe:", format="rawvideo", pix_fmt="rgb24", vsync="vfr")
                .global_args("-hide_banner")
                .global_args("-loglevel", "error")
                .run(capture_stdout=True)
            )
        else:
            out, _ = (
                ffmpeg.input(video_path)
                .filter("fps", fps=fps, round="up")
                .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .global_args("-hide_banner")
                .global_args("-loglevel", "error")
                .run(capture_stdout=True)
            )

        contents = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        frames = [Image.fromarray(contents[i]) for i in range(contents.shape[0])]

        return frames

    except Exception as e:
        print("load_frames: Could not load video", video_path)
        return False


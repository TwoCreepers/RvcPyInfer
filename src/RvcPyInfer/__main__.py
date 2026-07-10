import time

import soundfile
import numpy as np
from .audio.audio_utils import split_by_silence, print_segments_info

from .RvcContext import RvcContext
from .InferProviders import InferProviders

def test():

    wav, sr = soundfile.read("./input/#2_小幸运_洛天依-主1.wav", dtype="float32")
    if len(wav.shape) != 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    s = split_by_silence(
        (wav, sr)
    )
    print(len(s))
    print_segments_info(s)

    exit()

    start = time.time()

    p = InferProviders.ORT_CPU
    print(p)
    context = RvcContext(p)
    task = context.build_task(
        "./models/vec-768-layer-12-sim.onnx",
        "./models/illue.onnx",
        48000,
        "./input/#2_小幸运_洛天依-主1.wav"
    )
    task.run_and_save(
        "./output/test.wav",
        subtype="FLOAT"
    )

    print(time.time() - start)

if __name__ == "__main__":
    test()
"""Decode ceilings: how many 1080p source-fps can the box decode at all?

Measures (each for --seconds of wall time, all workers concurrent):
  cpu  : P OpenCV(FFmpeg) decoder processes, P in --cpu-procs
  nvdec: K PyNvVideoCodec decoders (threads, one GPU), K in --gpu-decoders

Result rows go to $OUTPUT_ROOT/decode_probe.jsonl; headline scalars merge into metrics.json.
Real-time streams supported by decode alone = aggregate_fps / source_fps.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import threading
import time

from common import (CpuSampler, GpuSampler, append_jsonl, find_videos, merge_metrics_json,
                    output_root, probe_video)


# ---------------- CPU (OpenCV / FFmpeg) ----------------
def _cpu_worker(path: str, seconds: float, barrier, q, threads: int):
    import cv2
    cv2.setNumThreads(1)
    params = [cv2.CAP_PROP_N_THREADS, threads] if hasattr(cv2, "CAP_PROP_N_THREADS") else []
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(path)
    cap.read()
    barrier.wait()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok = cap.grab()  # full decode, skip BGR conversion (same as vid_stride skips)
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(path)
            continue
        n += 1
    q.put(n / (time.perf_counter() - t0))


def run_cpu(videos, procs: int, seconds: float, threads: int) -> dict:
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(procs), ctx.Queue()
    ps = [ctx.Process(target=_cpu_worker, args=(videos[i % len(videos)], seconds, barrier, q, threads))
          for i in range(procs)]
    for p in ps:
        p.start()
    with GpuSampler() as g, CpuSampler() as c:
        fps = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return {"backend": "cpu_opencv", "workers": procs, "threads_per_worker": threads,
            "aggregate_fps": sum(fps), "per_worker_fps_min": min(fps), **g.summary(), **c.summary()}


# ---------------- NVDEC (PyNvVideoCodec) ----------------
def run_nvdec(videos, decoders: int, seconds: float, batch: int) -> dict:
    import PyNvVideoCodec as nvc

    counts = [0] * decoders
    start = threading.Event()
    stop = threading.Event()
    errors: list[str] = []

    def make(path):
        return nvc.SimpleDecoder(path, gpu_id=0, use_device_memory=True,
                                 output_color_type=nvc.OutputColorType.RGBP)

    def worker(i):
        try:
            path = videos[i % len(videos)]
            dec = make(path)
            dec.get_batch_frames(1)
            start.wait()
            while not stop.is_set():
                frames = dec.get_batch_frames(batch)
                if not frames:
                    dec = make(path)
                    continue
                counts[i] += len(frames)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(decoders)]
    for t in ts:
        t.start()
    time.sleep(3)  # let decoders initialise
    with GpuSampler() as g, CpuSampler() as c:
        base = list(counts)
        start.set()
        t0 = time.perf_counter()
        time.sleep(seconds)
        got = [a - b for a, b in zip(counts, base)]
        dt = time.perf_counter() - t0
        stop.set()
    for t in ts:
        t.join(timeout=10)
    row = {"backend": "nvdec_pynvvideocodec", "workers": decoders, "batch": batch,
           "aggregate_fps": sum(got) / dt, "per_worker_fps_min": min(got) / dt,
           **g.summary(), **c.summary()}
    if errors:
        row["errors"] = errors[:3]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="glob; default $DATA_ROOT/videos/**")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--cpu-procs", default="1,4,8,16,0", help="0 = os.cpu_count()")
    ap.add_argument("--cpu-threads", type=int, default=1, help="ffmpeg threads per decoder")
    ap.add_argument("--gpu-decoders", default="1,2,4,8,16,32")
    ap.add_argument("--gpu-batch", type=int, default=8)
    args = ap.parse_args()

    import os
    out = output_root()
    videos = find_videos(args.videos)
    for v in videos:
        info = probe_video(v)
        print("VIDEO", v, info, flush=True)
        append_jsonl(out / "videos.jsonl", {"path": v, **info})
    src_fps = probe_video(videos[0]).get("fps") or 30.0

    best = {}
    for p in [int(x) for x in args.cpu_procs.split(",")]:
        p = p or os.cpu_count()
        row = run_cpu(videos, p, args.seconds, args.cpu_threads)
        row["realtime_streams"] = row["aggregate_fps"] / src_fps
        print("DECODE", row, flush=True)
        append_jsonl(out / "decode_probe.jsonl", row)
        best["decode_cpu_max_fps"] = max(best.get("decode_cpu_max_fps", 0), row["aggregate_fps"])

    for k in [int(x) for x in args.gpu_decoders.split(",")]:
        try:
            row = run_nvdec(videos, k, args.seconds, args.gpu_batch)
        except Exception as e:  # noqa: BLE001
            row = {"backend": "nvdec_pynvvideocodec", "workers": k, "error": repr(e)}
            print("DECODE", row, flush=True)
            append_jsonl(out / "decode_probe.jsonl", row)
            break
        row["realtime_streams"] = row["aggregate_fps"] / src_fps
        print("DECODE", row, flush=True)
        append_jsonl(out / "decode_probe.jsonl", row)
        best["decode_nvdec_max_fps"] = max(best.get("decode_nvdec_max_fps", 0), row["aggregate_fps"])

    best["source_fps"] = src_fps
    for k in ("decode_cpu_max_fps", "decode_nvdec_max_fps"):
        if k in best:
            best[k.replace("_max_fps", "_max_rt_streams")] = best[k] / src_fps
    merge_metrics_json(out, best)
    print("SUMMARY", best, flush=True)


if __name__ == "__main__":
    main()

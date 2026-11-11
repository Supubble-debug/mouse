"""L0 baseline: N independent processes, each = OpenCV decode + ultralytics model.track (ByteTrack).

Simulated real-time multi-stream from offline files: N streams round-robin over the
videos with staggered start offsets, all started together behind a barrier and run
unpaced for --seconds. A stream keeps up with real time iff it consumes source frames
at >= source fps (RTF >= 1). Frame-rate reduction = --strides (k=1 -> 30fps, 2 -> 15,
3 -> 10): skipped frames are still decoded (grab), only inference is skipped.

supported(N, k) := min over streams of RTF >= --rtf-ok (default 1.0)
Headline: l0_max_streams_<fps>fps for every stride.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

from common import (CpuSampler, GpuSampler, MetricSeries, append_jsonl, find_videos,
                    merge_metrics_json, output_root, probe_video)

TRACKER_YAML = Path(__file__).with_name("bytetrack_bench.yaml")


def _worker(idx, path, offset, stride, a, barrier, q):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    import torch
    from ultralytics import YOLO

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    model = YOLO(a["model"], task="segment")
    kw = dict(persist=True, tracker=str(TRACKER_YAML), imgsz=a["imgsz"], half=a["half"],
              conf=a["conf"], max_det=a["max_det"], retina_masks=a["retina_masks"],
              verbose=False, device=0)

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
    ok, frame = cap.read()
    for _ in range(10):  # warm-up (cuDNN autotune, tracker init)
        model.track(frame, **kw)
    torch.cuda.synchronize()
    barrier.wait()

    src, inf, ndet = 0, 0, 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a["seconds"]:
        if src % stride == 0:
            ok, frame = cap.read()
        else:
            ok = cap.grab()
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(path)
            continue
        if src % stride == 0:
            r = model.track(frame, **kw)[0]
            ndet += len(r.boxes)
            inf += 1
        src += 1
    dt = time.perf_counter() - t0
    q.put({"idx": idx, "src_fps": src / dt, "infer_fps": inf / dt,
           "dets_per_frame": ndet / max(inf, 1)})


def run_point(videos, n, stride, a, nframes_per_video):
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(n), ctx.Queue()
    ps = []
    for i in range(n):
        v = i % len(videos)
        # stagger so streams sharing a file are not frame-synchronous
        off = (i // len(videos)) * 997 % max(nframes_per_video[v] - 1000, 1)
        ps.append(ctx.Process(target=_worker, args=(i, videos[v], off, stride, a, barrier, q)))
    for p in ps:
        p.start()
    with GpuSampler() as g, CpuSampler() as c:
        rows = [q.get(timeout=a["seconds"] + 600) for _ in ps]
    for p in ps:
        p.join()
    src = [r["src_fps"] for r in rows]
    return {"pipeline": "l0_ultralytics", "streams": n, "stride": stride,
            "src_fps_min": min(src), "src_fps_mean": sum(src) / n, "agg_src_fps": sum(src),
            "agg_infer_fps": sum(r["infer_fps"] for r in rows),
            "dets_per_frame": sum(r["dets_per_frame"] for r in rows) / n,
            **g.summary(), **c.summary()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None)
    # weights live on the shared netdisk, NOT in /workspace (mutagen would delete them)
    ap.add_argument("--model", default=os.path.join(os.environ.get("DATA_ROOT", "data"),
                                                    "weights", "yolo26n-seg.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retina-masks", action="store_true")
    # COCO weights don't know mice: conf~0 + max_det=5 forces ~5 dets/frame, i.e. the
    # mask/track load of a 5-mouse cage. Replace with real thresholds once fine-tuned.
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--strides", default="1,2,3")
    ap.add_argument("--streams", default="1,2,4,6,8,12,16,20,24,32")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--rtf-ok", type=float, default=1.0)
    args = ap.parse_args()

    out = output_root()
    videos = find_videos(args.videos)
    infos = [probe_video(v) for v in videos]
    src_fps = infos[0].get("fps") or 30.0
    nframes = [int((i.get("duration") or 60) * (i.get("fps") or 30)) for i in infos]
    a = dict(model=args.model, imgsz=args.imgsz, half=args.half, conf=args.conf,
             max_det=args.max_det, retina_masks=args.retina_masks, seconds=args.seconds)
    ms = MetricSeries(out)
    head = {"source_fps": src_fps}

    for stride in [int(s) for s in args.strides.split(",")]:
        tgt = round(src_fps / stride)
        best = 0
        for n in [int(s) for s in args.streams.split(",")]:
            row = run_point(videos, n, stride, a, nframes)
            row["rtf_min"] = row["src_fps_min"] / src_fps
            row["ok"] = row["rtf_min"] >= args.rtf_ok
            row.update(imgsz=args.imgsz, half=args.half, retina_masks=args.retina_masks)
            print("POINT", row, flush=True)
            append_jsonl(out / "l0_points.jsonl", row)
            ms.log(n, f"l0_rtf_min_{tgt}fps", row["rtf_min"])
            ms.log(n, f"l0_agg_infer_fps_{tgt}fps", row["agg_infer_fps"])
            if row["ok"]:
                best = n
            elif row["rtf_min"] < 0.6 * args.rtf_ok:
                break  # far past the knee; don't waste time on larger N
        head[f"l0_max_streams_{tgt}fps"] = best
        print(f"RESULT stride={stride} ({tgt}fps): max realtime streams = {best}", flush=True)
    merge_metrics_json(out, head)
    print("SUMMARY", head, flush=True)


if __name__ == "__main__":
    main()

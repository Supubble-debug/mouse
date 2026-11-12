"""Does a decoder shortcut (default: skip_loop_filter=all, i.e. H.264 deblocking off) change
what the model sees?

Decodes the same frames of every video twice -- normal and with the option -- runs both
through the L2 GPU path (GpuEngine: GPU preprocess, TRT, NMS, proto-res masks) and compares
  pixels:      Y-plane PSNR
  detections:  for each reference detection, best-IoU match among the shortcut's detections
               -> mean IoU, share matched at IoU>=0.5, mask-area relative diff, centroid shift

Caveat: with COCO weights the low-confidence "mouse" detections are themselves unstable, so
detection differences are an upper bound; re-run with fine-tuned mouse weights.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import l1_pipeline as L
from common import append_jsonl, find_videos, merge_metrics_json, output_root


def decode(path: str, nframes: int, every: int, opts: dict | None) -> np.ndarray:
    import av
    out = []
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        if opts:
            vs.codec_context.options = dict(opts)
        for i, f in enumerate(c.decode(vs)):
            if i >= nframes:
                break
            if i % every == 0:
                buf = np.empty(L.SLOT_BYTES, np.uint8)
                L.copy_yuv(f, buf)
                out.append(buf)
    return np.stack(out)


def infer(eng, arr: np.ndarray) -> np.ndarray:
    import torch
    res = []
    for i in range(0, len(arr), eng.maxb):
        x = torch.from_numpy(arr[i:i + eng.maxb]).to(eng.dev)
        res.append(eng.infer(x).float().cpu().numpy())
    return np.concatenate(res)


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-6)


def psnr_y(a: np.ndarray, b: np.ndarray) -> float:
    ya = a[:L.H * L.W].astype(np.float32); yb = b[:L.H * L.W].astype(np.float32)
    mse = float(np.mean((ya - yb) ** 2))
    return 99.0 if mse == 0 else min(99.0, 10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None)
    ap.add_argument("--model", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "weights", "yolo26n-seg.pt"))
    ap.add_argument("--backend", choices=["trt", "torch"], default="trt")
    ap.add_argument("--opts", default="skip_loop_filter@all", help="'opt@val+opt@val'")
    ap.add_argument("--frames", type=int, default=450, help="first N frames of each video (15 GOPs)")
    ap.add_argument("--every", type=int, default=5, help="evaluate every k-th frame")
    a = ap.parse_args()
    opts = dict(kv.replace("@", "=").split("=", 1) for kv in a.opts.split("+") if kv)
    tag = "_".join(opts.values())

    eng = L.GpuEngine(argparse.Namespace(model=a.model, backend=a.backend, imgsz=640, max_batch=64, topk=5,
                                         mask_res="proto", pre_nms=100, nms_iou=0.7, batch_buckets=""))
    out = output_root()
    P, IOU, M50, AREA, CSHIFT, TOP1 = [], [], [], [], [], []
    for v in find_videos(a.videos):
        ref = decode(v, a.frames, a.every, None)
        tst = decode(v, a.frames, a.every, opts)
        n = min(len(ref), len(tst))
        ref, tst = ref[:n], tst[:n]
        ps = [psnr_y(ref[i], tst[i]) for i in range(n)]
        dr, dt = infer(eng, ref), infer(eng, tst)
        ious, m50, areas, cs, top1 = [], [], [], [], []
        for fr, ft in zip(dr, dt):
            r = fr[fr[:, 4] >= 0]; t = ft[ft[:, 4] >= 0]
            if not len(r):
                continue
            if not len(t):
                ious += [0.0] * len(r); m50 += [0.0] * len(r); top1.append(0.0)
                continue
            im = iou_mat(r[:, :4], t[:, :4])
            j = im.argmax(1); best = im.max(1)
            ious += best.tolist(); m50 += (best >= 0.5).astype(float).tolist()
            top1.append(float(best[0]))  # rows are sorted by confidence
            ok = best >= 0.5
            if ok.any():
                ra, ta = r[ok, 6], t[j[ok], 6]
                areas += (np.abs(ta - ra) / np.maximum(ra, 1)).tolist()
                cs += np.hypot(t[j[ok], 7] - r[ok, 7], t[j[ok], 8] - r[ok, 8]).tolist()
        row = {"video": os.path.basename(v), "opts": a.opts, "frames": n,
               "psnr_y_mean": float(np.mean(ps)), "psnr_y_min": float(np.min(ps)),
               "det_iou_mean": float(np.mean(ious)) if ious else None,
               "top1_iou_mean": float(np.mean(top1)) if top1 else None,
               "det_match50": float(np.mean(m50)) if m50 else None,
               "area_reldiff_median": float(np.median(areas)) if areas else None,
               "centroid_shift_px_median": float(np.median(cs)) if cs else None}
        print("VIDEO", row, flush=True)
        append_jsonl(out / f"check_{tag}.jsonl", row)
        P += ps; IOU += ious; M50 += m50; AREA += areas; CSHIFT += cs; TOP1 += top1
    summ = {f"{tag}_psnr_y_mean": float(np.mean(P)), f"{tag}_psnr_y_min": float(np.min(P)),
            f"{tag}_det_iou_mean": float(np.mean(IOU)), f"{tag}_top1_iou_mean": float(np.mean(TOP1)),
            f"{tag}_det_match50": float(np.mean(M50)),
            f"{tag}_area_reldiff_median": float(np.median(AREA)) if AREA else 0.0,
            f"{tag}_centroid_shift_px_median": float(np.median(CSHIFT)) if CSHIFT else 0.0,
            f"{tag}_frames": len(P)}
    merge_metrics_json(out, summ)
    print("SUMMARY", summ, flush=True)


if __name__ == "__main__":
    main()

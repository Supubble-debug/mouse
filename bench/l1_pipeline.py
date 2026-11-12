"""L1 multi-stream pipeline: everything except H.264 decode moves to the GPU.

L0 (bench/l0_ultralytics.py) is CPU-bound on this box: ~40 ms of CPU per frame,
of which decode is only ~2.5 ms, and AutoDL containers expose no NVDEC. L1 keeps
decode on the CPU but strips every other per-frame CPU cost:

  D decoder procs   PyAV, 1 thread each, raw yuv420p planes (no RGB conversion)
        |           -> shared-memory ring, S slots per stream, int8 state flags
        v              (no pickling of frames)
  1 GPU proc        batched H2D straight from the ring (cudaHostRegister'ed)
                    -> YUV->RGB (BT.601 limited) + area resize + letterbox on GPU
                    -> batched yolo26n-seg (TensorRT FP16 dynamic batch | torch FP16)
                    -> top-K dets, masks (coeffs @ protos, upsample, box crop,
                       threshold) + per-mask area / centroid, all on GPU;
                       only K x 9 floats per frame come back to the CPU
        |
        v
  T tracker procs   ultralytics BYTETracker, one per stream

Streams are unpaced, exactly like L0: decoders run as fast as backpressure
allows and a stream keeps up iff it advances >= source fps.
supported(N, k) := min over streams of RTF >= --rtf-ok.

Workload: COCO weights don't know mice, so --topk 5 --conf 0 forces exactly 5
detections + masks per frame (a 5-mouse cage) through post-processing and the
tracker. (In L0 only ~0.2 objects/frame survived, so L0 under-states the load.)

Bottleneck diagnostics per point: share of GPU-process wall time spent
  wait_gather  (waiting for decoded frames  -> decode-bound),
  wait_gpu     (blocked on GPU results      -> GPU-bound),
  cpu_launch / cpu_finish (its own Python work), plus tracker queue backlog.

A one-frame correctness check against the ultralytics reference pipeline runs
first (preprocessed-input MAD, box IoU, mask area); --check-only stops there.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from common import (CpuSampler, GpuSampler, MetricSeries, append_jsonl, effective_cpus,
                    find_videos, merge_metrics_json, output_root, probe_video)

TRACKER_YAML = Path(__file__).with_name("bytetrack_bench.yaml")
H, W = 1080, 1920
SLOT_BYTES = H * W * 3 // 2  # yuv420p
FREE, READY, BUSY = 0, 1, 2
NCOL = 9  # x1 y1 x2 y2 conf cls area cx cy  (original-frame coordinates)


def letterbox_geom(imgsz: int):
    """Ultralytics 'rect' letterbox for a 1920x1080 frame: 640 -> 640x360 padded to 640x384."""
    r = min(imgsz / W, imgsz / H)
    nh, nw = round(H * r), round(W * r)
    in_h, in_w = math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32
    return r, nh, nw, in_h, in_w, (in_h - nh) // 2, (in_w - nw) // 2


# ---------------------------------------------------------------- decoders
def _frames(path: str, threads: int):
    import av
    while True:  # loop the file forever
        with av.open(path) as c:
            vs = c.streams.video[0]
            try:
                vs.codec_context.thread_count = threads
                vs.thread_type = "AUTO" if threads > 1 else "SLICE"
            except Exception:  # noqa: BLE001
                pass
            yield from c.decode(vs)


def copy_yuv(f, dst: np.ndarray) -> None:
    """Copy a decoded frame's yuv420p planes (dropping line padding) into a ring slot."""
    if f.format.name != "yuv420p" or f.width != W or f.height != H:
        f = f.reformat(width=W, height=H, format="yuv420p")
    off = 0
    for i, (h, w) in enumerate(((H, W), (H // 2, W // 2), (H // 2, W // 2))):
        p = f.planes[i]
        src = np.frombuffer(p, np.uint8).reshape(-1, p.line_size)[:h, :w]
        dst[off:off + h * w].reshape(h, w)[...] = src
        off += h * w


def _decoder(assign, nslots, nring, stride, threads, ring_path, state_raw, ctr_raw, stop, barrier):
    os.environ["OMP_NUM_THREADS"] = "1"
    ring = np.memmap(ring_path, dtype=np.uint8, mode="r+", shape=(nring, SLOT_BYTES))
    state = np.frombuffer(state_raw, dtype=np.int8)
    ctr = np.frombuffer(ctr_raw, dtype=np.int64)
    gens = [(s, _frames(p, threads)) for s, p in assign]
    nsrc = {s: 0 for s, _ in assign}
    nput = {s: 0 for s, _ in assign}
    barrier.wait()
    while not stop.value:
        for s, g in gens:
            f = next(g)
            nsrc[s] += 1
            if (nsrc[s] - 1) % stride:
                continue  # decoded (unavoidable for H.264) but not analysed
            slot = s * nslots + nput[s] % nslots
            while state[slot] != FREE:
                if stop.value:
                    return
                time.sleep(0.0005)
            copy_yuv(f, ring[slot])
            ctr[slot] = nsrc[s]
            state[slot] = READY
            nput[s] += 1


# ---------------------------------------------------------------- trackers
def _tracker(q, prog_raw, objs_raw, nfr_raw, stride, conf, barrier):
    os.environ["OMP_NUM_THREADS"] = "1"
    from types import SimpleNamespace

    import yaml
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    cfg = yaml.safe_load(TRACKER_YAML.read_text())
    # BYTETracker(args) no longer takes frame_rate; track_buffer is in frames, so scale it
    # to the analysed frame rate (buffer is specified for 30 fps).
    cfg["track_buffer"] = max(1, round(cfg["track_buffer"] / stride))
    cfg = SimpleNamespace(**cfg)
    prog = np.frombuffer(prog_raw, np.int64)
    objs = np.frombuffer(objs_raw, np.int64)
    nfr = np.frombuffer(nfr_raw, np.int64)
    trackers = {}
    barrier.wait()
    while True:
        msg = q.get()
        if msg is None:
            break
        for s, c, det in msg:
            if s not in trackers:
                trackers[s] = BYTETracker(cfg)
            d = det[det[:, 4] >= conf]
            out = trackers[s].update(Boxes(d[:, :6], (H, W)))
            objs[s] += len(out)
            nfr[s] += 1
            prog[s] = c


# ---------------------------------------------------------------- GPU side
def parse_outputs(out, in_h: int, nm: int = 32):
    """Find (dets[B,Q,6+nm] xyxy/conf/cls/coeffs, protos[B,nm,h,w]) in any backend's output."""
    import torch
    ts: list = []

    def walk(o):
        if isinstance(o, torch.Tensor):
            ts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(out)
    protos = next((t for t in ts if t.ndim == 4 and t.shape[1] == nm and t.shape[2] * 4 == in_h), None)
    if protos is None:
        raise RuntimeError(f"no protos in outputs: {[tuple(t.shape) for t in ts]}")
    e2e = next((t for t in ts if t.ndim == 3 and t.shape[-1] == 6 + nm and t.shape[1] <= 1000), None)
    if e2e is not None:  # yolo26 end2end head: already NMS-free top-300
        return e2e, protos, False
    raw = next((t for t in ts if t.ndim == 3 and t.shape[1] > 4 + nm and t.shape[2] > 1000), None)
    if raw is None:
        raise RuntimeError(f"no detection tensor in outputs: {[tuple(t.shape) for t in ts]}")
    raw = raw.transpose(1, 2)  # (B, A, 4+nc+nm); NMS happens in GpuEngine.select
    nc = raw.shape[-1] - 4 - nm
    xy, wh = raw[..., :2], raw[..., 2:4]
    sc, cls = raw[..., 4:4 + nc].max(-1)
    return torch.cat([xy - wh / 2, xy + wh / 2, sc[..., None], cls[..., None].to(raw.dtype),
                      raw[..., 4 + nc:]], -1), protos, True


_TRT2TORCH = {}


class TrtRunner:
    """Minimal TensorRT 10/11 runner: dtype-correct inputs, torch-allocated outputs,
    execute_async_v3 on torch's current stream (no host sync, no reused output buffers).

    ultralytics' AutoBackend(fp16=True) fed half tensors into this engine's FP32 input
    binding -> NaN + illegal address (2026-11-12); this avoids that.
    """

    def __init__(self, path: str):
        import json

        import tensorrt as trt
        import torch
        data = Path(path).read_bytes()
        try:  # ultralytics prefixes engines with <int32 len><json metadata>
            n = int.from_bytes(data[:4], "little", signed=True)
            json.loads(data[4:4 + n])
            data = data[4 + n:]
        except Exception:  # noqa: BLE001
            pass
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if n not in self.inputs]
        m = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
             trt.int8: torch.int8, trt.bool: torch.bool}
        if hasattr(trt, "int64"):
            m[trt.int64] = torch.int64
        if hasattr(trt, "bfloat16"):
            m[trt.bfloat16] = torch.bfloat16
        self.dtype = {n: m[self.engine.get_tensor_dtype(n)] for n in names}
        print("TRT io:", {n: (str(self.dtype[n]), tuple(self.engine.get_tensor_shape(n))) for n in names},
              flush=True)

    def __call__(self, x):
        torch = self.torch
        name = self.inputs[0]
        x = x.to(self.dtype[name]).contiguous()
        self.ctx.set_input_shape(name, tuple(x.shape))
        self.ctx.set_tensor_address(name, x.data_ptr())
        outs = []
        for n in self.outputs:
            t = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=self.dtype[n], device=x.device)
            self.ctx.set_tensor_address(n, t.data_ptr())
            outs.append(t)
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return outs


class GpuEngine:
    def __init__(self, a):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        torch.backends.cudnn.benchmark = False
        self.dev = torch.device("cuda:0")
        # all GPU work of this process on one non-default stream: TRT enqueueV3 on the legacy
        # default stream adds cudaStreamSynchronize calls (kills CPU/GPU overlap)
        self.stream = torch.cuda.Stream(device=self.dev)
        torch.cuda.set_stream(self.stream)
        self.r, self.nh, self.nw, self.in_h, self.in_w, self.top, self.left = letterbox_geom(a.imgsz)
        self.K, self.maxb = a.topk, a.max_batch
        self.mask_res, self.pre_nms, self.nms_iou = a.mask_res, a.pre_nms, a.nms_iou
        self.off = torch.tensor([self.left, self.top, self.left, self.top], device=self.dev,
                                dtype=torch.float32)
        self._grids = {}
        self.backend_name, self.net = self._load(a)
        self.gpu_in = torch.empty((self.maxb, SLOT_BYTES), dtype=torch.uint8, device=self.dev)
        self.out = [torch.empty((self.maxb, self.K, NCOL), dtype=torch.float32).pin_memory()
                    for _ in range(2)]
        self.flip = 0
        self.xs = torch.arange(self.in_w, device=self.dev, dtype=torch.float32)
        self.ys = torch.arange(self.in_h, device=self.dev, dtype=torch.float32)
        self.host = None
        self.host_pinned = False
        for b in (1, self.maxb, 1):  # warm-up (TRT context / cuBLAS / allocator)
            self.infer(torch.zeros((b, SLOT_BYTES), dtype=torch.uint8, device=self.dev))
        torch.cuda.synchronize()

    # -- model -------------------------------------------------------------
    def _load(self, a):
        torch = self.torch
        if a.backend == "trt":
            # no silent fallback: a TRT failure must not masquerade as a TRT result
            return "trt", TrtRunner(ensure_engine(a, self.in_h, self.in_w))
        from ultralytics import YOLO
        net = YOLO(a.model, task="segment").model
        try:
            net = net.fuse(verbose=False)
        except Exception:  # noqa: BLE001
            pass
        return "torch", net.to(self.dev).half().eval()

    # -- stages ------------------------------------------------------------
    def preprocess(self, buf):
        """(B, SLOT_BYTES) uint8 yuv420p on GPU -> (B,3,in_h,in_w) fp16 RGB in [0,1], letterboxed."""
        torch, F = self.torch, self.F
        B = buf.shape[0]
        y = buf[:, :H * W].view(B, 1, H, W).half()
        uv = buf[:, H * W:].view(B, 2, H // 2, W // 2).half()
        y = F.interpolate(y, size=(self.nh, self.nw), mode="area")
        uv = F.interpolate(uv, size=(self.nh, self.nw), mode="area")
        y = (y - 16.0) * 1.164
        u, v = uv[:, :1] - 128.0, uv[:, 1:] - 128.0
        rgb = torch.cat([y + 1.596 * v, y - 0.392 * u - 0.813 * v, y + 2.017 * u], 1)
        x = torch.full((B, 3, self.in_h, self.in_w), 114 / 255, dtype=torch.half, device=self.dev)
        x[:, :, self.top:self.top + self.nh, self.left:self.left + self.nw] = rgb.clamp_(0, 255).div_(255)
        return x

    def _grid(self, h, w):
        """Pixel-centre coordinates (in letterboxed-input pixels) of an h x w mask grid."""
        if (h, w) not in self._grids:
            t = self.torch
            sx, sy = self.in_w / w, self.in_h / h
            self._grids[(h, w)] = ((t.arange(w, device=self.dev, dtype=t.float32) + 0.5) * sx,
                                   (t.arange(h, device=self.dev, dtype=t.float32) + 0.5) * sy, sx * sy)
        return self._grids[(h, w)]

    def select(self, dets, raw: bool):
        """Top-K detections per image. Raw heads (B,A,6+nm) get class-aware NMS first
        (pre-NMS top-P by score, like a real deployment); end2end heads are NMS-free.
        Slots with no surviving detection get conf = -1."""
        torch = self.torch
        B, Q, C = dets.shape
        if not raw:
            k = min(self.K, Q)
            _, idx = dets[..., 4].topk(k, dim=1)
            return dets.gather(1, idx.unsqueeze(-1).expand(-1, -1, C)).float()
        import torchvision
        P = min(self.pre_nms, Q)
        _, idx = dets[..., 4].topk(P, dim=1)
        cand = dets.gather(1, idx.unsqueeze(-1).expand(-1, -1, C)).float()  # (B,P,C) sorted by score
        boxes, scores = cand[..., :4].reshape(-1, 4), cand[..., 4].reshape(-1)
        grp = torch.arange(B, device=self.dev).repeat_interleave(P) * 1000 + cand[..., 5].reshape(-1).long()
        keep = torchvision.ops.batched_nms(boxes, scores, grp, self.nms_iou)
        masked = torch.full((B * P,), -1e4, device=self.dev)
        masked[keep] = scores[keep]
        k = min(self.K, P)
        top_s, top = masked.view(B, P).topk(k, dim=1)
        sel = cand.gather(1, top.unsqueeze(-1).expand(-1, -1, C))
        sel[..., 4] = torch.where(top_s < -1e3, torch.full_like(top_s, -1.0), sel[..., 4])
        return sel

    def postprocess(self, out):
        """-> (B, K, 9) float32 on GPU, original-frame coordinates."""
        torch, F = self.torch, self.F
        dets, protos, raw = parse_outputs(out, self.in_h)
        sel = self.select(dets, raw)
        B, k = sel.shape[:2]
        boxes, score, cls, coef = sel[..., :4], sel[..., 4], sel[..., 5], sel[..., 6:]
        ph, pw = protos.shape[-2:]
        m = torch.bmm(coef.half(), protos.half().flatten(2)).view(B, k, ph, pw)
        if self.mask_res == "input":  # what ultralytics does (retina_masks=False)
            m = F.interpolate(m, size=(self.in_h, self.in_w), mode="bilinear", align_corners=False)
        gx, gy, px_area = self._grid(*m.shape[-2:])
        x1, y1, x2, y2 = (t[..., None, None] for t in boxes.unbind(-1))
        xs, ys = gx.view(1, 1, 1, -1), gy.view(1, 1, -1, 1)
        mb = (m > 0) & (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)  # sigmoid>0.5, cropped
        cnt = mb.sum((2, 3)).float()
        cx = (mb.sum(2).float() @ gx) / cnt.clamp(min=1)
        cy = (mb.sum(3).float() @ gy) / cnt.clamp(min=1)
        r = self.r
        bo = (boxes - self.off) / r
        return torch.cat([bo, score[..., None], cls[..., None], (cnt * px_area / r ** 2)[..., None],
                          ((cx - self.left) / r)[..., None], ((cy - self.top) / r)[..., None]], -1)

    def infer(self, buf):
        with self.torch.inference_mode():
            return self.postprocess(self.net(self.preprocess(buf)))

    # -- ring + pipelined launch/finish -------------------------------------
    def attach_ring(self, ring: np.ndarray):
        torch = self.torch
        self.host = torch.from_numpy(np.asarray(ring))
        self.host_pinned = False
        try:
            rc = torch.cuda.cudart().cudaHostRegister(self.host.data_ptr(), self.host.numel(), 0)
            self.host_pinned = int(getattr(rc, "value", rc)) == 0
        except Exception as e:  # noqa: BLE001
            print(f"WARN: cudaHostRegister failed ({e!r}); H2D will be synchronous", flush=True)

    def detach_ring(self):
        if self.host is not None and self.host_pinned:
            try:
                self.torch.cuda.cudart().cudaHostUnregister(self.host.data_ptr())
            except Exception:  # noqa: BLE001
                pass
        self.host = None

    def launch(self, idx: np.ndarray):
        torch = self.torch
        B = len(idx)
        gi = self.gpu_in[:B]
        for i, j in enumerate(idx.tolist()):
            gi[i].copy_(self.host[j], non_blocking=True)
        res = self.infer(gi)
        out = self.out[self.flip][:B]
        self.flip ^= 1
        out.copy_(res, non_blocking=True)
        evt = torch.cuda.Event()
        evt.record()
        return idx, out, evt


def ensure_engine(a, in_h: int, in_w: int) -> str:
    p = Path(a.model)
    eng = p.with_name(f"{p.stem}_{in_h}x{in_w}_b{a.max_batch}_fp16.engine")
    if eng.exists():
        return str(eng)
    from ultralytics import YOLO
    print(f"exporting TensorRT engine -> {eng}", flush=True)
    out = YOLO(a.model, task="segment").export(format="engine", imgsz=(in_h, in_w), half=True,
                                                dynamic=True, batch=a.max_batch, device=0,
                                                verbose=False)
    Path(out).rename(eng)
    return str(eng)


# ---------------------------------------------------------------- one point
def gather(state, ctr, nslots, maxb, minb, timeout_s, have_pending):
    t0 = time.perf_counter()
    while True:
        idx = np.flatnonzero(state == READY)
        waited = time.perf_counter() - t0
        if len(idx) >= minb or (len(idx) and waited >= timeout_s):
            break
        if have_pending and (not len(idx) or waited >= timeout_s):
            return idx[:0]  # go finish the in-flight batch (frees slots)
        if waited > 1.0:
            return idx[:0]
        time.sleep(0.0002)
    if len(idx) > maxb:  # oldest frames first = fairness across streams
        idx = idx[np.argsort(ctr[idx], kind="stable")[:maxb]]
    idx = idx[np.lexsort((ctr[idx], idx // nslots))]  # per-stream frame order for the tracker
    state[idx] = BUSY
    return idx


def run_point(eng: GpuEngine, videos, n, stride, a):
    ncpu = effective_cpus()
    # one tracker proc handled >=1600 updates/s with zero backlog in L1; a 2nd one costs a decoder
    T = a.trackers or (1 if min(n * 30 / stride, 3600) <= 2000 else 2)
    D = a.decoders or max(1, min(n, ncpu - T - 1))
    S = a.slots
    nring = n * S
    need = nring * SLOT_BYTES
    if need > 0.9 * shutil.disk_usage(a.shm_dir).free:
        raise SystemExit(f"ring needs {need / 2**30:.1f} GiB, not enough free in {a.shm_dir}")
    ring_path = os.path.join(a.shm_dir, f"mouse_l1_{os.getpid()}.ring")
    ring = np.memmap(ring_path, dtype=np.uint8, mode="w+", shape=(nring, SLOT_BYTES))

    ctx = mp.get_context("spawn")
    state_raw, ctr_raw = ctx.RawArray("b", nring), ctx.RawArray("q", nring)
    prog_raw, objs_raw, nfr_raw = (ctx.RawArray("q", n) for _ in range(3))
    stop = ctx.RawValue("b", 0)
    barrier = ctx.Barrier(D + T + 1)
    qs = [ctx.Queue() for _ in range(T)]
    assign = [[] for _ in range(D)]
    for s in range(n):
        assign[s % D].append((s, videos[s % len(videos)]))
    procs = [ctx.Process(target=_decoder, args=(assign[d], S, nring, stride, a.dec_threads, ring_path,
                                                 state_raw, ctr_raw, stop, barrier), daemon=True)
             for d in range(D)]
    procs += [ctx.Process(target=_tracker, args=(qs[t], prog_raw, objs_raw, nfr_raw, stride, a.conf,
                                                 barrier), daemon=True) for t in range(T)]
    for p in procs:
        p.start()

    state = np.frombuffer(state_raw, np.int8)
    ctr = np.frombuffer(ctr_raw, np.int64)
    prog = np.frombuffer(prog_raw, np.int64)
    objs = np.frombuffer(objs_raw, np.int64)
    nfr = np.frombuffer(nfr_raw, np.int64)
    eng.attach_ring(ring)
    maxb = min(a.max_batch, n * S)
    minb = a.min_batch or max(1, min(maxb, n // 2))
    tmo = a.batch_timeout_ms / 1000

    st = dict(batches=0, frames=0, wait_gather=0.0, wait_gpu=0.0, cpu_launch=0.0, cpu_finish=0.0)

    def finish(p):
        idx, out, evt = p
        t0 = time.perf_counter()
        evt.synchronize()
        t1 = time.perf_counter()
        res = out.numpy().copy()
        streams = (idx // S).tolist()
        ctrs = ctr[idx].tolist()  # read before releasing the slots
        state[idx] = FREE
        msgs = [[] for _ in range(T)]
        for i, (s, c) in enumerate(zip(streams, ctrs)):
            msgs[s % T].append((s, c, res[i]))
        for t, m in enumerate(msgs):
            if m:
                qs[t].put(m)
        st["wait_gpu"] += t1 - t0
        st["cpu_finish"] += time.perf_counter() - t1
        st["batches"] += 1
        st["frames"] += len(idx)

    barrier.wait(timeout=600)
    t_meas0 = time.perf_counter() + a.warmup
    t_end = t_meas0 + a.seconds
    snap0 = None
    g = c = None
    pending = None
    next_alive = 0.0
    while True:
        now = time.perf_counter()
        if now >= next_alive:  # a dead worker would silently freeze progress -> fail loudly
            dead = [p.name for p in procs if not p.is_alive()]
            if dead:
                stop.value = 1
                for p in procs:
                    p.terminate()
                eng.detach_ring()
                os.unlink(ring_path)
                raise RuntimeError(f"worker process(es) died: {dead}")
            next_alive = now + 1.0
        if snap0 is None and now >= t_meas0:
            snap0 = (now, prog.copy(), objs.copy(), nfr.copy(), dict(st))
            g, c = GpuSampler().__enter__(), CpuSampler().__enter__()
        if now >= t_end:
            break
        idx = gather(state, ctr, S, maxb, minb, tmo, pending is not None)
        t1 = time.perf_counter()
        st["wait_gather"] += t1 - now
        launched = eng.launch(idx) if len(idx) else None
        st["cpu_launch"] += time.perf_counter() - t1
        if pending is not None:
            finish(pending)
        pending = launched
    t1 = time.perf_counter()
    snap1 = (t1, prog.copy(), objs.copy(), nfr.copy(), dict(st))
    c.__exit__(None, None, None)
    g.__exit__(None, None, None)
    backlog = [q.qsize() for q in qs]
    if pending is not None:
        finish(pending)

    stop.value = 1
    for q in qs:
        q.put(None)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    eng.detach_ring()
    del ring
    os.unlink(ring_path)

    dt = snap1[0] - snap0[0]
    src = (snap1[1] - snap0[1]) / dt
    frames = int((snap1[3] - snap0[3]).sum())
    s0, s1 = snap0[4], snap1[4]
    nb = max(s1["batches"] - s0["batches"], 1)
    share = {f"share_{k}": (s1[k] - s0[k]) / dt for k in ("wait_gather", "wait_gpu", "cpu_launch", "cpu_finish")}
    return {"pipeline": a.tag, "backend": eng.backend_name, "mask_res": eng.mask_res, "streams": n, "stride": stride,
            "decoders": D, "trackers": T, "slots": S, "host_pinned": eng.host_pinned,
            "src_fps_min": float(src.min()), "src_fps_mean": float(src.mean()),
            "agg_src_fps": float(src.sum()), "agg_infer_fps": frames / dt,
            "objs_per_frame": float((snap1[2] - snap0[2]).sum() / max(frames, 1)),
            "batch_mean": (s1["frames"] - s0["frames"]) / nb, "tracker_backlog_max": max(backlog),
            **share, **g.summary(), **c.summary()}


# ---------------------------------------------------------------- correctness check
def check(eng: GpuEngine, video: str, a) -> dict:
    """One frame through our GPU path vs the ultralytics reference path."""
    import av
    import cv2
    import torch
    from ultralytics import YOLO

    with av.open(video) as cont:
        f = next(cont.decode(video=0))
    yuv = np.empty(SLOT_BYTES, np.uint8)
    copy_yuv(f, yuv)
    bgr = f.to_ndarray(format="bgr24")

    ours_in = eng.preprocess(torch.from_numpy(yuv).to(eng.dev)[None])
    rs = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (eng.nw, eng.nh), interpolation=cv2.INTER_AREA)
    ref = np.full((eng.in_h, eng.in_w, 3), 114, np.uint8)
    ref[eng.top:eng.top + eng.nh, eng.left:eng.left + eng.nw] = rs
    ref_in = torch.from_numpy(ref).to(eng.dev).permute(2, 0, 1)[None].float() / 255
    mad = float((ours_in.float() - ref_in).abs().mean())

    ours = eng.infer(torch.from_numpy(yuv).to(eng.dev)[None])[0].cpu().numpy()
    rr = YOLO(a.model, task="segment").predict(bgr, imgsz=a.imgsz, conf=max(a.conf, 1e-4), max_det=a.topk,
                                                half=True, device=0, verbose=False)[0]
    rb = rr.boxes.xyxy.cpu().numpy()
    rc = rr.boxes.conf.cpu().numpy()
    ra = (rr.masks.data.sum((1, 2)).cpu().numpy() / eng.r ** 2) if rr.masks is not None else np.zeros(len(rb))

    def iou(b, bs):
        if not len(bs):
            return 0.0
        x1 = np.maximum(b[0], bs[:, 0]); y1 = np.maximum(b[1], bs[:, 1])
        x2 = np.minimum(b[2], bs[:, 2]); y2 = np.minimum(b[3], bs[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        ua = (b[2] - b[0]) * (b[3] - b[1]) + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1]) - inter
        return inter / np.maximum(ua, 1e-6)

    best = []
    for i, row in enumerate(ours):
        ious = iou(row[:4], rb)
        j = int(np.argmax(ious)) if len(rb) else -1
        best.append(float(ious[j]) if j >= 0 else 0.0)
        ref_s = (f"ref conf={rc[j]:.3f} area={ra[j]:.0f} box={np.round(rb[j]).tolist()}" if j >= 0 else "ref -")
        print(f"CHECK #{i} ours conf={row[4]:.3f} area={row[6]:.0f} box={np.round(row[:4]).tolist()} "
              f"| best IoU={best[-1]:.3f} {ref_s}", flush=True)
    res = {"check_input_mad": mad, "check_top1_iou": best[0] if best else 0.0,
           "check_mean_iou": float(np.mean(best)) if best else 0.0, "check_backend": eng.backend_name}
    print("CHECK", res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None)
    ap.add_argument("--model", default=os.path.join(os.environ.get("DATA_ROOT", "data"),
                                                    "weights", "yolo26n-seg.pt"))
    ap.add_argument("--backend", choices=["trt", "torch"], default="trt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--min-batch", type=int, default=0, help="0 = auto (half the streams)")
    ap.add_argument("--batch-timeout-ms", type=float, default=3.0)
    ap.add_argument("--topk", type=int, default=5, help="dets (+masks) per frame = mice per cage")
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--slots", type=int, default=3, help="ring slots per stream")
    ap.add_argument("--mask-res", choices=["proto", "input"], default="proto",
                    help="proto: area/centroid on the 96x160 proto grid; input: upsample to 384x640 first")
    ap.add_argument("--pre-nms", type=int, default=100, help="candidates per image entering NMS")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--tag", default="l1", help="prefix for points file / metric names")
    ap.add_argument("--decoders", type=int, default=0, help="0 = auto (cpus - trackers - 1)")
    ap.add_argument("--trackers", type=int, default=0, help="0 = auto")
    ap.add_argument("--dec-threads", type=int, default=1)
    ap.add_argument("--shm-dir", default="/dev/shm")
    ap.add_argument("--strides", default="1,2,3")
    ap.add_argument("--streams", default="8,16,24,32,48,64,80,96")
    ap.add_argument("--warmup", type=float, default=10)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--rtf-ok", type=float, default=1.0)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    out = output_root()
    videos = find_videos(a.videos)
    src_fps = probe_video(videos[0]).get("fps") or 30.0
    eng = GpuEngine(a)
    print(f"backend={eng.backend_name} input={eng.in_h}x{eng.in_w} cpus={effective_cpus()}", flush=True)
    head = {"source_fps": src_fps, **check(eng, videos[0], a)}
    merge_metrics_json(out, head)
    if not np.isfinite(head["check_top1_iou"]) or head["check_top1_iou"] < 0.5:
        raise SystemExit(f"{eng.backend_name} detections disagree with reference "
                         f"(top1 IoU {head['check_top1_iou']}); not benchmarking")
    if head["check_input_mad"] > 0.05:
        raise SystemExit(f"preprocessing mismatch (MAD {head['check_input_mad']:.3f}); not benchmarking")
    if a.check_only:
        return

    ms = MetricSeries(out)
    for stride in [int(s) for s in a.strides.split(",")]:
        tgt = round(src_fps / stride)
        best = 0
        for n in [int(s) for s in a.streams.split(",")]:
            row = run_point(eng, videos, n, stride, a)
            row["rtf_min"] = row["src_fps_min"] / src_fps
            row["ok"] = row["rtf_min"] >= a.rtf_ok
            row.update(imgsz=a.imgsz, topk=a.topk, max_batch=a.max_batch)
            print("POINT", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            append_jsonl(out / f"{a.tag}_points.jsonl", row)
            ms.log(n, f"{a.tag}_rtf_min_{tgt}fps", row["rtf_min"])
            ms.log(n, f"{a.tag}_agg_infer_fps_{tgt}fps", row["agg_infer_fps"])
            if row["ok"]:
                best = n
            elif row["rtf_min"] < 0.6 * a.rtf_ok:
                break
        head[f"{a.tag}_max_streams_{tgt}fps"] = best
        print(f"RESULT stride={stride} ({tgt}fps): max realtime streams = {best}", flush=True)
        merge_metrics_json(out, head)
    print("SUMMARY", head, flush=True)


if __name__ == "__main__":
    main()

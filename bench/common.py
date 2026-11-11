"""Shared helpers for the multi-stream throughput benchmark."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import threading
import time
from pathlib import Path

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".h264", ".h265")


def output_root() -> Path:
    p = Path(os.environ.get("OUTPUT_ROOT", "output"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_videos(pattern: str | None = None) -> list[str]:
    if pattern:
        vids = sorted(glob.glob(pattern))
    else:
        root = Path(os.environ.get("DATA_ROOT", "data")) / "videos"
        vids = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        raise SystemExit(f"no videos found (pattern={pattern!r}, DATA_ROOT={os.environ.get('DATA_ROOT')})")
    return vids


def probe_video(path: str) -> dict:
    """ffprobe codec / resolution / fps / GOP-ish info; empty dict if ffprobe missing."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,profile,width,height,r_frame_rate,avg_frame_rate,bit_rate,nb_frames,pix_fmt,has_b_frames",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=60, check=True).stdout
        d = json.loads(out)
        s = d["streams"][0]
        num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
        s["fps"] = num / den if den else None
        s["duration"] = float(d.get("format", {}).get("duration", 0) or 0)
        return s
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


class GpuSampler:
    """Background NVML sampler: GPU util, NVDEC util, memory used."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            return self

        def loop():
            while not self._stop.is_set():
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    dec = pynvml.nvmlDeviceGetDecoderUtilization(h)[0]
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30
                    self.samples.append((u, dec, mem))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        n = len(self.samples)
        return {
            "gpu_util_mean": sum(s[0] for s in self.samples) / n,
            "nvdec_util_mean": sum(s[1] for s in self.samples) / n,
            "gpu_mem_peak_gb": max(s[2] for s in self.samples),
        }


class CpuSampler:
    def __enter__(self):
        try:
            import psutil
            psutil.cpu_percent(None)
            self._ps = psutil
        except Exception:  # noqa: BLE001
            self._ps = None
        return self

    def __exit__(self, *exc):
        self.cpu = self._ps.cpu_percent(None) if self._ps else None

    def summary(self) -> dict:
        return {"cpu_util_mean": self.cpu, "ncpu": os.cpu_count()}


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class MetricSeries:
    """projtool metrics.jsonl writer (schema 1)."""

    def __init__(self, root: Path):
        self.path = root / "metrics.jsonl"
        if not self.path.exists():
            run_id = os.environ.get("PROJTOOL_RUN_ID", "")
            append_jsonl(self.path, {"__meta__": {"schema": 1, "run_id": run_id}})

    def log(self, step: int, name: str, value: float, split: str | None = None) -> None:
        row = {"step": int(step), "name": name, "value": float(value)}
        if split:
            row["split"] = split
        append_jsonl(self.path, row)


def merge_metrics_json(root: Path, new: dict) -> None:
    p = root / "metrics.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(new)
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False))

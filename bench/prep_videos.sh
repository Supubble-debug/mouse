#!/usr/bin/env bash
# Download the Zenodo 14580256 mouse videos and transcode them to a 1080p30 camera-like
# H.264 stream, which is what the benchmark targets.
#
# The source clips are small and high-fps (416x444..664x630 @ 40-94 Hz). Decode cost
# scales with pixel count, so benchmarking them as-is would overstate capacity ~5x.
# Transcode: stretch to 1920x1080 (keeps full-frame content density; aspect distortion
# is irrelevant for throughput), resample to 30 fps, H.264 High, GOP 30, no B-frames,
# ~4 Mbps CBR-ish -- typical of an IP / USB camera recording.
#
#   raw      -> $DATA_ROOT/videos_raw/   (kept for later accuracy work)
#   1080p30  -> $DATA_ROOT/videos/       (what find_videos() picks up)
set -euo pipefail
export PATH="/root/miniconda3/bin:$PATH"
# NOTE: do NOT source /etc/network_turbo here -- it routes zenodo through AutoDL's
# github/hf proxy at ~30 KB/s vs ~1.3 MB/s direct (measured 2026-11-11).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

DATA_ROOT=${DATA_ROOT:-/root/autodl-fs/datasets/mouse}
RAW=$DATA_ROOT/videos_raw
OUT=$DATA_ROOT/videos
mkdir -p "$RAW" "$OUT"

BASE=https://zenodo.org/records/14580256/files
NAMES=(
  mice_10_600x686_40Hz_6min.mp4
  mice_3_652x636_67Hz_6min57s.mp4
  mice_5_416x444_94Hz_11min30s.mp4
  mice_5_652x636_66Hz_8min4s.mp4
  mice_7_664x630_67Hz_7min21s.mp4
)

echo "== download ${#NAMES[@]} files in parallel"
for n in "${NAMES[@]}"; do
  wget -q -c --tries=10 --timeout=60 -O "$RAW/$n" "$BASE/$n?download=1" &
done
wait
for n in "${NAMES[@]}"; do
  echo "== $n"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames -of csv=p=0 "$RAW/$n"
done

for n in "${NAMES[@]}"; do
  o="$OUT/${n%.mp4}_1080p30.mp4"
  [ -s "$o" ] && { echo "== skip $o"; continue; }
  echo "== transcode $n -> $o"
  ffmpeg -nostdin -y -loglevel error -i "$RAW/$n" \
    -vf "scale=1920:1080:flags=bicubic,fps=30" \
    -c:v libx264 -preset fast -profile:v high -pix_fmt yuv420p \
    -g 30 -bf 0 -b:v 4M -maxrate 4M -bufsize 8M -an "$o.tmp.mp4"
  mv "$o.tmp.mp4" "$o"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,bit_rate -of csv=p=0 "$o"
done
ls -lh "$RAW" "$OUT"

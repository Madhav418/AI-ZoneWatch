# Sample Data

This directory is for sample video files for testing.

## Recommended Sample Videos

For a manageable scope, we recommend:

### 1. MOT17 Dataset (Multi-Object Tracking)

- **Source**: https://motchallenge.net/data/MOT17/
- **What it contains**: Clean ground truth tracking annotations for pedestrians
- **Recommended clips**:
  - MOT17-02 (KITTI): 600 frames
  - MOT17-04 (KITTI): 1050 frames
  - MOT17-14 (DPM): 600 frames
- **Size per clip**: 50-100 MB
- **Use case**: Ideal for tracking evaluation

### 2. UCF-Crime Dataset

- **Source**: https://www.dropbox.com/sh/75v5ehq4cdmuq5o/AADxirVJ7see_l0_bgqKpZjya/Videos?dl=0
- **What it contains**: Real-world CCTV footage with anomalies
- **Recommended clips**: Select 1-2 normal activity videos
- **Use case**: Test robustness on real-world messy data

### 3. VIRAT Dataset

- **Source**: https://www.viratdata.org/
- **What it contains**: Ground/aerial surveillance of diverse scenes
- **Recommended clips**: pedestrian detection scenes
- **Use case**: Multi-scale detection testing

## Quick Start Alternative

To test without real videos, you can:

1. Create synthetic video:

```python
import cv2
import numpy as np

# Create synthetic video
width, height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sample_data/synthetic_test.mp4', fourcc, 30.0, (width, height))

for i in range(150):  # 5 seconds at 30 FPS
    frame = np.ones((height, width, 3), dtype=np.uint8) * 50
    # Add moving rectangle
    x = 50 + i * 2
    y = 100
    cv2.rectangle(frame, (x, y), (x+100, y+100), (0, 255, 0), -1)
    out.write(frame)

out.release()
```

2. Or use an existing video from your camera or stock clips

## Downloading MOT17

```bash
# Using wget or curl
cd sample_data
wget https://motchallenge.net/data/MOT17/train/MOT17-02-KITTI.zip
unzip MOT17-02-KITTI.zip
```

## Testing Pipeline

```bash
python run.py --video sample_data/MOT17-02/img1/video.mp4 --zones config/zones.json --output results/
```

Note: You may need to convert image sequence to video:

```bash
ffmpeg -framerate 30 -pattern_type glob -i 'sample_data/MOT17-02/img1/*.jpg' -c:v libx264 -pix_fmt yuv420p output.mp4
```

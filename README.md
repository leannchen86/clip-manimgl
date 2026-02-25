# CLIP Encoding Animation

A ManimGL scene visualizing how CLIP encodes text and image pairs into embeddings.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies (Python ≤ 3.12 can use PyPI; Python 3.13 must install from source):
   ```bash
   # Python ≤ 3.12
   pip install manimgl

   # Python 3.13+ — install from the local manimgl source repo
   pip install /path/to/manimgl
   ```

## Running the Scene

**Interactive mode** (live preview, auto-reloads on file save):
```bash
manimgl clip_encoding.py CLIPEncoding -i --autoreload
```

**Render to video** (HD):
```bash
manimgl clip_encoding.py CLIPEncoding -w --hd
```

Output video will be saved to `videos/CLIPEncoding.mp4`.

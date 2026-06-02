# Vision-Based Access Intelligence

A lightweight pipeline for vehicle detection and license-plate recognition, with optional actuator control and access decision logic. The project bundles detection models, OCR tooling, and pipeline runners for image and video inputs.

Quick highlights
- Detection models: `models/vehicle_detector.*`, `models/plate_detector.*`
- OCR: `models/license_recognizer/*` and `src/vision/plate_ocr.py`
- Pipeline runners: `src/pipeline/run_image_pipeline.py`, `src/pipeline/run_video_pipeline.py`
- Actuator support: `src/actuator/gpio_relay_actuator.py` and `src/actuator/mock_actuator.py`

Supported platforms
- Linux / Raspberry Pi (ARM64) — recommended for deployment
- Desktop Linux / macOS / Windows — supported for development and testing

Prerequisites
- Python 3.12 is declared in `pyproject.toml`; Python 3.11 is a known working alternative.
- System libs for OpenCV and native wheels (varies by OS). On Debian/Ubuntu/Raspbian install common build deps such as `libjpeg-dev` and `libopenblas-dev`.

Quickstart (Linux / Raspberry Pi)
1. From the project root, create a virtual environment and activate it:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Notes on heavy deps
- If `torch` or `paddlepaddle` fails to install on ARM64, obtain the platform-specific wheels from the official projects and install manually.
- Hailo acceleration requires the Hailo runtime and converted models (not included).

Verify models

```bash
ls models/vehicle_detector.pt models/plate_detector.pt models/license_recognizer/lprnet.pth
```

Running
- Run a single-image pipeline (uses `data/pipeline_test/vehicles` by default):

```bash
python -m src.pipeline.run_image_pipeline
```

- Run the video / camera pipeline:

```bash
python -m src.pipeline.run_video_pipeline
```

Environment / configuration
- The project reads configuration via `src/configs/settings.py` and `configs/settings.yaml`.
- Common env vars:
	- `ACTUATOR__MODE` — `mock` or `gpio`
	- `ACTUATOR__RELAY_PIN` — GPIO pin for relay
	- `VIDEO__DISPLAY` — `False` for headless mode

Database
- Bootstrap the local DB (sqlite by default):

```bash
python -m src.db.bootstrap_db
```

Where key code lives
- Detection and OCR: `src/vision/` (`vehicle_detector.py`, `plate_detector.py`, `plate_ocr.py`)
- Pipeline orchestration: `src/pipeline/`
- Actuators: `src/actuator/`
- Decision logic: `src/decision/` (`rules_engine.py`, `temporal_voter.py`)
- Logging: `src/logging/event_logger.py`

Developer notes
- The repo includes Jupyter notebooks under `notebooks/` for experiments and model finetuning.
- Model formats: `.pt` (PyTorch), `.onnx`, and Hailo `.hef` variants are stored in `models/`.

Raspberry Pi / Hailo notes
- The core Python code is platform-independent; on Raspberry Pi you must use ARM64-compatible wheels and optionally install the Hailo runtime for accelerator support.
- To add Hailo support: convert models and implement an adapter under `src/vision/` (see `hailo_yolo_detector.py` as a starting point).



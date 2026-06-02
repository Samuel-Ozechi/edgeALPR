# Docker & Deployment Guide

Quick start guide for running edgeALPR with Docker or on bare Raspberry Pi hardware.

## Docker Quick Start

### Prerequisites

- Docker and Docker Compose installed
- For Raspberry Pi: Docker for ARM64

### Run with Docker Compose (Development)

```bash
# Start edgeALPR container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Environment Variables

Configure via `docker-compose.yml` or `.env` file:

```bash
# Detection backend: pytorch (default), onnx, or hailo
DETECTOR_BACKEND=pytorch

# Confidence thresholds
VEHICLE_CONF_THRESHOLD=0.45
PLATE_CONF_THRESHOLD=0.5
OCR_CONFIDENCE_THRESHOLD=0.8

# Access control
ENABLE_CLASS_MATCHING=true
VEHICLE_TARGET_CLASSES=car,truck,bus,motorcycle
```

### Volume Mounting

```bash
# Mount local data directory
docker run -v $(pwd)/data:/app/data edgealpr:latest

# Mount USB camera
docker run --device /dev/video0:/dev/video0 edgealpr:latest
```

## Raspberry Pi Native Installation

### Automated Setup

```bash
# Download and run setup script
curl -O https://raw.githubusercontent.com/YOUR_ORG/edgeALPR/main/scripts/setup-pi.sh
chmod +x setup-pi.sh
./setup-pi.sh

# With optional features
./setup-pi.sh --onnx --hailo
```

### Manual Setup

```bash
# Prerequisites
sudo apt-get install python3.12 python3.12-venv python3.12-dev
sudo apt-get install build-essential libsqlite3-dev libssl-dev libffi-dev

# Clone repository
git clone https://github.com/YOUR_ORG/edgeALPR.git
cd edgeALPR

# Setup virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/unit/ -v
```

### Start as System Service

```bash
# Start service
sudo systemctl start edgealpr

# Enable on boot
sudo systemctl enable edgealpr

# View logs
journalctl -u edgealpr -f

# Stop service
sudo systemctl stop edgealpr
```

## Hardware Integration

### GPIO Relay Control

For controlling physical barriers, gates, or warning lights:

```bash
# Install GPIO support
pip install gpiozero

# Run GPIO example
python examples/with_gpio_relay.py
```

See `examples/with_gpio_relay.py` for wiring diagram and implementation details.

### Hailo Accelerator Setup

For optimized inference on Hailo-8 edge accelerator:

```bash
# Install Hailo SDK
bash ./scripts/setup-pi.sh --hailo

# Configure to use Hailo backend
export DETECTOR_BACKEND=hailo

# Run with Hailo optimization
python -m src.pipeline.run_video_pipeline
```

### ONNX Runtime CPU/GPU Inference

For cross-platform inference without specialized hardware:

```bash
# Install ONNX Runtime
pip install onnxruntime

# Configure backend
export DETECTOR_BACKEND=onnx

# Run with ONNX
python -m src.pipeline.run_video_pipeline
```

## Building Docker Images

### Raspberry Pi (ARM64)

```bash
# Build for Pi
docker build -f Dockerfile.pi -t edgealpr:pi .

# Build with BuildKit for multi-platform
docker buildx build -f Dockerfile.pi -t edgealpr:pi --platform linux/arm64 .
```

### Push to Registry

```bash
# Tag image
docker tag edgealpr:pi YOUR_REGISTRY/edgealpr:pi

# Push
docker push YOUR_REGISTRY/edgealpr:pi
```

## Configuration

### settings.yaml

Main configuration file at `src/configs/settings.yaml`:

```yaml
# Model paths
models:
  vehicle_detector: models/vehicle_detector.pt
  plate_detector: models/plate_detector.pt
  vehicle_detector_onnx: models/vehicle_detector.onnx
  plate_detector_onnx: models/plate_detector.onnx
  vehicle_detector_hef: models/plate_detector.hef

# Detection thresholds
thresholds:
  vehicle_conf: 0.45
  plate_conf: 0.5
  ocr_conf: 0.8

# Runtime configuration
runtime:
  detector_backend: pytorch  # pytorch, onnx, or hailo
  use_gpu: false
```

### Database

SQLite database at `data/edgealpr.db` with tables:

- `vehicles`: Authorized vehicle database
- `access_events`: Access control audit trail

## Troubleshooting

### Container fails to start

```bash
# Check logs
docker-compose logs edgealpr

# Rebuild image
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### No camera detected

```bash
# List available video devices
ls -la /dev/video*

# Mount specific device
docker run --device /dev/video0:/dev/video0 edgealpr:latest
```

### Out of memory

Reduce memory usage:

```bash
# Disable GPU (uses more RAM)
export DETECTOR_BACKEND=pytorch

# Reduce batch size in settings
batch_size: 1

# Use ONNX for lower memory footprint
export DETECTOR_BACKEND=onnx
```

### Poor performance

```bash
# Enable Hailo accelerator (best performance)
export DETECTOR_BACKEND=hailo

# Use quantized models
# Available in models/ directory

# Check inference metrics
python -c "from src.health.metrics import get_metrics_collector; print(get_metrics_collector().get_summary())"
```

## Examples

See `examples/` directory:

1. **minimal_plate_detection.py** - Simple plate detection demo (20 lines)
2. **with_access_control.py** - Full access control workflow (100 lines)
3. **with_gpio_relay.py** - Hardware relay integration

Run examples:

```bash
# Minimal detection
python examples/minimal_plate_detection.py path/to/image.jpg

# Full workflow
python examples/with_access_control.py path/to/image.jpg

# GPIO control (requires hardware)
python examples/with_gpio_relay.py
```

## Health Checks

Monitor system health:

```python
from src.health.health_check import HealthCheck
from src.health.metrics import get_metrics_collector

checker = HealthCheck("data/edgealpr.db")
status = checker.get_full_health_status({
    "vehicle_detector": "models/vehicle_detector.pt",
    "plate_detector": "models/plate_detector.pt"
})

print(status)  # {overall_status: "healthy", database: {...}, models: {...}}

# Get performance metrics
metrics = get_metrics_collector()
print(metrics.get_summary())
```

## Performance Notes

### Latency (on Raspberry Pi 5)

- Vehicle detection: 200-300ms (PyTorch)
- Plate detection: 50-100ms (PyTorch)
- OCR recognition: 150-250ms
- **Total**: ~400-650ms per frame

With Hailo accelerator: ~100-150ms per frame (4-5x speedup)

### Memory Usage

- PyTorch backend: ~300MB
- ONNX backend: ~150MB
- With Hailo: ~200MB

## Support & Debugging

- Check logs: `journalctl -u edgealpr -f`
- View coverage: `open htmlcov/index.html` (after running tests)
- Run tests: `python -m pytest tests/unit/ -v --cov=src`
- Health status: `curl http://localhost:8000/health` (if API enabled)

## Next Steps

1. Deploy to Raspberry Pi with `setup-pi.sh`
2. Configure database with authorized vehicles
3. Test with sample images in `data/pipeline_test/`
4. Integrate GPIO relay for physical access control
5. Monitor performance with metrics dashboard

See README.md for architecture overview and API documentation.

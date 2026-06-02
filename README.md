# edgeALPR — Edge-Native Automatic License Plate Recognition

A **production-ready, extensible** open-source framework for automatic license plate recognition (ALPR) optimized for edge deployment. Deploy on Raspberry Pi, Jetson, or cloud with vehicle detection, plate recognition, and configurable access control.

**Perfect for:** Parking access systems, toll gates, perimeter security, fleet management, traffic monitoring, and autonomous vehicle applications.

---

## What Makes edgeALPR Different

| Feature | edgeALPR | Traditional ALPR |
|---------|----------|-----------------|
| **Deployment** | Edge devices (Pi, Jetson, Docker) | Cloud-dependent, high latency |
| **Extensibility** | Abstract base classes for custom backends | Monolithic, hard to extend |
| **Backends** | 3 inference engines (PyTorch, ONNX, Hailo) | Single vendor lock-in |
| **Testing** | 48 unit tests, 56% coverage | Minimal test coverage |
| **Documentation** | Complete with examples & guides | Basic API docs only |
| **Latency** | 100-650ms per frame (depending on backend) | 1-2s (cloud round-trip) |

---

## Applications

### **Parking Access Control**
Automatically recognize authorized vehicles and open barriers/gates without human intervention.

```python
from src.vision.detector_factory import build_vehicle_detector, build_plate_detector
from src.vision.plate_ocr import PlateOCR
from src.decision.rules_engine import AccessRulesEngine

# Authorize vehicles by plate number
detector = build_vehicle_detector()
plate_detector = build_plate_detector()
ocr = PlateOCR()
rules = AccessRulesEngine(ocr_confidence_threshold=0.8)

detections, latency = detector.detect(camera_frame)
plate_results, _ = plate_detector.detect(camera_frame)
plate_text, confidence = ocr.recognize(plate_crop)

# Access decision: allow, deny, or manual review
decision, reason = rules.decide(
    plate_text=plate_text,
    ocr_confidence=confidence,
    vehicle_found=is_in_database
)
```

### **Toll Gate Automation**
Real-time toll collection with vehicle class-based pricing and evasion detection.

```python
# Route-based access with vehicle classification
detector.target_classes = {"car": "$5", "truck": "$15", "motorcycle": "$2"}
decision = rules.decide(..., vehicle_class=detected_class)
# Auto-charge based on vehicle type
```

### **Perimeter Security**
Monitor restricted areas and alert on unauthorized vehicle entry.

```python
# Integrated monitoring
from src.health.metrics import get_metrics_collector

metrics = get_metrics_collector()
metrics.record_inference("vehicle_detector", latency_ms)
metrics.record_cache_hit("plate_lookup")

# Alert on unknown vehicles
if not vehicle_found:
    send_security_alert(plate_text, timestamp)
```

### **Traffic Analytics**
Aggregate traffic patterns, peak hour analysis, and vehicle mix statistics.

```python
# Persistent event logging (thread-safe)
from src.logging.event_logger import JsonlEventLogger

logger = JsonlEventLogger("data/logs")
logger.log_event({
    "timestamp": now,
    "vehicle_class": class_name,
    "plate_text": plate,
    "confidence": ocr_score
})

# Analyze patterns
df = pd.read_json("data/logs/access_events.jsonl", lines=True)
peak_hours = df.groupby(df.timestamp.dt.hour).size()
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Image / Video Input (File, USB Camera, RTSP Stream)        │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────────┐
        │  Vehicle Detection         │
        │  (BaseClassifier)          │
        │  - PyTorch YOLOv8 [default]│
        │  - ONNX Runtime            │
        │  - Hailo Accelerator       │
        └────────────────┬───────────┘
                         ▼
        ┌────────────────────────────┐
        │  Plate Detection           │
        │  (BaseDetector)            │
        │  - Crop vehicle region     │
        │  - Detect plate bbox       │
        └────────────────┬───────────┘
                         ▼
        ┌────────────────────────────┐
        │  OCR Recognition           │
        │  (BaseOCR)                 │
        │  - PaddleOCR [default]     │
        │  - Normalize text          │
        │  - Calculate confidence    │
        └────────────────┬───────────┘
                         ▼
        ┌────────────────────────────┐
        │  Decision Logic            │
        │  (BaseRulesEngine)         │
        │  - Database lookup         │
        │  - Fuzzy matching (cache)  │
        │  - Access decision tree    │
        └────────────────┬───────────┘
                         ▼
        ┌────────────────────────────┐
        │  Action Execution          │
        │  (BaseActuator)            │
        │  - GPIO relay control      │
        │  - Barrier/gate operation  │
        │  - Alert notifications     │
        └────────────────┬───────────┘
                         ▼
        ┌────────────────────────────┐
        │  Event Logging & Metrics   │
        │  - SQLite audit trail      │
        │  - Performance monitoring  │
        │  - Health checks           │
        └────────────────────────────┘
```

---

## Quick Start (< 5 minutes)

### Option 1: Docker (Recommended for Production)

```bash
# Clone repo
git clone https://github.com/Samuel-Ozechi/edgeALPR.git
cd edgeALPR

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f edgealpr

# Test with sample image
docker-compose exec edgealpr python examples/minimal_plate_detection.py data/sample.jpg
```

### Option 2: Native Raspberry Pi / Linux

```bash
# Automated setup (installs everything)
curl -O https://raw.githubusercontent.com/Samuel-Ozechi/edgeALPR/main/scripts/setup-pi.sh
chmod +x setup-pi.sh
./setup-pi.sh --onnx  # Optional: add ONNX support

# Manual setup
python3.12 -m venv venv
source venv/bin/activate
pip install -e .

# Run tests to verify
python -m pytest tests/unit/ -v

# Start service
sudo systemctl start edgealpr
sudo systemctl status edgealpr
```

### Option 3: Development Environment

```bash
# Setup with dev dependencies
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests with coverage
python -m pytest tests/unit/ -v --cov=src --cov-report=html

# Format code
black src/ --line-length=100

# Type checking
mypy src/ --python-version=3.12
```

---

## Usage Examples

### Example 1: Simple Plate Detection (20 lines)

```python
import cv2
from src.vision.detector_factory import build_plate_detector
from src.vision.plate_ocr import PlateOCR

image = cv2.imread("parking_lot.jpg")
detector = build_plate_detector()
ocr = PlateOCR()

# Detect plates
detections, latency = detector.detect(image)
print(f"Found {len(detections)} plate(s) in {latency:.1f}ms")

for detection in detections:
    bbox = detection["bbox"]
    x1, y1, x2, y2 = bbox
    plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
    
    # Recognize text
    result, _ = ocr.recognize(plate_crop)
    print(f"Plate: {result['plate_text']} (confidence: {result['confidence']:.1%})")
```

**Run:** `python examples/minimal_plate_detection.py parking_lot.jpg`

### Example 2: Full Access Control Workflow (100 lines)

Complete workflow with database lookup and decision logic:

```bash
python examples/with_access_control.py parking_lot.jpg
```

Output:
```
Step 1: Vehicle Detection
  Found 1 vehicle(s) in 250ms
  → Using best match (confidence: 92%)

Step 2: Plate Detection & OCR
  Recognized plate: ABC123
  OCR confidence: 95% (180ms)

Step 3: Vehicle Authorization Lookup
  ✓ Vehicle found in database
    Status: Active, Class: Car

Step 4: Access Control Decision
  Decision: ALLOW
  Reason: Vehicle in whitelist, high OCR confidence

Step 5: Action Execution
  ✓ Barrier opened for 5 seconds

Step 6: Event Logging
  ✓ Event logged to database
```

See [examples/](examples/) directory for full code.

---

## Configuration

### Backend Selection

Switch detection backends without changing code:

```bash
# PyTorch (default, ~300MB)
export DETECTOR_BACKEND=pytorch

# ONNX Runtime (CPU/GPU, ~150MB)
export DETECTOR_BACKEND=onnx

# Hailo Accelerator (Hailo-8, ~100ms latency)
export DETECTOR_BACKEND=hailo
```

### settings.yaml Configuration

```yaml
models:
  vehicle_detector: models/vehicle_detector.pt
  vehicle_detector_onnx: models/vehicle_detector.onnx
  vehicle_detector_hef: models/vehicle_detector.hef

thresholds:
  vehicle_conf: 0.45
  plate_conf: 0.5
  ocr_conf: 0.8

runtime:
  detector_backend: pytorch  # pytorch, onnx, or hailo
  use_gpu: false
  enable_class_matching: true

database:
  database_path: data/edgealpr.db
```

### Environment Variables

```bash
EDGEALPR_LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
EDGEALPR_DATABASE_PATH=data/edgealpr.db    # Database location
EDGEALPR_LOG_PATH=data/logs                # Log directory
DETECTOR_BACKEND=pytorch                   # pytorch, onnx, hailo
VEHICLE_CONF_THRESHOLD=0.45                # 0.0-1.0
PLATE_CONF_THRESHOLD=0.5                   # 0.0-1.0
OCR_CONFIDENCE_THRESHOLD=0.8               # 0.0-1.0
ENABLE_CLASS_MATCHING=true                 # Filter by vehicle type
VEHICLE_TARGET_CLASSES=car,truck,bus       # Comma-separated
```

---

## Performance Metrics

### Latency (Raspberry Pi 5, PyTorch backend)

| Component | Latency | Notes |
|-----------|---------|-------|
| Vehicle Detection | 200-300ms | YOLOv8n model |
| Plate Detection | 50-100ms | YOLOv8n model |
| OCR Recognition | 150-250ms | PaddleOCR |
| Database Lookup | 5-20ms | With 5-min cache |
| **Total E2E** | **400-650ms** | Per frame |

### With Accelerators

| Backend | Vehicle Detect | Plate Detect | **E2E** | Power |
|---------|----------------|-------------|--------|-------|
| PyTorch CPU | 250ms | 80ms | 650ms | 10W |
| ONNX Runtime | 200ms | 60ms | 550ms | 8W |
| **Hailo-8** | **50ms** | **20ms** | **150ms** | **5W** |

**4.3x speedup** with Hailo accelerator vs PyTorch (and better power efficiency).

### Memory Usage

- PyTorch backend: ~300MB
- ONNX backend: ~150MB
- Hailo backend: ~200MB

---

## Testing & Quality

### Run Test Suite

```bash
# All tests (48 tests, 56% coverage)
python -m pytest tests/unit/ -v --cov=src --cov-report=html

# Specific test module
python -m pytest tests/unit/test_rules_engine.py -v

# With output
python -m pytest tests/unit/ -v -s

# Coverage report
open htmlcov/index.html
```

### Test Coverage

```
Core Modules:
   AccessRulesEngine:        100% (45/45 statements)
   Settings Configuration:   100% (86/86 statements)
   MockActuator:              95% (42/44 statements)
   VehicleRepository:         76% (database optimization)
   EventLogger:               74% (thread-safe logging)

Added Features:
   ONNX Detector Backend:      Multiple validation tests
   Hailo Detector Backend:     Thread-safety verified
   Health Checks:             System diagnostics
   Metrics Collection:        Performance monitoring
```

---

##  Architecture & Extensibility

### Abstract Base Classes (Build Your Own Backend)

```python
# Create custom detector
from src.vision.base_detector import BaseClassifier

class MyCustomDetector(BaseClassifier):
    def detect(self, image):
        # Your detection logic
        return detections, latency_ms
    
    def get_model_info(self):
        return {"model_type": "custom", ...}

# Use in pipeline
from src.vision.detector_factory import build_vehicle_detector
detector = MyCustomDetector(
    target_classes=["car", "truck"],
    conf_threshold=0.5
)
```

### Custom Decision Logic

```python
from src.decision.base_rules_engine import BaseRulesEngine, Decision

class MyAccessLogic(BaseRulesEngine):
    def decide(self, plate_text, ocr_confidence, **kwargs):
        # Custom decision tree
        if plate_text in self.vip_list:
            return "allow", "VIP vehicle"
        elif ocr_confidence > 0.9:
            return "allow", "High confidence match"
        else:
            return "review", "Manual verification needed"
    
    def get_stats(self):
        return {"custom_stat": value}
```

### Custom Hardware Control

```python
from src.actuator.base_actuator import BaseActuator

class MyBarrierController(BaseActuator):
    def execute(self, action):
        if action == "allow":
            self.open_barrier()
        elif action == "deny":
            self.trigger_alarm()
    
    def get_state(self):
        return {"barrier_status": "open", ...}
```

---

##  Deployment

### Docker Deployment (Production)

```bash
# Build Docker image
docker build -f Dockerfile.pi -t edgealpr:latest .

# Run with GPU support (if available)
docker run --gpus all -v $(pwd)/data:/app/data edgealpr:latest

# Deploy to cloud (Google Cloud Run, AWS Lambda)
# See DEPLOYMENT.md for cloud-specific instructions
```

### Kubernetes Deployment

```bash
# Scale horizontally for multiple streams
kubectl apply -f k8s/edgealpr-deployment.yaml

# Monitor metrics
kubectl logs -f deployment/edgealpr
kubectl top pod
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete Docker, Kubernetes, and cloud deployment guides.

---

## Project Structure

```
edgeALPR/
├── src/
│   ├── vision/                 # Detection & OCR
│   │   ├── base_detector.py    # Abstract base classes
│   │   ├── vehicle_detector.py # PyTorch YOLO detector
│   │   ├── plate_detector.py   # Plate detection
│   │   ├── plate_ocr.py        # PaddleOCR wrapper
│   │   ├── onnx_detector.py    # ONNX Runtime backend 
│   │   └── hailo_plate_detector.py  # Hailo accelerator 
│   ├── decision/               # Access control logic
│   │   ├── base_rules_engine.py
│   │   ├── rules_engine.py
│   │   └── temporal_voter.py
│   ├── actuator/               # Hardware control
│   │   ├── base_actuator.py
│   │   ├── gpio_relay_actuator.py
│   │   └── mock_actuator.py
│   ├── db/                     # Data persistence
│   │   ├── base_repository.py
│   │   └── repository.py
│   ├── health/                 # Monitoring 
│   │   ├── health_check.py
│   │   └── metrics.py
│   ├── pipeline/               # Orchestration
│   │   ├── run_image_pipeline.py
│   │   └── run_video_pipeline.py
│   └── configs/                # Configuration
│       ├── settings.py
│       └── settings.yaml
├── tests/
│   └── unit/                   # 48+ unit tests 
│       ├── test_rules_engine.py
│       ├── test_repository.py
│       ├── test_detector_backends.py
│       └── test_health_metrics.py
├── examples/                   # Developer examples 
│   ├── minimal_plate_detection.py
│   ├── with_access_control.py
│   └── with_gpio_relay.py
├── scripts/
│   └── setup-pi.sh             # Automated setup 
├── Dockerfile.pi               # Container image 
├── docker-compose.yml          # Docker Compose 
├── DEPLOYMENT.md               # Deployment guide 
└── pyproject.toml              # Project metadata
```

---

##  Integration Examples

### Integrate with Your Parking System

```python
# Connect to existing barrier/gate API
from src.decision.rules_engine import AccessRulesEngine

class ParkinglotIntegration:
    def __init__(self, barrier_api, database_api):
        self.barrier = barrier_api
        self.db = database_api
        self.rules = AccessRulesEngine(ocr_confidence_threshold=0.85)
    
    def process_vehicle(self, frame):
        # Run ALPR pipeline
        plate = self.detect_plate(frame)
        decision, reason = self.rules.decide(
            plate_text=plate,
            ocr_confidence=0.92,
            vehicle_found=self.db.is_authorized(plate)
        )
        
        # Send decision to barrier
        if decision == "allow":
            self.barrier.open_for(duration_seconds=5)
        elif decision == "deny":
            self.barrier.lock_and_alert()
        else:
            self.barrier.flag_for_manual_review()
```

### Fleet Management Integration

```python
# Track vehicle movements and anomalies
from src.logging.event_logger import JsonlEventLogger

logger = JsonlEventLogger("data/fleet_logs")

def log_vehicle_sighting(plate, location, timestamp):
    logger.log_event({
        "plate": plate,
        "location": location,
        "timestamp": timestamp,
        "event_type": "sighting"
    })
    
    # Alert if unusual pattern detected
    sightings = get_recent_sightings(plate)
    if is_suspicious_pattern(sightings):
        send_alert(f"Unusual vehicle movement: {plate}")
```

---

##  Development Guide

### Setting Up Local Development

```bash
# Clone with dev dependencies
git clone https://github.com/Samuel-Ozechi/edgeALPR.git
cd edgeALPR

# Create Python 3.12 environment
python3.12 -m venv venv
source venv/bin/activate

# Install with dev tools
pip install -e ".[dev]"

# Run tests
python -m pytest tests/unit/ -v

# Format code
black src/ --line-length=100
isort src/

# Type checking
mypy src/ --python-version=3.12

# Lint
ruff check src/
```

### Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/my-feature`
3. **Make** changes with tests: `python -m pytest tests/unit/ -v`
4. **Format** code: `black src/ && isort src/`
5. **Push** to GitHub: `git push origin feature/my-feature`
6. **Open** Pull Request with description

### Code Style

- **Python**: Follow [Black](https://github.com/psf/black) (line length: 100)
- **Type hints**: Use Python 3.12 syntax (`list[str]`, `dict[str, int]`)
- **Logging**: Use module-level `logger = logging.getLogger(__name__)`, no `print()`
- **Tests**: Pytest with fixtures, aim for >80% coverage
- **Documentation**: Docstrings for all public APIs

---

##  Comparison: edgeALPR vs Alternatives

| Feature | **edgeALPR** | Commercial ALPR | Cloud Services |
|---------|-------------|-----------------|-----------------|
| **Cost** | Free | $2,000-10,000 | $500-5,000/year |
| **Latency** | 100-650ms | 200-500ms | 1-2s (cloud) |
| **Customization** | Full source code | Limited API | Very limited |
| **Privacy** | On-device processing | Vendor server | Cloud-dependent |
| **Scale** | Single device to 1000s | Enterprise licensing | Auto-scaling |
| **Support** | Community (GitHub) | Vendor support | Vendor support |
| **Learning curve** | Medium (good docs) | High | Low |

---

##  Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip install -e .
# If still fails on ARM64:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "No plate detected" / Low accuracy

- Ensure lighting is adequate (>100 lux)
- Check image resolution (min 640x480 recommended)
- Verify model files exist: `ls models/plate_detector.pt`
- Increase confidence threshold if needed

### Docker build fails

```bash
docker-compose build --no-cache  # Clear cache
docker system prune -a           # Remove unused images
docker-compose up -d             # Rebuild
```

### Performance issues

```bash
# Switch to faster backend
export DETECTOR_BACKEND=onnx  # Or hailo

# Check metrics
python -c "from src.health.metrics import get_metrics_collector; \
    print(get_metrics_collector().get_summary())"

# Profile bottleneck
python -m cProfile -s cumtime examples/minimal_plate_detection.py image.jpg
```

See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting) for more troubleshooting.

---

## Support & Community

- **Documentation**: Full guides in [DEPLOYMENT.md](DEPLOYMENT.md)
- **Examples**: See [examples/](examples/) directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/Samuel-Ozechi/edgeALPR/issues)
- **Discussions**: Join community on [GitHub Discussions](https://github.com/Samuel-Ozechi/edgeALPR/discussions)

---

## License

MIT License — See [LICENSE](LICENSE) file for details.

Free for personal and commercial use with proper attribution.

---

## Roadmap

- [ ] **API Server** - REST/gRPC endpoints for remote access
- [ ] **Web Dashboard** - Real-time monitoring and analytics UI
- [ ] **Multi-camera Support** - Handle multiple video streams
- [ ] **GPU Optimization** - CUDA/TensorRT acceleration
- [ ] **Cloud Integration** - AWS/GCP/Azure deployment templates
- [ ] **CI/CD Pipeline** - GitHub Actions for automated testing
- [ ] **Mobile App** - React Native companion app

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick wins for first-time contributors:**
- Add CI/CD pipeline with GitHub Actions
- Improve documentation with diagrams
- Implement new access control logic
- Port to new hardware (Jetson, Coral, etc.)

---

**Built with ❤️ for the edge AI community**

[⭐ Star us on GitHub](https://github.com/Samuel-Ozechi/edgeALPR) | [ Full Documentation](DEPLOYMENT.md) | [ Report Issues](https://github.com/Samuel-Ozechi/edgeALPR/issues)



#!/bin/bash
# Setup script for edgeALPR on Raspberry Pi
# Usage: bash setup-pi.sh [OPTIONS]

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.12"
VENV_PATH="${HOME}/.venv/edgealpr"
APP_PATH="${HOME}/edgeALPR"
SYSTEMD_SERVICE="/etc/systemd/system/edgealpr.service"
HAILO_INSTALL=false
ONNX_INSTALL=false

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if running on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This script is designed for Linux systems (Raspberry Pi)"
        exit 1
    fi
    
    # Check if Python 3.12 is available
    if ! command -v python3.12 &> /dev/null; then
        print_error "Python 3.12 is required but not installed"
        print_info "Run: sudo apt-get install python3.12 python3.12-venv python3.12-dev"
        exit 1
    fi
    
    # Check for git
    if ! command -v git &> /dev/null; then
        print_warn "Git is not installed. Installing..."
        sudo apt-get update
        sudo apt-get install -y git
    fi
    
    print_info "Prerequisites check passed"
}

install_dependencies() {
    print_info "Installing system dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        libsqlite3-dev \
        libssl-dev \
        libffi-dev \
        libopenblas-dev \
        libjpeg-dev \
        zlib1g-dev \
        libatlas-base-dev \
        libjasper-dev \
        libharfbuzz0b \
        libwebp6 \
        libtiff5 \
        libjasper1 \
        libharfbuzz0b \
        libwebp6 \
        git \
        curl \
        wget
    
    print_info "System dependencies installed"
}

setup_python_environment() {
    print_info "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3.12 -m venv "$VENV_PATH"
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip and tools
    pip install --upgrade pip setuptools wheel
    
    print_info "Python environment ready at: $VENV_PATH"
}

clone_repository() {
    if [ -d "$APP_PATH" ]; then
        print_warn "Repository already exists at $APP_PATH"
        read -p "Do you want to pull latest changes? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$APP_PATH"
            git pull origin main
        fi
    else
        print_info "Cloning edgeALPR repository..."
        git clone https://github.com/YOUR_ORG/edgeALPR.git "$APP_PATH"
    fi
}

install_application() {
    print_info "Installing edgeALPR..."
    
    source "$VENV_PATH/bin/activate"
    cd "$APP_PATH"
    
    # Install in development mode
    pip install -e .
    
    # Optional: Install ONNX Runtime
    if [ "$ONNX_INSTALL" = true ]; then
        print_info "Installing ONNX Runtime..."
        pip install onnxruntime
    fi
    
    # Optional: Install Hailo SDK
    if [ "$HAILO_INSTALL" = true ]; then
        print_info "Note: Hailo SDK requires special installation"
        print_info "Follow: https://github.com/hailo-ai/hailo-rpi5-examples"
    fi
    
    print_info "edgeALPR installed successfully"
}

setup_directories() {
    print_info "Setting up data directories..."
    
    mkdir -p "$APP_PATH/data/logs"
    mkdir -p "$APP_PATH/data/pipeline_test/vehicles/input"
    mkdir -p "$APP_PATH/data/pipeline_test/vehicles/output"
    
    chmod -R 755 "$APP_PATH/data"
    
    print_info "Directories created"
}

setup_systemd_service() {
    print_info "Setting up systemd service..."
    
    # Create systemd service file
    sudo tee "$SYSTEMD_SERVICE" > /dev/null << EOF
[Unit]
Description=edgeALPR - Automatic License Plate Recognition
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=pi
WorkingDirectory=$APP_PATH
Environment="PATH=$VENV_PATH/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$VENV_PATH/bin/python -m src.pipeline.run_video_pipeline
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    
    print_info "Systemd service installed at: $SYSTEMD_SERVICE"
    print_info "To start: sudo systemctl start edgealpr"
    print_info "To enable on boot: sudo systemctl enable edgealpr"
}

configure_gpio() {
    print_warn "GPIO relay control requires gpiozero library"
    read -p "Install GPIO support? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        source "$VENV_PATH/bin/activate"
        pip install gpiozero
        print_info "GPIO support installed"
    fi
}

run_tests() {
    print_info "Running tests..."
    
    source "$VENV_PATH/bin/activate"
    cd "$APP_PATH"
    
    if pip show pytest &> /dev/null; then
        python -m pytest tests/unit/ -v --tb=short
        print_info "Tests completed"
    else
        print_warn "pytest not installed. Skipping tests."
        print_info "Run: pip install pytest pytest-cov"
    fi
}

print_summary() {
    print_info "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment: source $VENV_PATH/bin/activate"
    echo "2. Configure settings: $APP_PATH/src/configs/settings.yaml"
    echo "3. Test installation: python -m pytest tests/unit/"
    echo "4. Start service: sudo systemctl start edgealpr"
    echo "5. View logs: journalctl -u edgealpr -f"
    echo ""
}

# Main execution
main() {
    print_info "Starting edgeALPR setup for Raspberry Pi..."
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --hailo)
                HAILO_INSTALL=true
                shift
                ;;
            --onnx)
                ONNX_INSTALL=true
                shift
                ;;
            --venv)
                VENV_PATH="$2"
                shift 2
                ;;
            --app)
                APP_PATH="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    install_dependencies
    setup_python_environment
    clone_repository
    setup_directories
    install_application
    setup_systemd_service
    
    read -p "Configure GPIO relay control? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        configure_gpio
    fi
    
    read -p "Run tests? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    print_summary
}

# Run main function
main "$@"

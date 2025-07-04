# GPU Verification Docker Container
# Based on NVIDIA CUDA runtime image
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for GPU verification
RUN pip3 install --no-cache-dir \
    torch \
    tensorflow \
    numpy \
    gpustat

# Create verification script
RUN cat > /usr/local/bin/verify-gpu.py << 'EOF'
#!/usr/bin/env python3
"""
GPU Verification Script for Docker Containers
"""
import subprocess
import sys
import os

def check_nvidia_smi():
    print("🔍 Checking nvidia-smi...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi: SUCCESS")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi: FAILED")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi: ERROR - {e}")
        return False

def check_cuda_devices():
    print("\n🔍 Checking CUDA device files...")
    cuda_devices = []
    if os.path.exists('/dev'):
        for item in os.listdir('/dev'):
            if item.startswith('nvidia'):
                cuda_devices.append(f"/dev/{item}")
    
    if cuda_devices:
        print("✅ CUDA devices found:")
        for device in sorted(cuda_devices):
            print(f"   {device}")
        return True
    else:
        print("❌ No CUDA devices found")
        return False

def check_pytorch():
    print("\n🔍 Checking PyTorch CUDA...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        return False
    except ImportError:
        print("❌ PyTorch not available")
        return False

def check_tensorflow():
    print("\n🔍 Checking TensorFlow GPU...")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices: {len(gpus)}")
        for gpu in gpus:
            print(f"   {gpu}")
        return len(gpus) > 0
    except ImportError:
        print("❌ TensorFlow not available")
        return False

def main():
    print("🚀 GPU Verification in Docker Container")
    print("=" * 50)
    
    results = []
    results.append(check_nvidia_smi())
    results.append(check_cuda_devices())
    results.append(check_pytorch())
    results.append(check_tensorflow())
    
    print("\n" + "=" * 50)
    if any(results):
        print("✅ GPU ACCESS: VERIFIED")
    else:
        print("❌ GPU ACCESS: FAILED")
        print("\nTroubleshooting tips:")
        print("1. Ensure NVIDIA Docker runtime is installed")
        print("2. Run with: docker run --gpus all ...")
        print("3. Check host GPU drivers: nvidia-smi")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
RUN chmod +x /usr/local/bin/verify-gpu.py

# Create a simple test script
RUN cat > /usr/local/bin/gpu-test.sh << 'EOF'
#!/bin/bash
echo "🔧 Quick GPU Tests"
echo "=================="

echo "1. nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi failed"

echo -e "\n2. CUDA devices:"
ls -la /dev/nvidia* 2>/dev/null || echo "No CUDA devices found"

echo -e "\n3. Environment variables:"
env | grep -E "(CUDA|NVIDIA)" || echo "No CUDA/NVIDIA env vars"

echo -e "\n4. Running full verification..."
python3 /usr/local/bin/verify-gpu.py
EOF

RUN chmod +x /usr/local/bin/gpu-test.sh

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/usr/local/bin/gpu-test.sh"]

#!/usr/bin/env python3
"""
GPU Verification Script for Docker Containers
Checks if GPU is accessible through various methods
"""

import subprocess
import sys
import os

def check_nvidia_smi():
    """Check GPU using nvidia-smi command"""
    print("🔍 Checking nvidia-smi...")
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi: FOUND")
            # Extract GPU count from output
            lines = result.stdout.split('\n')
            gpu_lines = [line for line in lines if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line or 'GTX' in line]
            print(f"   GPUs detected: {len(gpu_lines)}")
            return True
        else:
            print("❌ nvidia-smi: NOT FOUND or ERROR")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi: COMMAND NOT AVAILABLE")
        return False

def check_cuda_devices():
    """Check CUDA device files"""
    print("\n🔍 Checking CUDA device files...")
    cuda_devices = []
    
    # Check for /dev/nvidia* devices
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
        print("❌ No CUDA devices found in /dev/")
        return False

def check_pytorch_gpu():
    """Check GPU availability through PyTorch"""
    print("\n🔍 Checking PyTorch GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ PyTorch CUDA: AVAILABLE ({gpu_count} GPU(s))")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
            return True
        else:
            print("❌ PyTorch CUDA: NOT AVAILABLE")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed")
        return False

def check_tensorflow_gpu():
    """Check GPU availability through TensorFlow"""
    print("\n🔍 Checking TensorFlow GPU support...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU: AVAILABLE ({len(gpus)} GPU(s))")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("❌ TensorFlow GPU: NOT AVAILABLE")
            return False
    except ImportError:
        print("⚠️  TensorFlow not installed")
        return False

def check_environment_variables():
    """Check relevant environment variables"""
    print("\n🔍 Checking environment variables...")
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'NVIDIA_VISIBLE_DEVICES', 
        'CUDA_DEVICE_ORDER',
        'CUDA_HOME',
        'CUDA_PATH'
    ]
    
    found_any = False
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
            found_any = True
        else:
            print(f"❌ {var}: not set")
    
    return found_any

def main():
    print("🚀 GPU Verification Script for Docker Containers")
    print("=" * 50)

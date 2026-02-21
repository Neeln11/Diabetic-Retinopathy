import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch imported: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")
    import traceback
    traceback.print_exc()

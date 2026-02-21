import os
import ctypes
import sys

# Get the venv root by going up one level from Scripts
venv_root = os.path.dirname(os.path.dirname(sys.executable))
site_packages = os.path.join(venv_root, 'Lib', 'site-packages')
torch_lib = os.path.join(site_packages, 'torch', 'lib')

print(f"Checking torch lib: {torch_lib}")
if not os.path.exists(torch_lib):
    print("❌ torch/lib not found")
    sys.exit(1)

# Add DLL directory for Python > 3.8
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(torch_lib)

dlls_to_check = [
    "libiomp5md.dll",
    "fbgemm.dll",
    "torch_python.dll"
]

for dll_name in dlls_to_check:
    dll_path = os.path.join(torch_lib, dll_name)
    print(f"Testing {dll_name}...")
    if not os.path.exists(dll_path):
        print(f"   ❌ File not found: {dll_path}")
        continue
    
    try:
        ctypes.CDLL(dll_path)
        print(f"   ✅ Loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")

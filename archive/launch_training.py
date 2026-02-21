"""
Training launcher with MONAI compatibility fixes for Python 3.12
"""
import os
import sys

# Disable MONAI's automatic submodule loading which causes Python 3.12 issues
os.environ['MONAI_SKIP_LOAD_ALL'] = '1'

# Suppress optional import warnings
os.environ['MONAI_SUPPRESS_WARNINGS'] = '1'

# Now import and run the training script
if __name__ == "__main__":
    # Import the train module
    from src import train
    
    # Run the training function
    train.train()

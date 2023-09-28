# What version of Python do you have?
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
# Check if GPU is available
gpu_available = tf.test.is_gpu_available()
#tf.config.list_physical_devices('GPU')
print("GPU available:", gpu_available)
import tensorflow as tf
tf.test.is_gpu_available()
from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
[x.name for x in local_device_protos if x.device_type == 'GPU']
print(local_device_protos)

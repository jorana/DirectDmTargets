print('Start import')
import DirectDmTargets as dddm
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numericalunits as nu
import wimprates as wr
import pandas as pd
import scipy
import os
import sys
import numba
print('Done import')
print("SYSTEM")
print(f"\tRunning on {sys.platform}")
print(f"\tPython version "+ sys.version.replace('\n',''))
print(f"\tPython installation {sys.executable}")
print("MODULES")
for module in [dddm, wr]:
    print(f'''\t{module.__name__}\n\t\tver.:\t{module.__version__}\n\t\tPath:\t{str(module).split('from')[-1].split('__init__')[0][2:]}\n''')
import os
import HyperGP
import numpy as np

def statistics_record(data, save_path):
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write('\t'.join(['min', 'max', 'mean', 'var', 'std']) + '\n')

    with open(save_path, "+a") as f:
        f.write('\t'.join([str(np.min(data)), str(np.max(data)), str(np.mean(data)), str(np.var(data)), str(np.std(data))]) + '\n')
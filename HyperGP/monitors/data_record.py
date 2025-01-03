import os
import HyperGP
import numpy as np

def statistics_record_initprint():
    print(f"|{"min":-^20}|{"max":-^20}|{"mean":-^20}|{"var":-^20}|{"std":-^20}|")
    print(f"|{'-'*20:^20}|{'-'*20:^20}|{'-'*20:^20}|{'-'*20:^20}|{'-'*20:^20}|")

def statistics_record(data, save_path=None):
    if save_path is not None:
        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                f.write('\t'.join(['min', 'max', 'mean', 'var', 'std']) + '\n')

        with open(save_path, "+a") as f:
            f.write('\t'.join([str(HyperGP.tensor.min(data)), str(HyperGP.tensor.max(data)), str(HyperGP.tensor.mean(data)), str(HyperGP.tensor.var(data)), str(HyperGP.tensor.std(data))]) + '\n')
    print(f"|{str(HyperGP.tensor.min(data)):^20}|{str(HyperGP.tensor.max(data)):^20}|{str(HyperGP.tensor.mean(data)):^20}|{str(HyperGP.tensor.var(data)):^20}|{str(HyperGP.tensor.std(data)):^20}|")

statistics_record.init = statistics_record_initprint
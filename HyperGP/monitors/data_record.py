import os
import HyperGP
import numpy as np

def statistics_record_initprint():
    print(f'|{"min":-^30}|{"max":-^30}|{"mean":-^30}|{"var":-^30}|{"std":-^30}|')
    print(f"|{'-'*30:^30}|{'-'*30:^30}|{'-'*30:^30}|{'-'*30:^30}|{'-'*30:^30}|")

def statistics_record(fits, save_path=None):
    if save_path is not None:
        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                f.write('\t'.join(['min', 'max', 'mean', 'var', 'std']) + '\n')

        with open(save_path, "+a") as f:
            f.write('\t'.join([float(HyperGP.tensor.min(fits)), float(HyperGP.tensor.max(fits)), float(HyperGP.tensor.mean(fits)), float(HyperGP.tensor.var(fits)), float(HyperGP.tensor.std(fits))]) + '\n')
    print(fits[0])
    return f"|{float(HyperGP.tensor.min(fits)):^30}|{float(HyperGP.tensor.max(fits)):^30}|{float(HyperGP.tensor.mean(fits)):^30}|{float(HyperGP.tensor.var(fits)):^30}|{float(HyperGP.tensor.std(fits)):^30}|" 

statistics_record.init = statistics_record_initprint
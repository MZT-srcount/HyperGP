import os
import HyperGP
import numpy as np

def statistics_record_initprint():
    print(f'|{"iteration":-^10}||{"min":-^15}|{"max":-^15}|{"mean":-^15}|{"var":-^15}|{"std":-^15}|')
    print(f"|{'-'*10:^10}||{'-'*15:^15}|{'-'*15:^15}|{'-'*15:^15}|{'-'*15:^15}|{'-'*15:^15}|")

def statistics_record(fits, save_path=None):
    if save_path is not None:
        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                f.write('\t'.join(['min', 'max', 'mean', 'var', 'std']) + '\n')

        with open(save_path, "+a") as f:
            f.write('\t'.join([float(HyperGP.tensor.min(fits)), float(HyperGP.tensor.max(fits)), float(HyperGP.tensor.mean(fits)), float(HyperGP.tensor.var(fits)), float(HyperGP.tensor.std(fits))]) + '\n')
    
    return f"|{statistics_record.iter:^6}||{'{:.4e}'.format(float(HyperGP.tensor.min(fits))):^15}|{'{:.4e}'.format(float(HyperGP.tensor.max(fits))):^15}|{'{:.4e}'.format(float(HyperGP.tensor.mean(fits))):^15}|{'{:.4e}'.format(float(HyperGP.tensor.var(fits))):^15}|{'{:.4e}'.format(float(HyperGP.tensor.std(fits))):^15}|" 

statistics_record.init = statistics_record_initprint
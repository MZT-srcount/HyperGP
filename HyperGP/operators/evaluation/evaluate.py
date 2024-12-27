import numpy as np

class EvaluateMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")

class RMSE(EvaluateMethod):
    def __call__(self, label: np.array, output: np.array):
        return np.sqrt(np.dot(label - output, label - output)) / len(label)

'''In GPU method, we use the proposed GPU-CPU mapping func?'''

class R2(EvaluateMethod):
    def __call__(self, label: np.array, output: np.array):
        avg_label = np.mean(label)
        avg_output = np.mean(output)
        return 1 - np.dot(output - avg_output, output - avg_output) / np.dot(label - avg_label, label - avg_label)
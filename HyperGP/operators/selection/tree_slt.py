import random
import numpy as np

class SltMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")

class TourWithRep(SltMethod):
    def __call__(self, fits, rd_state=random):
        return np.argmin(fits)

class TourNoRep(SltMethod):
    def __call__(self, fits, slt_num, cdd_size=2, rd_state=random):
        fit_list = list(range(len(fits)))
        winner = []
        for i in range(slt_num):
            cdds_idx = random.sample(fit_list, cdd_size)
            cdds_fit = list(map(lambda idx: fits[idx], cdds_idx))
            winner.append(cdds_idx[np.argmin(cdds_fit)])
        return winner





if __name__ == '__main__':
    fit1 = [1, 2, 3]
    t = TourNoRep()
    winner = t(fit1, 2)
    print(winner)

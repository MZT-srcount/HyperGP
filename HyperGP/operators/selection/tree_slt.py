import random
import numpy as np
from ... import tensor

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


def tournament(p1, p2, f1, f2, tour_size=3, len_limit=100, best_keep=True):
    p_list, f_list = p1 + p2,  tensor.concatenate((f1, f2))
    legal_list = [z for z, prog in enumerate(p_list) if len(prog) < len_limit]
    if best_keep:
        sample_list = np.array(legal_list)[np.random.randint(0, len(legal_list), size=(len(p1) - 1, tour_size))]
        # sample_list = np.array([(random.sample(legal_list, tour_size)) for i in range(len(p1) - 1)]).flatten()
        tour_list = tensor.argmin(f_list[sample_list.tolist()], dim=1)
        tour_list = [legal_list[int(tensor.argmin(f_list[legal_list]))]] + np.take_along_axis(sample_list, tour_list.numpy().reshape((-1, 1)), axis=1).reshape(-1).tolist()
        
    else:
        sample_list = np.array(legal_list)[np.random.randint(0, len(legal_list), size=(len(p1), tour_size))].tolist()
        tour_list = np.take_along_axis(sample_list, tensor.argmin(f_list[sample_list], dim=1).numpy().reshape((-1, 1)), axis=1).reshape(-1).tolist()
    p_new, f_new = [p_list[sample] for sample in tour_list], f_list[tour_list]
    # print('------------', np.mean([len(ind) for ind in p_new]), np.mean([len(p_list[idx]) for idx in legal_list]))
    return [ind.copy() for ind in p_new], p_new, f_new, f_new.copy()



if __name__ == '__main__':
    fit1 = [1, 2, 3]
    t = TourNoRep()
    winner = t(fit1, 2)
    print(winner)

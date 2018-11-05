import torch
import numpy as np


def cosine_similiarity(query,features):
    norm_q=torch.div(query,torch.norm(query,2,1,True))
    norm_f = torch.div(features, torch.norm(features, 2, 1, True))

    cos = torch.t(torch.mm(norm_f, torch.t(norm_q)))
    score,idx=torch.sort(cos)

    return score,idx

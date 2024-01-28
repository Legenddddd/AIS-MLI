import math

import torch



def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


class Angular_Isotonic_Loss(torch.nn.Module):
    def __init__(self, n_way, lamda=32, mrg=0.1, threshold=0.7):
        torch.nn.Module.__init__(self)

        self.n_way = n_way
        self.lamda = lamda
        self.threshold = threshold

        self.mrg = mrg
        self.cos_m = math.cos(mrg)
        self.sin_m = math.sin(mrg)


    def forward(self, cos, T):
        P_one_hot = binarize(T=T, nb_classes=self.n_way)
        N_one_hot = 1 - P_one_hot

        sin = torch.sqrt((1.0-torch.pow(cos,2)).clamp(0,1))
        pos_phi = cos * self.cos_m - sin * self.sin_m
        pos_phi = torch.where(cos>self.threshold,pos_phi,cos)
        neg_phi = cos * self.cos_m + sin * self.sin_m
        neg_phi = torch.where(cos<self.threshold,neg_phi,cos)

        pos_exp = torch.exp(-self.lamda * (pos_phi - self.threshold ))
        neg_exp = torch.exp(self.lamda * (neg_phi - self.threshold ))


        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=1)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=1)

        pos_term = torch.log(1 + P_sim_sum)
        neg_term = torch.log(1 + N_sim_sum)
        loss = torch.mean(pos_term + neg_term)

        return loss


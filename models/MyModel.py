import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.module.sms3f import SMS3F
from models.module.jpesm import JPESM


class FMSFSDI(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False, ffc=True,enable_lfu=True):

        super(FMSFSDI, self).__init__()

        if resnet:
            l2_channel = 160
            l3_channel = 320
            last_channel = 640
        else:
            l2_channel = 64
            l3_channel = 64
            last_channel = 64

        self.sms3f_extractor = SMS3F(resnet,ffc,enable_lfu)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        self.l2_scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.l3_scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.l4_scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.l2_resolution = 21*21
        self.l3_resolution = 10*10
        self.last_resolution = 5*5

        self.l2_c = l2_channel
        self.l3_c = l3_channel
        self.last_c = last_channel

        self.l2r = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.l3r = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.l4r = nn.Parameter(torch.zeros(2), requires_grad=True)

        self.eita_m1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.eita_m2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.eita_h = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.jpesm = JPESM(resnet)

    def get_feature_map(self, inp):

        batch_size = inp.size(0)
        l2_xl_xg,l3_xl_xg,l4 = self.sms3f_extractor(inp)
        if self.resnet:
            l2_xl_xg = l2_xl_xg/np.sqrt(160)
            l3_xl_xg = l3_xl_xg/np.sqrt(320)
        l2 = l2_xl_xg.view(batch_size, self.l2_c, -1).permute(0, 2, 1).contiguous()
        l3 = l3_xl_xg.view(batch_size, self.l3_c, -1).permute(0, 2, 1).contiguous()

        return l2,l3,l4

    def get_recon_dist(self, query, support, alpha, beta, eita, Woodbury=True, useQ=False):

        reg = support.size(1) / support.size(2)
        lam = reg * alpha.exp() + 1e-6
        rho = beta.exp()
        st = support.permute(0, 2, 1)
        if Woodbury:
            sts = st.matmul(support)
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()
            hat = m_inv.matmul(sts)
        else:
            sst = support.matmul(st)
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(
                lam)).inverse()
            hat = st.matmul(m_inv).matmul(support)
        Q_bar = query.matmul(hat).mul(rho)
        if useQ:
            Q_bar = Q_bar + eita * query
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)

        return dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False):

        l2_rs = self.l2_resolution
        l3_rs = self.l3_resolution
        l4_rs = self.last_resolution
        l2c = self.l2_c
        l3c = self.l3_c
        l4c = self.last_c
        l2_alpha = self.l2r[0]
        l2_beta = self.l2r[1]
        l3_alpha = self.l3r[0]
        l3_beta = self.l3r[1]
        l4_alpha = self.l4r[0]
        l4_beta = self.l4r[1]

        l2,l3,l4 = self.get_feature_map(inp)
        l2_support = l2[:way * shot].view(way, shot * l2_rs, l2c)
        l2_query = l2[way * shot:].view(way * query_shot * l2_rs, l2c)

        l3_support = l3[:way * shot].view(way, shot * l3_rs, l3c)
        l3_query = l3[way * shot:].view(way * query_shot * l3_rs, l3c)

        l4_support = l4[:way * shot]
        l4_query = l4[way * shot:]
        # apply JPESM to last layer
        l4_support = self.jpesm(l4_support)
        if self.resnet:
            l4_query = l4_query / np.sqrt(640)
            l4_support = l4_support / np.sqrt(640)

        l4_query = l4_query.view(way * query_shot, l4c, -1).permute(0, 2, 1).contiguous()
        l4_support = l4_support.view(way * shot, l4c, -1).permute(0, 2, 1).contiguous()
        l4_support = l4_support.view(way, shot * l4_rs, l4c)
        l4_query = l4_query.view(way * query_shot * l4_rs, l4c)

        l2_recon_dist = self.get_recon_dist(query=l2_query, support=l2_support, alpha=l2_alpha, beta=l2_beta, eita = self.eita_m1, useQ=True)
        l2_neg_dist = l2_recon_dist.neg().view(way * query_shot, l2_rs, way).mean(1)

        l3_recon_dist = self.get_recon_dist(query=l3_query, support=l3_support, alpha=l3_alpha, beta=l3_beta, eita = self.eita_m2, useQ=True)
        l3_neg_dist = l3_recon_dist.neg().view(way * query_shot, l3_rs, way).mean(1)

        l4_recon_dist = self.get_recon_dist(query=l4_query, support=l4_support, alpha=l4_alpha,beta=l4_beta, eita = self.eita_h, useQ=True)
        l4_neg_dist = l4_recon_dist.neg().view(way * query_shot, l4_rs, way).mean(1)

        if return_support:
            return l2_neg_dist, l3_neg_dist, l4_neg_dist, l2_support, l3_support, l4_support
        else:
            return l2_neg_dist, l3_neg_dist, l4_neg_dist

    def meta_test(self, inp, way, shot, query_shot):

        l2_neg_dist, l3_neg_dist, l4_neg_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)
        neg_dist = l2_neg_dist * self.l2_scale + l3_neg_dist * self.l3_scale + l4_neg_dist * self.l4_scale
        _, max_index = torch.max(neg_dist, 1)

        return max_index



    def forward(self, inp):

        l2_neg_dist, l3_neg_dist, l4_neg_dist, l2_support, l3_support, l4_support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)

        l2_logits = l2_neg_dist * self.l2_scale
        l3_logits = l3_neg_dist * self.l3_scale
        l4_logits = l4_neg_dist * self.l4_scale
        l2_log_prediction = F.log_softmax(l2_logits, dim=1)
        l3_log_prediction = F.log_softmax(l3_logits, dim=1)
        l4_log_prediction = F.log_softmax(l4_logits, dim=1)

        return l2_log_prediction, l3_log_prediction, l4_log_prediction, l2_support, l3_support, l4_support

import torch.nn as nn
import torch
import torch.nn.functional as F


class Criterion(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(Criterion, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # N x 1 distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # postive sample label = 0 distance descend
        # negative sample label = 1
        # negative sample distance has lower bound self.margin
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # cosine_similarity = torch.cosine_similarity(output1, output2, dim=1)
        # loss = torch.mean((cosine_similarity - label) ** 2)
        return loss




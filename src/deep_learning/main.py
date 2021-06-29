from src.deep_learning.data import DataSampler
from src.deep_learning.model import SiameseNetwork
from src.deep_learning.criterion import Criterion
import torch

from src.deep_learning.train import Train

pkl_path = "../../data/files_dump.pkl"
data_sampler = DataSampler(train_num=4000, batch_size=32, pkl_path=pkl_path)
siamese_nn = SiameseNetwork().cuda()
criterion = Criterion()
train = Train(data_sampler=data_sampler, model=siamese_nn, criterion=criterion)
train.train(iterations=50000, lr=3e-4)


with open("model.pth", "wb+") as f:
    torch.save(siamese_nn, f)




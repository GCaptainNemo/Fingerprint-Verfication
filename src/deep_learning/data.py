import torch.utils.data.dataset as dataset
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

# training_sample_count = 4000
# validation_sample_count = 1000
# test_sample_count = 1000


class DataSampler:
    def __init__(self, train_num, batch_size, pkl_path):
        self.train_num = train_num
        self.load_dataset(pkl_path)
        self.batch_size = batch_size

    def load_dataset(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.files_lst = pickle.load(f)

    def sample(self):
        """
        batch_size: positive and negative
        """
        # ############################################################################
        # sample postive samples
        # ############################################################################
        # 对正样本(同一人的同一手指)，标签为0
        pos_label = torch.zeros([self.batch_size, 1], dtype=torch.float32)

        pos_index = np.random.randint(0, self.train_num, [self.batch_size, 1])
        each_index = np.random.randint(0, len(self.files_lst[pos_index[0, 0]]), [1, 2])

        address_1 = self.files_lst[pos_index[0, 0]][each_index[0, 0]]
        # print(address_1)
        siamese_1_img_batch = torch.unsqueeze(transforms.ToTensor()(
            cv2.imread(address_1, cv2.IMREAD_GRAYSCALE))
            , dim=0)

        address_2 = self.files_lst[pos_index[0, 0]][each_index[0, 1]]
        siamese_2_img_batch = torch.unsqueeze(transforms.ToTensor()(
            cv2.imread(address_2, cv2.IMREAD_GRAYSCALE))
            , dim=0)

        for i in range(1, self.batch_size):
            each_index = np.random.randint(0, len(self.files_lst[pos_index[i, 0]]), [1, 2])
            address_1 = self.files_lst[pos_index[i, 0]][each_index[0, 0]]
            siamese_1_pos = torch.unsqueeze(transforms.ToTensor()(
                cv2.imread(address_1, cv2.IMREAD_GRAYSCALE))
                , dim=0)
            # ToTensor: normalized to [0, 1] division 255
            address_2 = self.files_lst[pos_index[i, 0]][each_index[0, 1]]
            siamese_2_pos = torch.unsqueeze(transforms.ToTensor()(
                cv2.imread(address_2, cv2.IMREAD_GRAYSCALE))
                , dim=0)
            # print("address_1 = ", address_1)
            # print("address_2 = ", address_2)

            # print(siamese_1_pos.shape)
            # print(siamese_2_pos.shape)

            siamese_1_img_batch = torch.cat([siamese_1_img_batch, siamese_1_pos], dim=0)
            siamese_2_img_batch = torch.cat([siamese_2_img_batch, siamese_2_pos], dim=0)

        # ############################################################################
        # sample negative samples
        # ############################################################################
        # 对负样本，标签为1
        neg_label = torch.ones([self.batch_size, 1], dtype=torch.float32)
        label_tensor = torch.cat([pos_label, neg_label], dim=0)
        for i in range(self.batch_size):
            while True:
                neg_index = np.random.randint(0, self.train_num, [1, 2])
                if neg_index[0, 0] != neg_index[0, 1]:
                    break
            index_1 = np.random.randint(0, len(self.files_lst[neg_index[0, 0]]), [1])
            address_1 = self.files_lst[neg_index[0, 0]][index_1[0]]
            siamese_1_neg = torch.unsqueeze(transforms.ToTensor()(
                cv2.imread(address_1, cv2.IMREAD_GRAYSCALE))
                , dim=0)
            index_2 = np.random.randint(0, len(self.files_lst[neg_index[0, 1]]), [1])

            address_2 = self.files_lst[neg_index[0, 1]][index_2[0]]
            siamese_2_neg = torch.unsqueeze(transforms.ToTensor()(
                cv2.imread(address_2, cv2.IMREAD_GRAYSCALE))
                , dim=0)
            # print("address_1 = ", address_1)
            # print("address_2 = ", address_2)

            # print(siamese_1_neg.shape)
            # print(siamese_2_neg.shape)

            siamese_1_img_batch = torch.cat([siamese_1_img_batch, siamese_1_neg], dim=0)
            siamese_2_img_batch = torch.cat([siamese_2_img_batch, siamese_2_neg], dim=0)

        return siamese_1_img_batch, siamese_2_img_batch, label_tensor








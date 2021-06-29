import os
import pickle
import cv2
import torch
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

hand_encode_dict = {"Left": 0, "Right": 1}
finger_encode_dict = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}


def dump_files(input_path, output_path):
    """
    output a 6000 x d (600 people x 10 fingers) dimension list
    6000 x d, each dimension is positive sample, otherwise it's negative
    """
    six_thousand_lst = [[] for _ in range(6000)]
    for root, dirs, files in os.walk(input_path):
        for file in files:
            lst = file.split("_")
            index = (int(lst[0]) - 1) * 10 + hand_encode_dict[lst[3]] * 5 + finger_encode_dict[lst[4]]
            address = root + "/" + file
            print("address = ", address)
            six_thousand_lst[index].append(address)

    with open(output_path + "/files_dump.pkl", "wb") as f:
        pickle.dump(six_thousand_lst, f)


def load_pkl(pkl_dir):
    with open(pkl_dir + "/files_dump.pkl", "rb") as f:
        files_name_lst = pickle.load(f)
    print(files_name_lst[0])
    print(max(len(files_name) for files_name in files_name_lst))
    print(min(len(files_name) for files_name in files_name_lst))

    img = cv2.imread(files_name_lst[0][0], cv2.IMREAD_GRAYSCALE)
    print(transforms.ToTensor()(img).dtype)
    print(type(img))
    print(img.shape)
    cv2.imshow("test", img)
    cv2.waitKey(0)


def construct_pos_neg_samples(train_num):
    with open("../../data/files_dump.pkl", "rb") as f:
        files_name_lst = pickle.load(f)
    with open("../pos_samples.txt", "w+") as txt:
        # pos samples
        pos_num = 0
        for i in range(train_num):
            for j in range(len(files_name_lst[i])):
                # self-self is positive samples
                for k in range(j, len(files_name_lst[i])):
                    pos_string = files_name_lst[i][j] + "    " + files_name_lst[i][k] + "    1\n"
                    txt.writelines(pos_string)
                    pos_num += 1
        print("there are ", pos_num, "positive samples")

        # neg samples
        neg_num = 0
        for i in range(train_num):
            for neg_i in range(len(files_name_lst[i])):
                for j in range(i + 1, train_num):
                    for neg_j in range(len(files_name_lst[j])):
                        neg_num += 1
                        neg_string = files_name_lst[i][neg_i] + "    " + files_name_lst[j][neg_j] + "    0\n"
                        txt.writelines(neg_string)
        print("there are ", neg_num, "negative samples")


def construct_pos_neg_samples_test():
    with open("../../data/files_dump.pkl", "rb") as f:
        files_name_lst = pickle.load(f)
    files_name_lst_test = files_name_lst[5000:6000]

    with open("../pos_samples.txt", "w+") as txt:
        # pos samples
        pos_num = 0
        total_num = 0
        for i in range(len(files_name_lst_test)):
            total_num += len(files_name_lst_test[i])
            for j in range(len(files_name_lst_test[i])):
                # self-self is positive samples
                for k in range(j, len(files_name_lst_test[i])):
                    pos_string = files_name_lst_test[i][j] + "    " + files_name_lst_test[i][k] + "    0\n"
                    txt.writelines(pos_string)
                    pos_num += 1
        print("total num = ", total_num)
        print("there are ", pos_num, "positive samples")

        # neg samples
        neg_num = 0
        for i in range(1000):
            for neg_i in range(len(files_name_lst_test[i])):
                for j in range(i + 1, 1000):
                    for neg_j in range(len(files_name_lst_test[j])):
                        neg_num += 1
                        neg_string = files_name_lst_test[i][neg_i] + "    " + files_name_lst_test[j][neg_j] + "    1\n"
                        txt.writelines(neg_string)
            if neg_num > 10000:
                break
        print("there are ", neg_num, "negative samples")


def pre_process(file_path):
    # Load the image grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Get rid of the excess pixels
    # img = img[2:-4, 2:-4]
    # all the images are of the same size (96 * 96)
    img = cv2.resize(img, (96, 96))
    return img


def traversal_total_dir(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for name in files:
            address = os.path.join(root, name)
            print("address = ", address)
            out_img = pre_process(address)
            cv2.imwrite(output_path + name, out_img)


def plot_precision_recall_curve(address):
    with open(address, "rb+") as f:
        dis_label_lst = pickle.load(f)
    print(len(dis_label_lst))
    dis_total = dis_label_lst[0][0]
    label_total = dis_label_lst[0][0]

    for i, dis_label in enumerate(dis_label_lst):
        if i == 0:
            continue
        dis = dis_label[0]
        label = dis_label[1]
        dis_total = torch.cat([dis_total, dis], dim=0)
        label_total = torch.cat([label_total, label], dim=0)
    label_total = label_total.numpy().astype(int)
    dis_total = dis_total.numpy()
    max_dis = np.max(dis_total)
    # dis_total = 1 - dis_total / max_dis
    precision, recall, thresh = precision_recall_curve(label_total, dis_total, pos_label=1)
    print(precision)
    print(recall)
    print(thresh)
    # print((1 - thresh) * max_dis)
    plt.figure(1)  # 创建图表1
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.show()


if __name__ == "__main__":
    # dump_files("../../data/process", "../../data/")
    # load_pkl("../../data")
    # construct_pos_neg_samples_test()
    plot_precision_recall_curve("dis_label.pkl")
from sklearn.metrics import precision_recall_curve
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_precision_recall_curve(dst_lst_address):
    with open(dst_lst_address, "rb+") as f:
        dis_total = pickle.load(f)
    print(len(dis_total))
    with open("pos_samples.txt", "r") as f:
        sample_lst = f.readlines()
        label_total = []
        for sample in sample_lst:
            label_total.append(int(sample.strip().split("    ")[-1]))

    label_total = np.array(label_total)
    dis_total = -np.array(dis_total)  # 越大越为正
    precision, recall, thresh = precision_recall_curve(label_total, dis_total, pos_label=0)
    print("precision = ", precision)
    print("recall = ", recall)
    print("thresh = ", thresh)
    for i, value in enumerate(recall):
        if recall[i] > 0.85:
            print(thresh[i])
            break
    # print((1 - thresh) * max_dis)
    plt.figure(1)  # 创建图表1
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.show()


def cal_confusion_matrix(predicted_label_address):
    with open("pos_samples.txt", "r") as f:
        sample_lst = f.readlines()
        label_total = []
        for sample in sample_lst:
            label_total.append(int(sample.strip().split("    ")[-1]))
    actual_label = np.array(label_total)
    conf_matrix = np.zeros([2, 2])
    with open(predicted_label_address, "rb+") as f:
        predicted_label_address = np.array(pickle.load(f))
    for i in range(len(predicted_label_address)):
        conf_matrix[int(actual_label[i]), int(predicted_label_address[i])] += 1
    plot_confusion_matrix(conf_matrix, ["positive", "negative"])




def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : confusion matrix
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:percentage, False:Num
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.show()


if __name__ == "__main__":
    plot_precision_recall_curve("dist_lst.pkl")
    # cal_confusion_matrix("predicted.pkl")
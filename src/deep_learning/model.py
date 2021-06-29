import torch.nn as nn

# 96 x 96


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),  # inplace=True 直接在原地址上修改变量
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 4 x 48 x 48

            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 8 x 24 x 24

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 16 x 12 x 12

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 32 x 6 x 6
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 6 * 6, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 30)
        )


    def forward_once(self, x):
        output = self.cnn1(x)
        # print("output.shape = ", output.shape)
        # reshape N x d feature
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2









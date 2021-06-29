import torch.optim as optim


class Train:
    def __init__(self, data_sampler, model, criterion):
        self.data_sampler = data_sampler
        self.criterion = criterion
        self.model = model

    def train(self, iterations, lr):
        optimizer = optim.Adam(self.model.parameters(), lr)
        avg_loss = 0
        for e in range(iterations):
            siamese_1, siamese_2, label = self.data_sampler.sample()
            siamese_1, siamese_2, label = siamese_1.cuda(), siamese_2.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = self.model(siamese_1, siamese_2)
            loss = self.criterion(output1, output2, label)

            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 50 == 49:
                loss = avg_loss / 50
                print("Step {} - lr {} -  loss: {}".format(e, lr, loss))
                avg_loss = 0

                # error = self.siamese_nn.loss_func(2 ** 8)
                # self.siamese_nn.append(error.detach())





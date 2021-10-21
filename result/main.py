
import matplotlib.pyplot as plt

with open("./result.txt", "r+") as f:
    lines_lst = f.readlines()
loss_lst = []
iterations = []
i = 0
for line in lines_lst:
    str_ = line.strip()
    loss = float(str_.split("loss:")[-1])
    loss_lst.append(loss)
    iterations.append(i * 50)
    i+= 1
 
plt.figure(1)
plt.plot(iterations, loss_lst)
plt.title("Loss-Iteration Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()


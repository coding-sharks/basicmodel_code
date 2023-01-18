import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0], [0], [1]])
y_data = y_data.float()

#注意BCELoss要求input参数类型一样,最好为float!需进行数据类型转换
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    y_pred = model(x_data)
    y_pred = y_pred.float()

    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#预测:x = 2.5
x_test = torch.tensor([2.5])
x_test = x_test.float()
y_test = model(x_test)
print("y_test = ", y_test.data)

#画图准备
x = np.linspace(0, 10, 200)
x_t = torch.tensor(x).view(200, 1)
x_t = x_t.float()
y_t = model(x_t)
y = y_t.data.numpy()

#可视化
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('probability')
plt.grid()
plt.show()

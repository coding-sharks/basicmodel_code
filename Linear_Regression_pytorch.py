import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class Linearmoudle(torch.nn.Module):
    def __init__(self):
        super(Linearmoudle, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = Linearmoudle()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(3400):
    y_pred = model.linear(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

x_test = torch.tensor([4.0])
y_test = model.linear(x_test)

print("y_test = ", y_test.data)

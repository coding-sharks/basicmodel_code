import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad = True

a = 0.05


def forward(x):
    return w * x


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict(before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\tgrad:", x, y, w.grad.item())
        w.data = w.data - w.grad.item() * a
        w.grad.data.zero_()
    if l.item() == 0.0:
        break
    print("epoch:", epoch, l.item())


print("predict(after training)", 4, forward(4).item())

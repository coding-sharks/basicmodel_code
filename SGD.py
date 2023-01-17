x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = 0.5
a = 0.05

def forward(x):
    return w * x

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (w * x - y)

print("predict(before training) 4:", forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - a * gradient(x, y)
        print("\tgrad ",x,y,grad)
        l = loss(x, y)
    if l == 0:
        break
    print("epoch:", epoch, "w = ", w, "loss = ", l)

print("epoch:", epoch, "w = ", w, "loss = ", l)
print("predict(after training) 4:", forward(4))

x_data = [1,2,3]
y_data = [2,4,6]

w = 9999
a = 0.05

def forward(x):
    return w * x

def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x *(w * x - y)
    return grad / len(xs)

print("predict(before training) 4:",forward(4))

for epoch in range(100):
    cost_val = cost(x_data,y_data)
    grad_val = gradient(x_data,y_data)
    w = w - a * gradient(x_data,y_data)
    print("epoch:",epoch,"w = ",w,"grad = ",grad_val)
    if grad_val == 0:
        break

print("predict(after training) 4:",forward(4))





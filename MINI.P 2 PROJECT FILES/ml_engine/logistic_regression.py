import math

class LogisticRegression:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.w = 0
        self.b = 0
        self.history = []
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    def predict(self, x):
        predicted = []
        for val in x:
            z = self.w*val + self.b
            prediction = self.sigmoid(z)
            predicted.append(prediction)
        return predicted 
    def compute_loss(self, y, predicted):
        total_loss = 0
        for i in range(len(y)):
            loss = -(y[i] * math.log(predicted[i]) + (1-y[i]) * math.log(1 - predicted[i]))
            total_loss += loss
        return total_loss/ len(y)
    def compute_gradient(self, x, y, predicted):
        grad_w = 0
        grad_b = 0
        for i in range(len(x)):
            grad_w += x[i] * (predicted[i] - y[i])
            grad_b += (predicted[i] - y[i])
        grad_w = grad_w / len(x)
        grad_b = grad_b / len(x)
        return grad_w, grad_b
    def parameter_update(self, grad_w, grad_b):
        self.w = self.w - self.lr*grad_w
        self.b = self.b - self.lr*grad_b
    def fit(self, x, y):
        for epoch in range(self.epochs):
            predicted = self.predict(x)
            loss = self.compute_loss(y, predicted)
            grad_w, grad_b = self.compute_gradient(x, y, predicted)
            self.parameter_update(grad_w, grad_b)
            self.history.append({
                "epoch": epoch + 1,
                "loss": loss,
                "weight": self.w,
                "bias": self.b
            })
            print("predicted: ",predicted)
            print("loss: ", loss)
            print("w: ", grad_w)
            print("b: ", grad_b)

x = [1,2,3]
y = [0,1,1]
model = LogisticRegression(0.01, 40000)
result = model.fit(x, y)
print(result)


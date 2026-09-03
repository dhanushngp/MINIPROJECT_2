class MultipleLinearRegression:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = []
        self.b = 0
        self.history = []

    def predict(self, x):
        predicted = []
        for row in x:
            prediction = 0
            for i in range(len(row)):
                prediction += self.w[i] * row[i]
            prediction += self.b
            predicted.append(prediction)
        return predicted
    
    def compute_loss(self, y, predicted):
        total_loss = 0
        for i in range(len(y)):
            loss = (y[i] - predicted[i]) ** 2
            total_loss += loss
        return total_loss / len(y)
    
    def compute_gradient(self, x, y, predicted):
        grad_w = [0] * len(self.w)
        grad_b = 0
        for i in range(len(x)):
            for j in range(len(self.w)):
                grad_w[j] += 2 * x[i][j] * (predicted[i] - y[i])
            grad_b += 2 * (predicted[i] - y[i])
        for j in range(len(grad_w)):
            grad_w[j] = grad_w[j] / len(x)
        grad_b = grad_b / len(x)
        return grad_w, grad_b
    
    def parameter(self, grad_w, grad_b):
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.learning_rate * grad_w[i]
        self.b = self.b - self.learning_rate * grad_b

    def fit(self, x, y):
        self.w = [0] * len(x[0])
        self.b = 0
        for epoch in range(self.epochs):
            predicted = self.predict(x)
            loss = self.compute_loss(y, predicted)
            grad_w, grad_b = self.compute_gradient(
                x, y, predicted
            )
            self.parameter(grad_w, grad_b)
            print(f"{epoch + 1}")
            print(f"Loss: {loss}")
            print(f"W : {self.w}")
            print(f"B : {self.b}")
            self.history.append({
                "epoch": epoch + 1,
                "loss": loss,
                "weights": self.w.copy(),
                "bias": self.b
            })

x = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
]

y = [5, 8, 11, 14, 17]
model = MultipleLinearRegression(0.01, 40000)
model.fit(x, y)
new_data = [
    [6, 7],
    [7, 8]
]
prediction = model.predict(new_data)
print("Final Prediction:", prediction)
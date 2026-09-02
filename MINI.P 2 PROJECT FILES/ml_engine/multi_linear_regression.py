class LinearRegression:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0
        self.b = 0
        self.history = []

    def predict(self, x):
        predicted = []

        for val in x:
            prediction = self.w * val + self.b
            predicted.append(prediction)

        return predicted

    def compute_loss(self, y, predicted):
        total_loss = 0

        for i in range(len(y)):
            loss = (y[i] - predicted[i]) ** 2
            total_loss += loss

        return total_loss / len(y)

    def compute_gradient(self, x, y, predicted):
        grad_w = 0
        grad_b = 0

        for i in range(len(x)):
            grad_w += 2 * x[i] * (predicted[i] - y[i])
            grad_b += 2 * (predicted[i] - y[i])

        grad_w = grad_w / len(x)
        grad_b = grad_b / len(x)

        return grad_w, grad_b

    def parameter(self, grad_w, grad_b):
        self.w = self.w - self.learning_rate * grad_w
        self.b = self.b - self.learning_rate * grad_b

    def fit(self, x, y):
        self.history = []

        for epoch in range(self.epochs):
            predicted = self.predict(x)
            loss = self.compute_loss(y, predicted)
            grad_w, grad_b = self.compute_gradient(x, y, predicted)

            self.parameter(grad_w, grad_b)

            new_predicted = self.predict(x)
            new_loss = self.compute_loss(y, new_predicted)

            self.history.append({
                "epoch": epoch + 1,
                "prediction": new_predicted,
                "loss": new_loss,
                "gradient_w": grad_w,
                "gradient_b": grad_b,
                "w": self.w,
                "b": self.b
            })

            print(epoch + 1, "Loss:", new_loss, "W:", self.w, "B:", self.b)


x = [1, 2, 3]
y = [7, 12, 6]

model = LinearRegression(0.001, 4000000)

model.fit(x, y)

print("Final W:", model.w)
print("Final B:", model.b)

new_x = [4, 5, 6]

prediction = model.predict(new_x)

print("New Predictions:", prediction)
class LinearRegression:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0
        self.b = 0  
        self.history = []
    def predict(self, x):
        predicted = self.w * x + self.b
        return predicted
    def compute_loss(self, y, predicted):
        loss = (y - predicted) ** 2
        return loss
    def compute_gradient(self, x, y, predicted):
        grad_w = 2 * x * (predicted - y)
        grad_b = 2 * (predicted - y)
        return grad_w, grad_b
    def parameter(self, grad_w, grad_b):
        self.w = self.w - self.learning_rate * grad_w
        self.b = self.b - self.learning_rate * grad_b
    def fit(self, x, y):
        for epoch in range(self.epochs):
            predicted = self.predict(x)
            loss = self.compute_loss(y, predicted)
            grad_w, grad_b = self.compute_gradient( x, y, predicted )
            self.parameter(grad_w, grad_b)
            print("Epoch:", epoch + 1)
            print("Prediction:", predicted)
            print("Loss:", loss)
            self.history.append({
                "epoch" : epoch +1,
                "prediction" : predicted,
                "loss" : loss,
                "gradient_w" : grad_w,
                "gradient_b" : grad_b,
                "w" : self.w,
                "b" : self.b
            })

model = LinearRegression(0.01, 100)

x = 16
y = 24

model.fit(x, y)

print("Final W:", model.w)
print("Final B:", model.b)

print(model.history[0])
print(model.history[-1])

print(len(model.history))

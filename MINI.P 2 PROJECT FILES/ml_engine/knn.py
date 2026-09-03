class KNN:
    def __init__(self, k):
        self.k = k
    def distance(self, x1, x2):
        return abs(x1 - x2)
    def fit(self, x, y):
        self.x = x
        self.y = y
    def get_neighbours(self, x):
        distances = []
        for i in range(len(self.x)):
            distance = self.distance(x, self.x[i])
            distances.append((distance, self.y[i])) 
        distances.sort()
        return distances[:self.k]
    def majority_vote(self, neighbors):
        votes = []
        for distance, label in neighbors:
            votes.append(label)
        return max(set(votes), key=votes.count)
    def predict(self, x):
        neighbors = self.get_neighbours(x)
        prediction = self.majority_vote(neighbors)      
        return prediction

x = [1, 2, 3, 4, 5]
y = [0, 0, 1, 1, 1]

model = KNN(3)
model.fit(x, y)
prediction = model.predict(3.5)
print("Prediction:", prediction)
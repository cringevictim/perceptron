import numpy as np

class Perceptron:
    def init(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def train(self, X, Y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                error = Y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# Generate dataset
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
Y = np.array([0, 0, 0, 0, 1, 1, 0, 1])
# Y = np.array([0, 1, 0, 0, 1, 1, 1, 0])  #Wrong dataset for example


# Train perceptron
perceptron = Perceptron()
perceptron.train(X, Y)

# Test perceptron
print(f"Testing:")
testing_completion = True
for i in range(len(X)):
    test_input = X[i]
    predicted_output = Y[i]
    output = perceptron.predict(test_input)
    if output != predicted_output: testing_completion = False
    print(f"Input: {test_input}, Output: {output}, Predicted output: {predicted_output}")
if testing_completion:
    print(f"Test passed")
else:
    print(f"Test failed")
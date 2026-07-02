import numpy as np

class Logistic_Regression():

  def __init__(self, learning_rate, no_of_iterations):
    self.no_of_iterations = no_of_iterations
    self.learning_rate = learning_rate
    # Create an empty list to store the loss of each iteration
    self.loss_history = []

  def fit(self, X, Y):
     self.m, self.n = X.shape

     self.w = np.zeros(self.n)
     self.b = 0
     self.X = X
     self.Y = Y

     for i in range(self.no_of_iterations):
          # 1. Update the weights
          self.update_weights()

          # 2. Calculate and store the current loss
          current_loss = self.calculate_loss()
          self.loss_history.append(current_loss)

          # Optional: Print loss every 100 iterations to monitor progress
          if i % 100 == 0:
              print(f"Iteration {i}: Loss = {current_loss:.4f}")

  def update_weights(self):
       Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

       dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))
       db = (1 / self.m) * np.sum(Y_hat - self.Y)

       self.w = self.w - self.learning_rate * dw
       self.b = self.b - self.learning_rate * db

  def calculate_loss(self):
       # Predict probabilities using current weights
       Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

       # Add a tiny epsilon value to prevent log(0) which causes NaN errors
       epsilon = 1e-15
       Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)

       # Formula for Binary Cross-Entropy Loss
       loss = - (1 / self.m) * np.sum(self.Y * np.log(Y_hat) + (1 - self.Y) * np.log(1 - Y_hat))
       return loss

  def predict(self, X):
       Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
       Y_pred = np.where(Y_pred > 0.5, 1, 0)
       return Y_pred
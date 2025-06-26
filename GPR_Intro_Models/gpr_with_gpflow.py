import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Generate some sample data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
Y = np.sin(X) + 0.5 * np.random.randn(*X.shape)

# Kernel function: Matern 3/2
k = gpflow.kernels.Matern32()

# Mean function: Zero mean
mean_function = None

# Construct Gaussian Process Regression Model
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=mean_function)

# Optimize hyperparameters 
opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, m.trainable_variables)


print(gpflow.utilities.print_summary(m))


# Make predictions
X_test = np.linspace(-2, 20, 200).reshape(-1, 1)
mean, var = m.predict_y(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X, Y, 'kx', alpha=0.6, label='Observations')
plt.plot(X_test, mean, 'b-', label='Predicted mean')

mean_np = mean.numpy()
var_np = var.numpy()

plt.fill_between(X_test.flatten(),
                 (mean_np - 2 * np.sqrt(var_np)).flatten(),
                 (mean_np + 2 * np.sqrt(var_np)).flatten(),
                 alpha=0.2, color='blue', label='95% Confidence Interval')

plt.xlabel('Input X')
plt.ylabel('Output Y')
plt.title('Gaussian Process Regression with Matern 3/2 Kernel with GPflow')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
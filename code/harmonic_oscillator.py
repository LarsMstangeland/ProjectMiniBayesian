import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from Kalmante import *


# Define the state-space model
dt = 0.1  # Time step
omega = 1.0  # Natural frequency
A_euler = np.array([[0, dt], [-omega**2 * dt, 0]])  # State transition matrix using forward Euler
A = scipy.linalg.expm(A_euler)
H = np.array([[1, 0]])  # Measurement matrix
Q = np.eye(2) * 0.005 * dt  # Process noise covariance
R = np.eye(1) * 0.5  # Measurement noise covariance

# Initialize state estimate and covariance
x_est = np.array([[1], [0]])  # Initial position and velocity
P = np.eye(2) * 1.0  # Initial uncertainty

# Simulate noisy measurements
periods = 6
time_steps = round(periods*2*np.pi/dt)
true_states = []
measurements = []
for _ in range(time_steps):
    x_est = A @ x_est + np.random.multivariate_normal(mean=[0, 0], cov=Q).reshape(-1, 1) # Propagate state forward
    measurement = H @ x_est + np.random.normal(0, np.sqrt(R[0, 0]))  # Add measurement noise
    true_states.append(x_est.flatten())
    measurements.append(measurement.flatten())





#lab 4
plt.plot([s[0] for s in true_states], label='True State')
plt.plot([s[0] for s in filtered_states], label='Filtered State')
plt.scatter(range(time_steps), [m[0] for m in measurements], label='Measurements', color='red', s=1)
plt.legend()
plt.show()
# Run Kalman filters
kalman_update()



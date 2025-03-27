import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def s(m):
    return sum(abs(x) for x in m)


# ------------------------------------------------------------
# 1. Generate the "true" system trajectory + measurement noise
# ------------------------------------------------------------

def generate_data(periods=6, dt=0.2, omega=1.0, Q_scale=0.005, R_val=0.2, seed=42):
    """
    Generates:
      - x_true: shape (time_steps, 2), the "true" states of the system
                with process noise added at each step
      - A:      2x2 matrix = expm(A_euler)
      - Q:      2x2 process noise covariance
      - R_val:  scalar measurement noise variance
      - measurement_noises: shape (time_steps,), one noise sample per step

    The idea is to keep the same x_true and measurement_noises for
    both fixed and adaptive filters, so they see the *exact same* random draws.
    """
    np.random.seed(seed)  # reproducible
    
    # 1) Define system matrix A
    A_euler = np.array([[0, dt],
                        [-omega**2 * dt, 0]])
    A = scipy.linalg.expm(A_euler)  # discrete-time transition
    Q = np.eye(2) * Q_scale * dt    # process noise covariance
    
    # 2) Number of time steps
    time_steps = round(periods * 2 * np.pi / dt)
    t_array = np.linspace(0, periods*2*np.pi, time_steps)
    
    # 3) Generate "true" states with process noise
    x_true = np.zeros((time_steps, 2))
    x_true[0] = [1.0, 0.0]  # initial condition
    for k in range(time_steps - 1):
        w_k = np.random.multivariate_normal([0, 0], Q)
        x_true[k+1] = A @ x_true[k] + w_k
    
    # 4) Generate measurement noises (one per step)
    #    We'll reuse these for *both* filters.
    measurement_noises = np.random.normal(0, np.sqrt(R_val), size=time_steps)
    
    return t_array, x_true, A, Q, R_val, measurement_noises, dt



def kalman_predict_lorenz(x_est, P_est, A, Q, dt=0.02):
    J = lorenz_jacobian(x_est)
    A = scipy.linalg.expm(J*dt)

    x_pred = A @ x_est
    P_pred = A @ P_est @ A.T + Q
    return x_pred, P_pred

def kalman_predict(x_est, P_est, A, Q):

    x_pred = A @ x_est
    P_pred = A @ P_est @ A.T + Q
    return x_pred, P_pred

def kalman_update(x_pred, P_pred, y, m, R):
    y_pred = m @ x_pred
    innovation = y - y_pred
    S = m @ P_pred @ m + R  # scalar
    K = (P_pred @ m) / S    # shape (2,)
    x_est = x_pred + K * innovation
    P_est = P_pred - np.outer(K, m) @ P_pred
    return x_est, P_est





def select_measurement_direction_from_M(P, OnlyBasis):
    """
    Given a covariance matrix P and a set of candidate unit vectors M,
    choose the m in M that maximizes m^T P m (i.e. gives the largest expected information gain).
    """

    # Define a set M of candidate unit vectors in R^n.
    # For a 3D system, you might include the basis vectors and some linear combinations.
    # For a 2D system, you might include [1, 0], [0, 1], [1, 1], etc.



    if P.shape[1] == 2:
        M = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ])
        if OnlyBasis:
            M = np.array([
                [1, 0],
                [0, 1],
            ])
    else:

        M = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, 0],
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1],

        ])
        if OnlyBasis:
            M = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ])

    

    # Normalize each candidate in M to make them unit vectors.
    M = np.array([m / np.linalg.norm(m) for m in M])

    best_value = -np.inf
    best_m = None
    for m in M:
        value = m.T @ P @ m
        if value > best_value:
            best_value = value
            best_m = m
    return best_m


# def vector_max_ExpectatioG(P):
#     #Expected information gain for measurement
#     eigenvalue, eigenvector = np.linalg.eig(P)
    
#     max_idx = np.argmax(eigenvalue)
#     m = eigenvector[:, max_idx]

#     #normalize with l1
#     m = m/s(m)
#     return m

# ------------------------------------------------------------
# 2. Run the Kalman filter
# ------------------------------------------------------------

def run_kalman_max_m(x_true, A, Q, R_val, measurement_noises):
    """
    Uses m = L1 Max direction at each step, reusing the same measurement_noises
    so that measurement_noises[k] is the random part of y_k.
    """
    time_steps = x_true.shape[0]
    x_est = np.array([0.0, 0.0])
    P_est = np.eye(2) * 1.0
    
    x_est_array = np.zeros((time_steps, 2))
    measurement_vectors = np.zeros((time_steps, 2))

    information_gain = np.zeros(time_steps)
    EntropyRate = np.zeros(time_steps)
    
    for k in range(time_steps):
        # Predict
        x_pred, P_pred = kalman_predict(x_est, P_est, A, Q)
        
        # True state at step k
        xk_true = x_true[k]
        
        # Generate measurement: same noise draw, but direction = [1,0]
        m = select_measurement_direction_from_M(P_pred, OnlyBasis = False)
        y = m @ xk_true + measurement_noises[k]


        
        # Update
        x_est, P_est = kalman_update(x_pred, P_pred, y, m, R_val)
        
        x_est_array[k] = x_est
        measurement_vectors[k, :] = m
    
    return x_est_array, measurement_vectors



# ------------------------------------------------------------
# Lorentz case
# ------------------------------------------------------------

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8/3

R = np.eye(2) * 0.5  # Measurement noise covariance

# Define the Lorenz system

def lorenz_system(t, state):
    del t
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(t_end = 20, dt = 0.02, x0=[1,1,1], R_val=0.2, seed=42):
    """
    Generates a Lorenz trajectory by numerically integrating the ODE,
    then adds measurement noise for observations of (x, y).

    :param sigma, rho, beta: Lorenz system parameters.
    :param dt: time step for the solver output.
    :param t_end: end time.
    :param x0: initial condition [x, y, z].
    :param R_val: measurement noise variance for each of x, y.
    :param seed: random seed for reproducibility.

    :return:
      time_span: array of time points of length N.
      true_states: shape (N, 3), the solution of the Lorenz system [x, y, z].
      measurements: shape (N, 2), noisy measurements of [x, y].
    """



    np.random.seed(seed)

    # Build the time array and solve the ODE
    N = round(t_end / dt)
    time_span = np.linspace(0, t_end, N)
    sol = solve_ivp(lorenz_system, [time_span[0], time_span[-1]], x0, t_eval=time_span)

    # True states: shape (N, 3) after transposing sol.y (which is (3, N))
    true_states = sol.y.T

    R = np.eye(2) * R_val

    # Generate measurement noise. We draw N samples from N(0, R_val).
    # shape: (N,2). Then we multiply by scipy.linalg.sqrtm(R).T for correlated noise if needed.
    noise = np.random.normal(0, 1, (N, 2)) @ scipy.linalg.sqrtm(R).T
    #so we take (x, y) from the true states and add noise


    measurements = true_states[:, :2] + noise

    return time_span, true_states, measurements


# Generate Lorenz data
t_array, X_true, Y_meas = generate_lorenz_data()
    

def lorenz_jacobian(x_est, sigma=10, rho=28, beta=8/3):
    """Compute the Jacobian matrix for the Lorenz system at state x"""
    # Extract state variables as float values
    x1 = x_est[0]
    x2 = x_est[1]
    x3 = x_est[2]
    
    # Construct Jacobian with proper float dtype
    J = np.array([
        [-sigma, sigma, 0], 
        [rho - x3, -1, -x1],
        [x2, x1, -beta]
    ], dtype=float)
    
    return J


def kalman_update_lorenz(x, P, y, m, R):
    """
    Perform the Kalman update step with adaptive measurement direction.
    
    Args:
        x: State estimate (3x1 vector)
        P: State covariance matrix (3x3 matrix)
        y: Scalar measurement value
        sigma2: Measurement noise variance (scalar)
    
    Returns:
        x_new: Updated state estimate
        P_new: Updated state covariance matrix
    """

    m = m.reshape(-1, 1)
    x = x.reshape(-1, 1)

    # Compute predicted measurement (scalar)
    y_pred = (m.T @ x).item()
    
    # Compute innovation (difference between observed and predicted measurement)
    innovation = y - y_pred
    
    # Innovation variance (scalar)
    S = (m.T @ (P @ m)) + R
    
    # Kalman Gain (vector)
    if np.isscalar(S):
        K = (P @ m) / S
    else:
        K = (P @ m) @ np.linalg.inv(S)
    
    # Update state estimate
    x_new = x + K * innovation
    
    # Update covariance matrix using outer product
    P_new = P - np.outer(K.flatten(), (m.T @ P).flatten())
        
    return x_new.flatten(), P_new


def run_kalman_lorenz(x_true, A, Q, R_val, dt):
    """
    Run the Kalman filter on the Lorenz system with adaptive measurement direction.

    """

    time_steps = x_true.shape[0]
    x_est = np.array([0.0, 0.0, 0.0])
    P_est = np.eye(3) * 1.0
    
    x_est_array = np.zeros((time_steps, 3))
    measurements_vector = np.zeros((time_steps, 3))

    information_gain = np.zeros(time_steps)
    EntropyRate = np.zeros(time_steps)
    
    for k in range(time_steps):
        # Predict
        Jacobian = lorenz_jacobian(x_est)
        A = scipy.linalg.expm(Jacobian*dt)

        x_pred, P_pred = kalman_predict_lorenz(x_est, P_est, A, Q, dt)
        
        # True state at step k
        xk_true = x_true[k]
        
        # Generate measurement: same noise draw, but direction is the max eigenvector
        m = select_measurement_direction_from_M(P_pred, OnlyBasis = False)
        y = m @ xk_true + np.random.normal(0, np.sqrt(R_val))
        # Update
        x_est, P_est = kalman_update_lorenz(x_pred, P_pred, y, m, R_val)
        
        diff_entropy = np.log(np.sqrt(2*np.pi*np.e*np.linalg.det(Q)))


        
        EntropyRate[k] = diff_entropy + np.log(np.abs(np.linalg.det(Jacobian)) + 1e-10)            
        #information gain
        SignalNoise = (m.T @ P_pred @ m) / R_val
        information_gain[k] = 1/2*np.log(1 + SignalNoise)    


        x_est_array[k] = x_est
        measurements_vector[k, :] = m
    
    return x_est_array, measurements_vector, information_gain, EntropyRate


#run the filters
t_array, x_true, A, Q, R_val, measurement_noises, dt = generate_data()

states1, measurements_vectors = run_kalman_max_m(x_true, A, Q, R_val, measurement_noises)
#print the results for the first filter

angles = np.arctan2(measurements_vectors[:, 1], measurements_vectors[:, 0])



plt.figure(figsize=(10, 6))
plt.plot(t_array, x_true[:, 0], label='True x')
plt.plot(t_array, states1[:, 0], label='Filtered x')

# color the scatter plot
sc1 = plt.scatter(t_array, measurements_vectors[:,0], c=angles, cmap='viridis', s=5, label='vector')
plt.colorbar(sc1, label='Measurement vector angle (rad)')

plt.legend()
plt.show()

"""
Change to the Lorentz system
"""

#Lorenz data
t_array, X_true, Y_meas = generate_lorenz_data()
A = np.eye(3)
Q = np.eye(3) * 0.01
R_val = 0.2

measurements2 = Y_meas[:,0]

states2, measurement_vectors, info, entro = run_kalman_lorenz(X_true, A, Q, R_val, dt = 0.02)


angles = np.arctan2(measurement_vectors[:, 1], measurement_vectors[:, 0])

# Plot the results for the lorentz filter
plt.figure(figsize=(10, 6))
plt.plot(t_array, X_true[:, 0], label='True x')
plt.plot(t_array, states2[:, 0], label='Filtered x')

# color the scatter plot
sc1 = plt.scatter(t_array, measurement_vectors[:,0], c=angles, cmap='viridis', s=5, label='vector')
plt.colorbar(sc1, label='Measurement vector angle (rad)')
plt.legend()
plt.show()

#print the information gain and entropy rate

AverageInformationGain = np.mean(info)
AverageEntropyRate = np.mean(entro)

print(f"Average Information Gain: {AverageInformationGain}")
print(f"Average Entropy Rate: {AverageEntropyRate}")

"""
f)

By only using basis vectors the Average Information Gain reduced 0.01
While the Average Entropy Rate increased by 0.25

First one bing a 5% reduction and the second one
was a 50% reduction
"""


"""
g)
The choice of measurement direction is crucial in the Kalman filter.
The measurement direction is osiclating in both cases.
Much more frequent in the one with the ociilating system.
The lorentz system has a more complex system and the measurement direction
changes less but and keeps same direction for a longer time.
"""





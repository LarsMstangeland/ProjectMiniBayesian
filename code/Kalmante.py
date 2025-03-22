import numpy as np

# euclidean norm
def s(m):
    return np.sqrt(np.sum(m**2))


#c is normal with mean 0 and variance s(m) x sigma_y^2

def measure(x, m, sigma_y):
    
    c = np.random.normal(0, s(m) * sigma_y^2)  
    measurement_y = m.T @ x + c
    return measurement_y

def compute_eigenvector(A):
    # Compute the eigen decomposition of the matrix A
    eigvals, eigvecs = np.linalg.eig(A)
    # Find the index of the largest eigenvalue
    idx = np.argmax(eigvals)
    # Return the largest eigenvalue and the corresponding unit eigenvector
    return eigvals[idx], eigvecs[:,idx]


#find principal component of the covariance matrix
def select_measurement_direction(P):
    # Compute the largest eigenvector of the covariance matrix
    eigenvalue, eigenvector = compute_eigenvector(P)
    return eigenvector



def kalman_update(x, P, y, sigma2):
    # Select optimal measurement direction based on current uncertainty
    m = select_measurement_direction(P)
    
    # Compute innovation: difference between observed measurement and predicted measurement
    innovation = y - np.dot(m.T, x)
    
    # innovation variance
    S = np.dot(m.T, np.dot(P, m)) + sigma2
    
    # Kalman Gain.
    K = np.dot(P, m) / S
    
    # updating state estimate
    x_new = x + K * innovation
    
    # Update covariance matrix
    P_new = P - np.outer(K, np.dot(m.T, P))
    
    return x_new, P_new, m



### iterative process
def iterate_kalman(x, P, y, sigma2, n_iter):
    for i in range(n_iter):
        x, P, m = kalman_update(x, P, y, sigma2)
    return x, P, m

x = np.array([0, 0])
P = np.array([[1, 0], [0, 1]])
sigma2 = 0.1

y = 1
n_iter = 10
x_new, P_new, m = iterate_kalman(x, P, y, sigma2, n_iter)

print(x_new)
print(P_new)
print(m)








import numpy as np
import matplotlib.pyplot as plt

class static_2nd_AR:
    def __init__(self,N,a):
        self.N = N
        self.a = a
        self.ex = None
        self.S = None
        
    def generate_signal(self):
        # Gaussian random numbers as an excitation signal
        self.ex = np.random.randn(self.N)

        # Second order AR Process
        #a = np.array([1.5, -0.9])

        self.S = self.ex.copy();
        for n in range(3, self.N):
            x = np.array([self.S[n-1], self.S[n-2]])
            self.S[n] = np.dot(x, self.a) + self.ex[n]
            
    def plot(self):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,4))
        plt.tight_layout()
        ax[0].plot(range(self.N), self.ex)
        ax[0].grid(True)
        ax[0].set_title("Random Excitation Signal")
        ax[1].plot(range(self.N), self.S, color='m')
        ax[1].grid(True)
        ax[1].set_title("Autoregressive Process")

class dynamic_2nd_AR:
    def __init__(self,N):
        self.N = N
        self.ex = None
        self.A = None
        self.S = None
        
    def generate_signal(self):
        # Gaussian random numbers as an excitation signal
        self.ex = np.random.randn(self.N)

        # Second order AR Process with coefficients slowly changing in time
        a0 = np.array([1.2, -0.4])
        self.A = np.zeros((self.N,2))
        alpha = 0.1

        for n in range(self.N):
            self.A[n,0] = a0[0] + alpha * np.cos(2*np.pi*n/self.N)
            self.A[n,1] = a0[1] + alpha * np.sin(np.pi*n/self.N)

        self.S = self.ex.copy();
        for n in range(2, self.N):
            x = np.array([self.S[n-1], self.S[n-2]])
            self.S[n] = np.dot(x, self.A[n,:]) + self.ex[n]
            
    def plot(self):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,4))
        plt.tight_layout()

        ax[1,0].plot(range(self.N), self.A[:,0])
        ax[1,0].grid(True)
        ax[1,0].set_title("Coefficient a0", color='m')

        ax[1,1].plot(range(self.N), self.A[:,1], color='m')
        ax[1,1].grid(True)
        ax[1,1].set_title("Coefficient a1", color='m')

        ax[0,0].plot(range(self.N), self.ex)
        ax[0,0].grid(True)
        ax[0,0].set_title("Random Excitation Signal")

        ax[0,1].plot(range(self.N), self.S, color='m')
        ax[0,1].grid(True)
        ax[0,1].set_title("Time Varying Autoregressive Process")

class kalman_filter:
    def __init__(self,N, S, theta_n1_n1, P_n1_n1, x_n, Q, R, A,signal_type):
        self.N = N
        self.y = S.copy()
        self.theta_n1_n1 = theta_n1_n1
        self.P_n1_n1 = P_n1_n1
        self.x_n= x_n
        self.Q = Q
        self.R = R
        self.A = A
        self.signal_type = signal_type
        
        self.theta0 = []
        self.theta1 = []
    
        self.theta0.append(self.theta_n1_n1[0][0]) 
        self.theta0.append(self.theta_n1_n1[1][0]) 
        self.theta1.append(self.theta_n1_n1[0][0]) 
        self.theta1.append(self.theta_n1_n1[1][0])
        
        self.error = []
        
        self.theta_n_n = None
        self.theta_n_n1 = None
        self.P_n_n = None
        self.P_n_n1 = None
        self.k_n = None
        self.e_n = None
        
    def predict(self):
        self.theta_n_n1 = self.theta_n1_n1.copy()
        self.P_n_n1 = self.P_n1_n1 + self.Q
        
    def calculate_error(self,i):
        self.e_n = (self.y[i] - self.x_n.T@self.theta_n_n1)[0][0]
        self.error.append(self.e_n**2) 
        
    def update(self):
        num = self.P_n_n1@self.x_n
        denum = self.R + self.x_n.T@self.P_n_n1@self.x_n

        self.k_n = num/denum

        self.theta_n_n = self.theta_n_n1 + self.k_n*self.e_n
        self.P_n_n = (np.identity(2) - self.k_n@self.x_n.T)@self.P_n_n1

        self.theta0.append(self.theta_n_n[0][0])
        self.theta1.append(self.theta_n_n[1][0])

        self.theta_n1_n1 = self.theta_n_n.copy() 
        self.P_n1_n1 = self.P_n_n.copy()      

    def train(self):
        for i in range(2,self.N): 
            self.x_n[0] = self.y[i-1]
            self.x_n[1] = self.y[i-2]

            self.predict()
            self.calculate_error(i)

            self.update()
    
    def plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))

        ax[0].plot(self.theta0,label='Prediction')
        ax[1].plot(self.theta1,label='Prediction')

        if self.signal_type == 'dynamic':
            ax[0].plot(self.A[:,0], color='r',label='Target')
            ax[1].plot(self.A[:,1], color='r',label='Target')

        else:
            ax[0].axhline(y=self.A[0], color='r',label='Target')
            ax[1].axhline(y=self.A[1], color='r',label='Target')  

        ax[2].plot(self.error)

        ax[0].set_title("Coefficient 1 prediction")
        ax[1].set_title("Coefficient 2 prediction")
        ax[2].set_title("Error between target and predicted signals")

        ax[0].legend(loc='lower right')
        ax[1].legend(loc='upper right')

















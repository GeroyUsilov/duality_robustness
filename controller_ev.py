import numpy as np
import matplotlib.pyplot as plt
from simulation_functions import *
from tqdm import tqdm
from scipy.integrate import solve_ivp
import sys

ks = load_3d_matrix_from_csv("many_ks.csv")
Ks = load_3d_matrix_from_csv("many_KKs.csv")

i = int(sys.argv[1])
K = Ks[i,:,:]
k = ks[i,:,:]
g = K.shape[1]
lam = 5
lam = np.zeros(g)+lam
t_f = 200
n0 = np.zeros(g) + 0.5
nt, t = sim_dyn(n0,t_f,k,K,lam)
n_f = nt[:,-1]
J = calc_J(k,K,n_f,lam)
w, v = np.linalg.eig(J)

t_f = 5
m = 20
g = len(lam)
lam_c_0 = np.random.rand(g)-0.5 
s_hat_0 = np.random.rand(g)-0.5 
s_hat_0 = s_hat_0/np.linalg.norm(s_hat_0)
temperature = 1.0
cooling_rate = 0.96
num_iterations = 1000
best_s_hat, best_lam_c, costs = simulated_annealing(k,K,n_f,lam,lam_c_0,s_hat_0,m,t_f, num_iterations, temperature, cooling_rate,g)

fitness = evaluate_fitness_controlled(k,K,n_f,lam,best_lam_c,best_s_hat,m,t_f,g)/np.linalg.norm(1/w)

np.savetxt("best_s_hat_" + str(i) + ".csv", np.array(best_s_hat), delimiter=',')
np.savetxt("fitness" + str(i) + ".csv", np.array([fitness]), delimiter=',')
np.savetxt("best_lam_c" + str(i) + ".csv", np.array(best_lam_c), delimiter=',')

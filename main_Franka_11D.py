import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gym
import torch
import pandaenv #Library defined for the panda environment
import mujoco_py
import scipy
from pandaenv.utils import inverse_dynamics_control
import random
import time
import logging
import pandas as pd
import os
from pacsbo.pacsbo_main import compute_X_plot, ground_truth, initial_safe_samples, PACSBO, GPRegressionModel
from tqdm import tqdm
import warnings
from franka_emika_panda_simulation.manipulator import System
from gym.wrappers import RecordVideo
import pickle

# os.chdir(os.path.dirname(os.path.abspath(__file__)))


random_seed_number = 42

np.random.seed(random_seed_number)

# Fix seed for PyTorch (CPU)
torch.manual_seed(random_seed_number)




def acquisition_function(cube):
    # if sum(cube.G) != 0 and sum(cube.M) != 0:
    #     max_indices_G = torch.nonzero(cube.ucb_con[cube.G] == torch.max(cube.ucb_con[cube.G]), as_tuple=True)[0]
    #     max_indices_M = torch.nonzero(cube.ucb_rew[cube.M] == torch.max(cube.ucb_rew[cube.M]), as_tuple=True)[0]
    #     # random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
    #     if 
    # else:
    #     x_new = None
    warnings.warn("Switched from GP-UCB to most uncertain!")
    interesting_points = torch.logical_or(cube.M, cube.G)
    if sum(interesting_points) != 0:
        x_new = cube.discr_domain[interesting_points][torch.argmax(cube.var[interesting_points])]
    else: 
        x_new = None
    return x_new  # we can return cube, no parallelization needed


def update_model(cube):
    cube.compute_model(gpr=GPRegressionModel)  # compute confidence intervals?
    cube.compute_mean_var()
    cube.compute_confidence_intervals()


def compute_sets(cube):
    cube.compute_safe_set()
    cube.maximizer_routine()
    cube.expander_routine()
    

def plot_1D(cube):
    mean = cube.mean_rew.detach().numpy()
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_sample = cube.y_sample_rew.detach().numpy()
    ucb = cube.ucb_rew.detach().numpy()
    lcb = cube.lcb_rew.detach().numpy()
    plt.figure()
    plt.scatter(X_sample[1:], Y_sample[1:], color='black')
    plt.plot(X_sample[0], Y_sample[0], 'd', color='magenta', markersize=10)
    plt.plot(X_plot, mean, color='black')
    plt.fill_between(X_plot.flatten(), lcb, ucb, alpha=0.2)
    plt.title(cube.string)
    plt.xlabel('$a$')
    plt.ylabel('$y$')
    plt.show()


def noise_scenario(kappa, gamma_confidence, t, R_scenario):  # just Gaussian now
    pi_t = np.pi**2*t**2/6
    kappa_t = kappa/pi_t
    # How much N do we need?
    # Some log thing here which Lemma 2; scenario approach, we can get N directly! Then take the max.
    N_scenario = int(np.log(kappa_t)/np.log(1-gamma_confidence)) + 1  # log base change, add 1
    epsilon_t = np.abs(np.random.normal(loc=0.0, scale=R_scenario, size=N_scenario))  # taking abs value okay?
    # epsilon_t = np.abs(-R + 2*R*np.abs(np.random.uniform(size=N_scenario)))
    # epsilon_t = 1/10*np.random.standard_t(df=10, size=N_scenario)
    bar_epsilon_t = max(epsilon_t)
    '''
    t = np.array(range(1,1000))
    pi_t = np.pi**2*t**2/6
    kappa_1 = 1e-6
    kappa_t_1 = kappa_1/pi_t
    N_scenario_11 = np.log(kappa_t_1)/np.log(1-0.01) + 1
    N_scenario_12 = np.log(kappa_t_1)/np.log(1-0.001) + 1
    kappa_2 = 1e-3
    kappa_t_2 = kappa_2/pi_t
    N_scenario_21 = np.log(kappa_t_2)/np.log(1-0.01) + 1
    N_scenario_22 = np.log(kappa_t_2)/np.log(1-0.001) + 1
    plt.figure()
    plt.plot(t, N_scenario_11)
    plt.plot(t, N_scenario_12)
    plt.plot(t, N_scenario_21)
    plt.plot(t, N_scenario_22)
    plt.yscale("log")
    tikzplotlib.save("N_development.tex")
    '''
    return torch.tensor(bar_epsilon_t, dtype=torch.float32)


def update_noise_tensor(kappa_confidence, gamma_confidence, t, R, cube):
    bar_epsilon_rew_t = noise_scenario(kappa_confidence, gamma_confidence, t, R)
    bar_epsilon_con_t = noise_scenario(kappa_confidence, gamma_confidence, t, R)
    cube.bar_epsilon_tensor_rew = torch.cat((cube.bar_epsilon_tensor_rew, bar_epsilon_rew_t.unsqueeze(0)))
    cube.bar_epsilon_tensor_con = torch.cat((cube.bar_epsilon_tensor_con, bar_epsilon_con_t.unsqueeze(0)))
    return cube


def normalize_reward(reward):
    std_reward = 0.1
    mean_reward = -0.3
    reward -= mean_reward
    return reward/std_reward 

def normalize_constraint(constraint):
    upper_overshoot = 0.08
    constraint -= upper_overshoot
    return -constraint/upper_overshoot

if __name__ == '__main__':
    sys = System(rollout_limit=0, position_bound=0.5,velocity_bound=7, upper_eigenvalue=-10)
    bounds = [[1 / 3, 1], [-1, 1]]  # [1 / 3, 1]  # 
    X_plot = compute_X_plot(n_dimensions=2, points_per_axis=250, beginning=[bounds[0][0], bounds[1][0]], end=[bounds[0][1], bounds[1][1]])
    # q = X_plot[6500, 0].item()  # approx 0.4
    # r = X_plot[6500, 1].item()  # -1
    q = X_plot[6400, 0].item()  # approx 0.4
    r = X_plot[3, 1].item()  # approx -1

    params_init = np.array([q, r]) 
    params = params_init.copy()
    f, g1, g2, state = sys.simulate(params=params, update=True, render=False)  # self.sys.simulate(self.params,update=True)
    # Eigenvalue is g2. We can ignore it for now why not
    f = normalize_reward(f)
    g1 = normalize_constraint(g1)
    Y_sample_init = [torch.tensor([f], dtype=torch.float32), torch.tensor([g1], dtype=torch.float32)]
    X_sample_init = torch.tensor([q, r], dtype=torch.float32).unsqueeze(0)
    X_sample = X_sample_init.clone()
    Y_sample = Y_sample_init.copy()

    noise_type = "Gaussian"  # Student-t, Gaussian, uniform, heteroscedastic
    iterations = 200
    eta = 0.01
    R = 1e-4
    kappa_confidence = 1e-3   # 0.01  
    gamma_confidence = 0.1  
    exploration_threshold = 0.001  #  0.2
    lengthscale = 0.1  # 0.7/6  #  0.1  # 0.7/6
    RKHS_norm = 1

    safety_threshold = 0
    cube = PACSBO(delta_confidence=0.1, eta=eta, R=R, X_plot=X_plot, X_sample=X_sample,
                Y_sample=Y_sample, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold, RKHS_norm=RKHS_norm, string="Ours", lengthscale=lengthscale)
    best_parameter_reward_list = []
    best_parameter_iteration_list = []
    for t in tqdm(range(1, iterations + 1)):
        cube = update_noise_tensor(kappa_confidence, gamma_confidence, t, R, cube)
        update_model(cube)
        compute_sets(cube)
        x_new = acquisition_function(cube=cube)
        if x_new != None:
            params = x_new.detach().numpy()  # new parameter into system
            f, g1, g2, state = sys.simulate(params=params, update=True, render=False)
            # Reward 
            f = normalize_reward(f)
            f = torch.tensor(f, dtype=torch.float32).unsqueeze(0)
            Y_sample_rew = torch.cat((Y_sample[0], f), dim=0) 
            
            # Constraint
            g1 = normalize_constraint(g1)
            g1 = torch.tensor(g1, dtype=torch.float32).unsqueeze(0)
            Y_sample_con = torch.cat((Y_sample[1], g1), dim=0)
        else:
            print('Done!')
            break
        Y_sample = [Y_sample_rew, Y_sample_con]
        X_sample = torch.cat((X_sample, x_new.unsqueeze(0)), dim=0)
        cube.x_sample = X_sample
        cube.y_sample_rew = Y_sample_rew
        cube.y_sample_con = Y_sample_con
        if t%10 == 0 or t == 1:
            best_index = torch.argmax(cube.lcb_rew[cube.S]).item()
            best_parameter = cube.discr_domain[cube.S][best_index].detach().numpy()
            print(f'The best parameter is {best_parameter}')
            f, g1, g2, state = sys.simulate(params=best_parameter, update=True, render=False)
            best_parameter_reward_list.append(f)
            best_parameter_iteration_list.append(t)



best_index = torch.argmax(cube.lcb_rew[cube.S]).item()
best_parameter = cube.discr_domain[cube.S][best_index].detach().numpy()
print(f'The best parameter is {best_parameter}')
# sys.simulate(params=best_parameter, update=True, render=True)
cube.best_parameter = best_parameter
cube.best_parameter_iteration_list = best_parameter_iteration_list
cube.best_parameter_reward_list = best_parameter_reward_list
os.chdir("/u/08/tokmaka1/unix/Desktop/BO-general-noise")
with open("cube_test.pickle", "wb") as f:
    pickle.dump(cube, f)

raise Exception("Done")

plt.figure()
m = cube.mean_rew.detach().numpy()
sc = plt.scatter(
    X_plot[:, 0],
    X_plot[:, 1],
    c=m,
    cmap='plasma'
)
plt.colorbar(sc, label="Reward")  # Add a colorbar to show the mapping
plt.title("Franka mean")
plt.scatter(X_sample[:, 0], X_sample[:, 1], label="Sampled Points", color='k')
plt.scatter(X_sample[0, 0], X_sample[0, 1], label="Sampled Points", color='white')
plt.xlabel('$a_1$')
plt.ylabel('$a_2$')
plt.savefig("mean_reward_250.png")

plt.figure()
m = cube.mean_con.detach().numpy()
sc = plt.scatter(
    X_plot[:, 0],
    X_plot[:, 1],
    c=m,
    cmap='plasma'
)
plt.colorbar(sc, label="Constraint")  # Add a colorbar to show the mapping
plt.title("Franka constraint")
plt.scatter(X_sample[:, 0], X_sample[:, 1], label="Sampled Points", color='k')
plt.scatter(X_sample[0, 0], X_sample[0, 1], label="Sampled Points", color='white')
plt.xlabel('$a_1$')
plt.ylabel('$a_2$')
plt.savefig("mean_constraint_250.png")



plt.figure()
plt.plot(range(len(Y_sample[0])), Y_sample[0], label="Reward")
plt.plot(range(len(Y_sample[1])), Y_sample[1], label="Constraint")
plt.ylabel("Value")
plt.xlabel("Iteration")
plt.title("Reward and constraint development")
plt.legend()
plt.savefig("development_250.png")


plt.figure()
plt.plot(best_parameter_iteration_list, best_parameter_reward_list, label="Reward")
plt.plot(range(len(Y_sample[1])), Y_sample[1], label="Constraint")
plt.ylabel("Value")
plt.xlabel("Iteration")
plt.title("Reward and constraint development")
plt.legend()



# Add a 2D plot
print("hallo")
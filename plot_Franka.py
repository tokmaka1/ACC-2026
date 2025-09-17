import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gym
import torch
import os
from franka_emika_panda_simulation.manipulator import System
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import pickle
import tikzplotlib





def plot_mean(cube, save=False):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    plt.figure()
    m = cube.mean_rew.detach().numpy()
    sc = plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=m,
        cmap='plasma'
    )
    plt.colorbar(sc)  # Add a colorbar to show the mapping
    
    plt.scatter(X_sample[:, 0], X_sample[:, 1], color='k')
    plt.scatter(X_sample[0, 0], X_sample[0, 1], color='white', marker='D', s=50)
    plt.scatter(cube.best_parameter[0], cube.best_parameter[1], color='white', marker='*', s=200)
    if save:
        plt.savefig("mean_reward_250.pdf", dpi=600)
    else:
        plt.title("Franka mean")
        plt.xlabel('$a_1$')
        plt.ylabel('$a_2$')

def plot_constraint(cube, save=False):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    plt.figure()
    m = cube.mean_con.detach().numpy()
    sc = plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=m,
        cmap='plasma'
    )
    plt.colorbar(sc)  # Add a colorbar to show the mapping
    
    plt.scatter(X_sample[:, 0], X_sample[:, 1], color='k')
    plt.scatter(X_sample[0, 0], X_sample[0, 1], color='white', marker='D', s=50)
    plt.scatter(cube.best_parameter[0], cube.best_parameter[1], color='white', marker='*', s=200)
    if save:
        plt.savefig("mean_constraint_250.pdf", dpi=600)
    else:
        plt.title("Franka mean")
        plt.xlabel('$a_1$')
        plt.ylabel('$a_2$')

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_mean_and_constraint(cube, save=False):
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    m_rew = cube.mean_rew.detach().numpy()
    m_con = cube.mean_con.detach().numpy()

    # Shared color scale
    vmin = min(m_rew.min(), m_con.min())
    vmax = max(m_rew.max(), m_con.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'plasma'

    # Figure with an extra slim column for the colorbar
    fig = plt.figure(figsize=(10, 4), constrained_layout=False)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.08)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])  # colorbar axis at the far right

    # --- Left: mean reward ---
    sc1 = ax0.scatter(X_plot[:, 0], X_plot[:, 1], c=m_rew, cmap=cmap, norm=norm)
    ax0.scatter(X_sample[:, 0], X_sample[:, 1], color='k')
    ax0.scatter(X_sample[0, 0], X_sample[0, 1], color='white', marker='D', s=50)
    ax0.scatter(cube.best_parameter[0], cube.best_parameter[1], color='white', marker='*', s=200)
    ax0.set_title("Reward")
    ax0.set_xlabel(r'$a_1$')
    ax0.set_ylabel(r'$a_2$')

    # --- Right: mean constraint ---
    sc2 = ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=m_con, cmap=cmap, norm=norm)
    ax1.scatter(X_sample[:, 0], X_sample[:, 1], color='k')
    ax1.scatter(X_sample[0, 0], X_sample[0, 1], color='white', marker='D', s=50)
    ax1.scatter(cube.best_parameter[0], cube.best_parameter[1], color='white', marker='*', s=200)
    ax1.set_title("Constraint")
    ax1.set_xlabel(r'$a_1$')
    #ax1.set_yticks([])          # remove ticks
    ax1.set_yticklabels([])
    ax1.set_ylabel("")          # remove the label too, if you want


    # --- Single shared colorbar using the shared norm ---
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older Matplotlib
    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_label("Value")

    if save:
        fig.savefig("mean_and_constraint.pdf", dpi=600, bbox_inches="tight")
    else:
        plt.show()



def plot_max_development(cube, save=False):
    plt.figure()
    plt.plot(cube.best_parameter_iteration_list, cube.best_parameter_reward_list)
    plt.plot(range(len(cube.y_sample_con)), cube.y_sample_con.detach().numpy())
    if save:
        tikzplotlib.save("max_development.tex")

def plot_max_development_twin(cube, save=False):
    fig, ax1 = plt.subplots()

    # Left y-axis: best parameter reward
    ax1.plot(cube.best_parameter_iteration_list, cube.best_parameter_reward_list, color="tab:blue", label="Best reward")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best parameter reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Right y-axis: constraint values
    ax2 = ax1.twinx()
    ax2.plot(range(len(cube.y_sample_con)), cube.y_sample_con.detach().numpy(), color="tab:red", label="Constraint value")
    ax2.plot(range(len(cube.y_sample_con)), np.zeros(len(cube.y_sample_con)), color="tab:red")
    ax2.set_ylabel("Constraint value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Optional legend
    # fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    if save:
        # plt.savefig("max_development.pdf", dpi=300, bbox_inches="tight")
        tikzplotlib.save("max_development_twin.tex")

    plt.show()



if __name__ == '__main__':
    os.chdir("/u/08/tokmaka1/unix/Desktop/BO-general-noise")
    with open("cube_test.pickle", "rb") as f:
        cube = pickle.load(f)
    sys = System(rollout_limit=0, position_bound=0.5,velocity_bound=7, upper_eigenvalue=-10)
    params_init = np.array(cube.x_sample[0, :])
    best_params = np.array(cube.best_parameter)
    # Some sort of screenrecording"
    sys.simulate(params=params_init, update=True, render=True)
    sys.simulate(params=best_params, update=True, render=True)
    plot_mean(cube, save=False)
    plot_constraint(cube, save=False)
    plot_max_development(cube, save=False)
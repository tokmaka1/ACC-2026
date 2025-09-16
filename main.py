# BO with general noise
import warnings
import torch
import numpy as np
import gpytorch
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import dill
import os
import tikzplotlib

random_seed_number = 32  # 42 for uniform 
np.random.seed(random_seed_number)
torch.manual_seed(random_seed_number)



# Add the relative path to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Print to verify
print("Changed working directory to:", os.getcwd())

plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams.update({'font.size': 30})


from pacsbo.pacsbo_main import compute_X_plot, ground_truth, initial_safe_samples, PACSBO, GPRegressionModel




def acquisition_function(cube):
    # if sum(torch.logical_or(cube.M, cube.G)) != 0:
    #     max_indices = torch.nonzero(cube.ucb[torch.logical_or(cube.M, cube.G)] == torch.max(cube.ucb[torch.logical_or(cube.M, cube.G)]), as_tuple=True)[0]
    #     random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
    #     x_new = cube.discr_domain[torch.logical_or(cube.M, cube.G)][random_max_index, :]
    # else:
    #      x_new = None
    # return x_new  # we can return cube, no parallelization needed
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
    



def noise_scenario(kappa, gamma_confidence, t, R_scenario, x):  # just Gaussian now
    pi_t = np.pi**2*t**2/6
    kappa_t = kappa/pi_t
    # How much N do we need?
    # Some log thing here which Lemma 2; scenario approach, we can get N directly! Then take the max.
    N_scenario = int(np.log(kappa_t)/np.log(1-gamma_confidence)) + 1  # log base change, add 1
    # epsilon_t = np.abs(np.random.normal(loc=0.0, scale=R_scenario, size=N_scenario))  # taking abs value okay?
    # epsilon_t = np.abs(-R_scenario + 2*R_scenario*np.abs(np.random.uniform(size=N_scenario)))
    # epsilon_t = 1/10*np.random.standard_t(df=10, size=N_scenario)
    epsilon_t = 1/5*np.random.standard_t(df=10, size=N_scenario)*np.abs(x.item())
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


def update_noise_tensor(kappa_confidence, gamma_confidence, t, R, cube, x):
    bar_epsilon_t = noise_scenario(kappa_confidence, gamma_confidence, t, R, x)
    cube.bar_epsilon_tensor = torch.cat((cube.bar_epsilon_tensor, bar_epsilon_t.unsqueeze(0)))
    return cube





def plot(cube, gt):
    safety_threshold = cube.safety_threshold
    mean = cube.mean.detach().numpy()
    X_plot = cube.discr_domain.detach().numpy()
    X_sample = cube.x_sample.detach().numpy()
    Y_sample = cube.y_sample.detach().numpy()
    ucb = cube.ucb.detach().numpy()
    lcb = cube.lcb.detach().numpy()
    plt.figure()
    plt.plot(X_plot, gt.fX, color='blue')
    plt.scatter(X_sample[1:], Y_sample[1:], color='black')
    failures = gt.f(cube.x_sample)<safety_threshold
    plt.plot(X_sample[failures], Y_sample[failures], 'xr', markersize=25, markeredgewidth=3)    
    plt.plot(X_plot, [safety_threshold]*len(X_plot), '-r')
    plt.plot(X_sample[0], Y_sample[0], 'd', color='magenta', markersize=10)
    plt.plot(X_plot, mean, color='black')
    plt.fill_between(X_plot.flatten(), lcb, ucb, alpha=0.2)
    plt.title(cube.string)
    plt.xlabel('$a$')
    plt.ylabel('$y$')
    plt.show()




if __name__ == '__main__':
        noise_type = "heteroscedastic"  # Student-t, Gaussian, uniform, heteroscedastic
        iterations = 100
        eta = 0.001
        R = 1e-5  # 1e-3 for uniform, 1e-5 maybe Student t?
        R_gt = R
        delta_confidence = 0.1      
        bar_epsilon_tensor = torch.tensor([])

        kappa_confidence = 1e-3   # 0.01  
        gamma_confidence = 0.1  
        exploration_threshold = 0.1  #  0.2
        lengthscale = 0.1
        RKHS_norm = 1
        beta_list_chow = []
        beta_list_ours = []

        X_plot = compute_X_plot(n_dimensions=1, points_per_axis=1000)
        gt = ground_truth(num_center_points=500, dimension=1, RKHS_norm=RKHS_norm, lengthscale=lengthscale, X_plot=X_plot, noise_type=noise_type, R=R_gt)   
        safety_threshold = torch.quantile(gt.fX, 0.4).item() 
        X_sample_init, Y_sample_init = initial_safe_samples(gt, num_safe_points=1, X_plot=X_plot, R=R, safety_threshold=safety_threshold)
        

        ### Chowdhury

        X_sample_chow = X_sample_init.clone()
        Y_sample_chow = Y_sample_init.clone()
        string = "Chowdhury"
        cube_chow = PACSBO(delta_confidence=delta_confidence, eta=eta, R=R, X_plot=X_plot, X_sample=X_sample_chow,
                    Y_sample=Y_sample_chow, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold,
                    RKHS_norm=RKHS_norm, string=string, lengthscale=lengthscale)

        for t in range(1, iterations + 1):
            # cube_chow = update_noise_tensor(kappa_confidence, gamma_confidence, t, R, cube_chow)
            update_model(cube_chow)
            compute_sets(cube_chow)
            x_new = acquisition_function(cube=cube_chow)
            if x_new != None:
                y_new = torch.tensor(gt.conduct_experiment(x=x_new), dtype=torch.float32)
            else:
                # print('Done!')
                break
            Y_sample_chow = torch.cat((Y_sample_chow, y_new), dim=0)
            X_sample_chow = torch.cat((X_sample_chow, x_new.unsqueeze(0)), dim=0)
            cube_chow.x_sample = X_sample_chow
            cube_chow.y_sample = Y_sample_chow
            beta_list_chow.append(cube_chow.beta.item())

        # plot(cube_chow, gt, X_plot, X_sample_chow, Y_sample_chow, safety_threshold)


        X_sample = X_sample_init.clone()
        Y_sample = Y_sample_init.clone()
        bar_epsilon_tensor = torch.tensor([])
        string = "Ours"
        cube = PACSBO(delta_confidence=delta_confidence, eta=eta, R=R, X_plot=X_plot, X_sample=X_sample,
                    Y_sample=Y_sample, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold, RKHS_norm=RKHS_norm, string=string,
                    lengthscale=lengthscale)
        for t in range(1, iterations + 1):
            cube = update_noise_tensor(kappa_confidence, gamma_confidence, t, R, cube, X_sample[-1])  # update with last point
            update_model(cube)
            compute_sets(cube)
            x_new = acquisition_function(cube=cube)
            if x_new != None:
                y_new = torch.tensor(gt.conduct_experiment(x=x_new), dtype=torch.float32)
            else:
                # print('Done!')
                break
            Y_sample = torch.cat((Y_sample, y_new), dim=0)
            X_sample = torch.cat((X_sample, x_new.unsqueeze(0)), dim=0)
            cube.x_sample = X_sample
            cube.y_sample = Y_sample
            beta_list_ours.append(cube.beta.item())



        # plt.figure()
        # plt.plot(range(len(beta_list_chow)), beta_list_chow)
        # plt.plot(range(len(beta_list_ours)), beta_list_ours)
        # # plt.xlabel("Iterations")
        # # plt.ylabel("beta value")
        # # plt.title("Gaussian noise, true labels")
        # tikzplotlib.save("beta_value_comparison.tex")

        plot(cube, gt)
        tikzplotlib.save("ours_heavy_final.tex")

        plot(cube_chow, gt)
        tikzplotlib.save("chow_heavy_final.tex")
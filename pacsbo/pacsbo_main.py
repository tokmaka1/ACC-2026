import torch
import torch.nn as nn
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tqdm import tqdm
import time
import warnings
import copy
# import concurrent
import multiprocessing
import tikzplotlib
import pickle
import dill
from matplotlib.patches import Ellipse
import torch.multiprocessing as mp
from scipy.special import comb
# from plot import plot_1D, plot_2Dtour, plot_1D_SafeOpt_with_sets, plot_gym, plot_gym_together
import sys
import os
# from IPython import embed as IPS


class GPRegressionModel(gpytorch.models.ExactGP):  # this model has to be build "new"
    def __init__(self, train_x, train_y, eta, lengthscale):
        n_devices=1
        output_device=torch.device('cpu')
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(eta)
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # set to 0
        self.mean_module.constant.data.fill_(0.0)
        # freeze
        self.mean_module.constant.requires_grad = False
        self.kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        self.kernel.lengthscale = lengthscale
        # self.base_kernel.lengthscale.requires_grad = False; somehow does not work
        if output_device.type != 'cpu':
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                self.kernel, device_ids=range(n_devices), output_device=output_device)
        else:
            self.covar_module = self.kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




def compute_X_plot(n_dimensions, points_per_axis, beginning=0, end=1):
    if type(beginning) == int or type(beginning) == float:
        X_plot_per_domain = torch.linspace(beginning, end, points_per_axis)
        X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
        X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
    elif type(beginning) == list:
        X_plot = torch.cartesian_prod(*(torch.linspace(beginning[j], end[j], points_per_axis) for j in range(n_dimensions))).reshape(-1, n_dimensions)
    return X_plot


def initial_safe_samples(gt, num_safe_points, X_plot, R, safety_threshold):
    num_safe_points = num_safe_points
    # sampling_logic = torch.logical_and(fX > np.quantile(fX, 0.8), fX < np.quantile(fX, 0.99))
    # sampling_logic = gt.fX > safety_threshold
    # sampling_logic = torch.logical_and(gt.fX > np.quantile(gt.fX, 0.5), gt.fX < np.quantile(gt.fX, 0.7))  # uniform
    sampling_logic = torch.logical_and(gt.fX > np.quantile(gt.fX, 0.85), gt.fX < np.quantile(gt.fX, 0.95))  # Student t
    random_indices_sample = torch.randint(high=X_plot[sampling_logic].shape[0], size=(num_safe_points,))
    X_sample = X_plot[sampling_logic][random_indices_sample]
    # Y_sample = fX[sampling_logic][random_indices_sample] + gt.noise(eta)  # + torch.tensor(np.random.normal(loc=0, scale=noise_std, size=X_sample.shape[0]), dtype=torch.float32)
    Y_sample = gt.conduct_experiment(X_sample)
    return X_sample, Y_sample


class ground_truth():
    def __init__(self, num_center_points, dimension, RKHS_norm, lengthscale, X_plot, noise_type, R):
        def fun(kernel, alpha):
            return lambda X: kernel(X.reshape(-1, self.X_center.shape[1]), self.X_center).detach().numpy() @ alpha
        # For ground truth
        self.RKHS_norm = RKHS_norm
        self.noise_type=noise_type
        self.X_center = torch.rand(num_center_points, dimension)
        self.X_plot = X_plot
        alpha = np.random.uniform(-1, 1, size=self.X_center.shape[0])
        self.kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        self.kernel.lengthscale = lengthscale
        RKHS_norm_squared = alpha.T @ self.kernel(self.X_center, self.X_center).detach().numpy() @ alpha
        alpha /= np.sqrt(RKHS_norm_squared)/RKHS_norm  # scale to RKHS norm
        self.alpha= alpha
        self.f = fun(self.kernel, alpha)
        self.fX = torch.tensor(self.f(self.X_plot), dtype=torch.float32)
        self.R = R
        # self.safety_threshold = np.quantile(self.fX, 0.3)  # np.quantile(self.fX, np.random.uniform(low=0.15, high=0.5))

    def noise(self, type, x):
        if type == "Gaussian":
            return np.random.normal(loc=0.0, scale=self.R)
        if type == "uniform":
            return -self.R + 2*self.R*np.random.uniform()
        if type == "Student-t":
            return 1/10*np.random.standard_t(df=10)
        if type == "heteroscedastic":
            return 1/5*np.random.standard_t(df=10) * np.abs(x.item())
    def conduct_experiment(self, x):  # or maybe we need R, I do not know yet
        return torch.tensor(self.f(x) + self.noise(self.noise_type, x), dtype=torch.float32)


class PACSBO():
    def __init__(self, delta_confidence, eta, R, X_plot, X_sample,
                Y_sample, safety_threshold, exploration_threshold, RKHS_norm, string, lengthscale):
        # self.gt = gt  # at least for toy experiments it works like this.
        self.exploration_threshold = exploration_threshold
        self.delta_confidence = delta_confidence
        self.discr_domain = X_plot
        self.eta = torch.tensor(eta, dtype=torch.float32)
        self.n_dimensions = X_plot.shape[1]
        self.safety_threshold = safety_threshold
        self.x_sample = X_sample.clone().detach()
        self.y_sample = Y_sample.clone().detach()
        self.string = string  # type of confidence intervals; ours or Chowdhury
        self.B = RKHS_norm
        self.lengthscale = lengthscale
        self.R = torch.tensor(R, dtype=torch.float32)
        self.lcb_old = torch.ones(len(self.discr_domain))*(-np.infty)  # .flatten()
        self.ucb_old = torch.ones(len(self.discr_domain))*(np.infty)  # .flatten()
        # Needed for MC init
        self.bar_epsilon_tensor = torch.tensor([])
        self.S = self.discr_domain==self.x_sample
        if sum(self.S) == 0:
            raise Exception("Safe set is empty")


    def compute_model(self, gpr):
        self.model = gpr(train_x=self.x_sample, train_y=self.y_sample, eta=self.eta, lengthscale=self.lengthscale)
        # Does not matter which model now for covariance matrix
        self.K = self.model.covar_module(self.x_sample, self.x_sample).evaluate()  # self.model(self.x_sample).covariance_matrix

    def compute_mean_var(self):
        self.model.eval()
        self.f_preds = self.model(self.discr_domain)
        self.mean = self.f_preds.mean
        # self.var = self.f_preds.lazy_covariance_matrix.diag()  # This does not work in newer torch versions
        self.var = self.f_preds.covariance_matrix.diag()
    def compute_confidence_intervals(self):
        self.compute_beta()
        # Reward 
        self.lcb = self.mean - self.beta*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
        self.ucb = self.mean + self.beta*torch.sqrt(self.var)
        self.lcb = torch.max(self.lcb_old, self.lcb)  # pointwise!
        self.ucb = torch.min(self.ucb_old, self.ucb)
        self.lcb_old = self.lcb.clone().detach()
        self.ucb_old = self.ucb.clone().detach()



    def compute_safe_set(self):
        safe_points = self.discr_domain[self.S]
        kernel_distance_S = self.compute_kernel_distance(safe_points, self.discr_domain)
        lcb_extended = self.lcb.expand_as(kernel_distance_S)
        self.S = (lcb_extended - self.B*kernel_distance_S >= self.safety_threshold).any(dim=0)  # it only has to hold for one of them.
        # Auxiliary objects of potential maximizers M and potential expanders G
        self.G = self.S.clone()
        self.M = self.S.clone()

    def maximizer_routine(self):
        self.M[:] = False  # initialize
        self.max_M_var = 0  # initialize
        if not torch.any(self.S):  # no safe points
            return

        self.M[self.S] = self.ucb[self.S] >= max(self.lcb[self.S])
        self.M[self.M.clone()] = (self.ucb[self.M] - self.lcb[self.M]) > self.exploration_threshold
        self.M[self.M.clone()] = ~(self.discr_domain[self.M] == self.x_sample.unsqueeze(1)).any(dim=0).flatten()  # Adding this to not double-sample!
        warnings.warn("Add the double sampling line!")
        if not torch.any(self.M):
            return
        self.max_M_var = torch.max(self.ucb[self.M] - self.lcb[self.M])
        self.max_M_ucb = torch.max(self.ucb[self.M])

    def expander_routine(self):
        self.G[:] = False  # initialize
        if not torch.any(self.S) or torch.all(self.S):  # no safe points or all of them are safe points
            return
        # no need to consider points in M
        # Expanders is about constraints!
        s = torch.logical_and(self.S, ~self.M)  # do not consider points that are already maximizers
        s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.max_M_var
        s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.exploration_threshold  # only sufficiently uncertain.
        # still same size as the safe set! We are just over-writing the positive ones
        if not torch.any(s):
            return
        potential_expanders = self.discr_domain[s]
        unsafe_points = self.discr_domain[~self.S]
        kernel_distance = self.compute_kernel_distance(potential_expanders, unsafe_points)
        ucb_expanded = self.ucb[s].unsqueeze(1).expand(-1, kernel_distance.size(1))
        s[s.clone()] = torch.any(ucb_expanded - self.B*kernel_distance > self.safety_threshold, dim=1)
        # or go with for loop
        # boolean_expander = ~s[s.clone()]  # assume that all are NOT expanders and go in the loop
        # for i in range(len(potential_expanders)):
        #     potential_expander = potential_expanders[i]
        #     for unsafe_point in unsafe_points:
        #         if self.ucb[s][i] - self.compute_kernel_distance(potential_expander, unsafe_point) > self.safety_threshold:
        #             boolean_expander[i] = True
        #             break  # we only need one!  
        # s[s.clone()] = boolean_expander  # update the potential expanders on whether they can potentially expand to an unsafe point            
        self.G = s
        self.G[self.G.clone()] = ~(self.discr_domain[self.G] == self.x_sample.unsqueeze(1)).any(dim=0).flatten()  # Adding this to not double-sample!
        warnings.warn("Check double sample!")

    def compute_beta(self):
        if self.string == "Chowdhury":
            # Bound from Chowdhury et al. 2017
            inside_log = torch.det(torch.eye(self.x_sample.shape[0]) + self.K/self.eta)
            inside_sqrt = torch.log(inside_log) - (2*torch.log(torch.tensor(self.delta_confidence)))
            # self.beta = self.B + self.R/(torch.sqrt(self.eta))*torch.sqrt(inside_sqrt)
            self.beta = self.B + self.R/(torch.sqrt(self.eta))*torch.sqrt(inside_sqrt)
        elif self.string == "Ours":
            # Data-driven bound
            Xi = self.K @ torch.inverse(self.K + (self.eta)*torch.eye(self.x_sample.shape[0]))
            eigenvalues, _ = torch.linalg.eigh(Xi)
            lambda_max = max(eigenvalues)
            normed_noise = torch.norm(self.bar_epsilon_tensor)*torch.sqrt(lambda_max)
            self.beta = self.B + 1/(torch.sqrt(self.eta))*normed_noise
            # print(self.beta)


    def compute_kernel_distance(self, x, x_prime):
        k_xx = self.model.kernel(x, x).diag()
        K_x_xp = self.model.kernel(x, x_prime).evaluate()
        K_xx = k_xx[:,None].expand_as(K_x_xp)
        complete_matrix = torch.sqrt(2*K_xx - 2*K_x_xp)
        return complete_matrix
       
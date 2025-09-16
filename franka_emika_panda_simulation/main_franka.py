from __future__ import print_function, division, absolute_import

import GPy
import gymnasium as gym
#import GPyOpt
import numpy as np 
import pandas as pd
import copy 
# from lin_sys import lin_sys
# import safeopt
# from safeopt import SafeOptSwarm
# from diffusion_optimization import DiffOpt
# from IPython import embed as IPS
from robot_grasp import robot_grasp
import matplotlib.pyplot as plt
import matplotlib
from tikzplotlib import save as tikz_save
import time
import pickle
import dill
import random

class safe_opt:

    def __init__(self, noise_var=1e-4, error_bound=0.25, lengthscale=0.1, ARD=True, init_opt=True):
        self.noise_var, self.error_bound = noise_var, error_bound
        self.reward_factor=1
        self.robot=robot_grasp("172.16.0.2",w=np.pi/5000,sampling_time=1/200,reward_factor=self.reward_factor)
        self.steps=10000 # Corresponsd to 3 full rotations
        self.first_start=True
        params=np.asarray([0.6,0.6,0.6])
        self.safe_params=params.copy()
        self.failures_safeopt = 0
        self.failures_diffopt = 0
        self.seed=3 #1,2,3
        self.tol=0.076
        self.update_tol()
        #print(self.tol)
        # Simulate experiments
        if init_opt:
            reward,constraint = self.simulate(params)
            f=reward
            g=constraint
            reward=np.asarray([[reward]])
            constraint=np.asarray([[constraint]])
            reward=reward.reshape(-1,1)
            constraint=constraint.reshape(-1,1)
            # Define GP
            L = [lengthscale,lengthscale,lengthscale]
            x=params.reshape(1,-1)
            KERNEL_f = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
            KERNEL_g = GPy.kern.sde_RBF(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
            gp0 = GPy.models.GPRegression(x, reward, noise_var=0.05 ** 2, kernel=KERNEL_f)
            gp1 = GPy.models.GPRegression(x, constraint, noise_var=0.3 ** 2, kernel=KERNEL_g)
            # Setup optimizer
            bounds = [[0.0, 1.2], [0.0,1.2],[0.0, 1.2]]
            self.opt = SafeOptSwarm([gp0, gp1], fmin=[-np.inf, 0.25], bounds=bounds, beta=3)
            self.h = 0.5
            self.delta = 0.01
            self.l = 1.75
            parameter_set = safeopt.linearly_spaced_combinations(bounds, num_samples=100)
            self.diff_opt = DiffOpt(self.h, self.delta, self.l, [-np.inf, 0.25], parameter_set, 1, 0, self.noise_var)
            # Store data
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.dataset=pd.DataFrame(np.asarray([[params[0],params[1],params[2],f,g,f]]),columns=['K_x','K_y','K_z','f','g','f_max'])
            y = np.array([[f], [g]])
            y = y.squeeze()
            self.opt.add_new_data_point(x,y)
            self.diff_opt.add_new_data_point(x, y)

        self.time_safeopt_recorded = []
        self.time_diffopt_recorded = []

    def update_tol(self):
        ''' Run experiment with default parameter and update tolerance based on
         current sensor measurement '''
        params=np.asarray([0.6,0.6,0.6])
        reward,constraint = self.simulate(params)
        constraint*=self.tol/10
        constraint+=self.tol
        default_value=1.295
        self.tol=np.maximum(constraint/(1+default_value/10),0.07)
		
    def optimize(self, step, run_safeopt=True, run_diffopt=False):
        '''
        Performs 1 optimization step, i.e. recommend a parameter and run experiments with it
        '''
        # Get parameter
        if run_safeopt:
            start_time = time.time()
            param = self.opt.optimize()
            self.time_safeopt_recorded.append(time.time() - start_time)
            print(param, end="")
	# Run experiment with parameter
            f, g = self.simulate(param)
       # Update and add collected data to GP
            y = np.array([[f], [g]])
            y = y.squeeze()
            self.opt.add_new_data_point(param.reshape(1, -1), y)
            constraint_satisified = g >= 0 
            self.failures_safeopt += constraint_satisified
            print("\n SafeOpt")
            print(f, g, constraint_satisified)
            maximum, fval = self.opt.get_maximum()
            if step%5 == 0:
                f_max, g_max = self.simulate(maximum)
                df = pd.DataFrame(np.asarray([[f_max]]), columns=['f_max'])
                # self.dataset = self.dataset.append(df)
                self.dataset = pd.concat([self.dataset, df])

        if run_diffopt:
            start_time = time.time()
            param = self.diff_opt.optimize()
            self.time_diffopt_recorded.append(time.time() - start_time)
            print(param, end="")
            # Run experiment with parameter
            f, g = self.simulate(param)
            # Update and add collected data to GP
            y = np.array([[f], [g]])
            y = y.squeeze()
            self.diff_opt.add_new_data_point(param.reshape(1, -1), y)
            constraint_satisified = g >= 0 
            self.failures_diffopt += constraint_satisified
            print("\n DiffOpt")
            print(f, g, constraint_satisified)
            maximum, fval = self.diff_opt.get_maximum()
            if step%5 == 0:
                f_max, g_max = self.simulate(maximum)
                df = pd.DataFrame(np.asarray([[f_max]]), columns=['f_max'])
                self.dataset = pd.concat([self.dataset, df])

    def simulate(self,params):
        '''
        Simulate trajectory on the robot for the provided params

        Inputs:
        params: ndarray, Parameters recommended by safeopt to investigate

        Outputs: 
        
        total_reward: Total accumulated rewards

        constraint: Value of the constraint

        '''

        
        # Calculate impedances from the parameters
        impedance=np.multiply(params,self.robot.default_impedance)
        
        

        # If running the experiments for the first time, don't have to call setup_experiment
        if self.first_start:
            self.first_start=False

            self.robot.change_impedance(impedance)
                
            k=0
            total_reward=0
            #constraint=np.inf
            constraint=[]
            # Loop till 3 circulations are over 
            while(k<self.steps):
                # Take a step
                late,state,reward,dist=self.robot.step(k)
                # If targets were updated, update k
                if not late:
                    k+=1
                # Get constraint values and add rewards
                dist-=self.tol
                dist=dist/self.tol*10
                constraint.append(dist)
                total_reward+=reward
            
        else:
            # go to start position for the experiment
            self.robot.setup_experiment()
            # Change robot impedance
            self.robot.change_impedance(impedance)
            # Same as before
            k=0
            total_reward=0
            constraint=[]
            while(k<self.steps):
                late,state,reward,dist=self.robot.step(k)
                if not late:
                    #print("Updated_k")
                    k+=1
                dist-=self.tol
                dist=dist/self.tol*10
                constraint.append(dist)
                total_reward+=reward
        
        # Normalize reward
        total_reward/=self.robot.num_samples
        total_reward+=0.7*self.error_bound
        total_reward/=0.5*self.error_bound
        constraint=np.asarray(constraint)
        constraint=np.min(constraint)
        print(self.robot.num_samples,total_reward,constraint,self.robot.frequency_diff/self.robot.num_samples)
        # self.robot.step(np.inf)
        return total_reward,constraint

    def save(self,enumerator=0):
        '''
        Saves dataset and optimizer

        Inputs: 
        enumerator: Used to indicate which iteration we are saving
        '''
        
        # Save dataset to csv
        self.dataset.to_csv("SafeOpt_robotArm_grasp"+str(enumerator)+".csv",index=False)
        # Save safeoptswarm class using pickle
        with open('optimizer'+str(enumerator)+'.pkl', 'wb') as output:
            dill.dump(self.opt,output,pickle.HIGHEST_PROTOCOL)
        with open('optimizer_diff_opt'+str(enumerator)+'.pkl', 'wb') as output:
            dill.dump(self.diff_opt,output,pickle.HIGHEST_PROTOCOL)

    def load(self,enumerator=0):
        '''
        Loads dataset and optimizer

        Inputs:
        enumerator: Used to tell which iteration to load
        '''

        # Load data from csv
        self.dataset=pd.read_csv("SafeOpt_robotArm_grasp"+str(enumerator)+".csv")
        # Load safeoptswarm class
        with open('optimizer'+str(enumerator)+'.pkl', 'rb') as input:
            self.opt=dill.load(input)
        
        random.seed(self.seed)
        np.random.seed(self.seed)

def Optimize(num_optimizations=25,save_frequency=2,Load_enumerator=None):
    '''
    Runs all optimization iterations
    '''
    # Initialize optimizer
    
    offset=0
    if Load_enumerator is not None:
        Optimizer=safe_opt(init_opt=False)
        Optimizer.load(enumerator=Load_enumerator)
        offset=Load_enumerator
    else:
        Optimizer=safe_opt()
    # Run optimization steps
    for i in range(num_optimizations):
        # Save data based on frequency
        if i%save_frequency==0:
            Optimizer.save(enumerator=i+offset)        
        # Run 1 optimization step
        Optimizer.optimize(i)

    
    Optimizer.save(enumerator=i+1+offset)  

if __name__ == '__main__':
    Optimize()

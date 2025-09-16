import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gym
import pandaenv #Library defined for the panda environment
import mujoco_py
import scipy
from pandaenv.utils import inverse_dynamics_control
import random
import time
import logging
plt.rcParams.update({'font.size': 16})
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class System(object):

    def __init__(self,position_bound,velocity_bound,rollout_limit=0,upper_eigenvalue=0):
        self.env = gym.make("PandaEnvBasic-v0")
        
        # Define Q,R, A, B, C matrices
        self.Q=np.eye(6)
        self.R=np.eye(3)/100
        self.env.seed(0)
        self.obs = self.env.reset()
        self.A = np.zeros([6, 6])
        # A=np.zeros([18,18])
        self.A[:3, 3:] = np.eye(3)
        self.B = np.zeros([6, 3])
        self.B[3:, :] = np.eye(3)
        self.T=2000
        # Set up inverse dynamics controller
        self.ID = inverse_dynamics_control(env=self.env, njoints=9, target=self.env.goal)
        self.id = self.env.sim.model.site_name2id("panda:grip")
        self.rollout_limit=rollout_limit
        self.at_boundary=False
        self.Fail=False
        self.approx=True
        self.position_bound=position_bound
        self.velocity_bound=velocity_bound
        self.upper_eigenvalue=upper_eigenvalue
        # Define weighting matrix to set torques 4,6,7,.. (not required for movement) to 0.
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        self.N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))

    def simulate(self,params=None,render=False,opt=None,update=False):
        # Simulate the robot/run experiments
        x0=None
        # Update parameters
        if params is not None:

            if update:
                param_a=self.set_params(params)
                self.Q = np.diag(param_a)
            else:
                self.Q=np.diag(params)

            self.R=np.eye(3)/100*np.power(10,3*params[1]) #param is between -1 and 1
            # If want to change inition condition update, IC
            if opt is not None:
                if opt.criterion in ["S2"]:
                    x0=params[opt.state_idx]
                    x0[3:]=np.zeros(3)

        # Get controller
        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
        K = scipy.linalg.inv(self.R) * (self.B.T * P)


        K = np.asarray(K)
        eigen_value = np.linalg.eig(self.A - np.dot(self.B, K))
        eigen_value = np.max(np.asarray(eigen_value[0]).real)

        Kp = K[:, :3]
        Kd = K[:, 3:]
        Objective=0
        self.reset(x0)
        state = []
        constraint2 = 0
        if x0 is not None:
            x=np.hstack([params[:2].reshape(1,-1),x0.reshape(1,-1)])
            state.append(x)

        else:
            obs = self.obs["observation"].copy()
            obs[:3] = obs[:3] - self.env.goal
            if self.position_bound == 0.0 or self.velocity_bound == 0.0:
                breakpoint()
            obs[:3] /= self.position_bound
            obs[3:] /= self.velocity_bound
            x = np.hstack([params[:2].reshape(1, -1), obs.reshape(1, -1)])
            state.append(x)


        if opt is not None:
            if eigen_value>self.upper_eigenvalue and opt.criterion=="S3":
                self.at_boundary=True
                opt.s3_steps=np.maximum(0,opt.s3_steps-1)
                return 0,0,0,state

        elif eigen_value>self.upper_eigenvalue:
            self.at_boundary=True
            self.Fail=True
            print("Eigenvalues too high ",end="")
            return 0,0,0,state


        if render:
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            #init_dist=self.init_dist
            for i in range(self.T):

                bias = self.ID.g()


                J = self.ID.Jp(self.id)

                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"])) - np.dot(Kd, self.obs["observation"][3:]
                                                                                                     - np.ones(3) * 1 / self.env.Tmax * (i < self.env.Tmax))
                u=-bias


                u += np.dot(J.T, wM_des)
                u=np.dot(self.N_bar,u)
                self.obs, reward, done, info = self.env.step(u)
                Objective+=reward
                constraint2=np.maximum(np.linalg.norm(self.env.goal - self.obs["achieved_goal"])-init_dist,constraint2)
                #constraint_2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint_2)
                self.env.render()


        else:
            # Simulate the arm
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            #init_dist = self.init_dist
            for i in range(self.T):
                if opt is not None and not self.at_boundary:
                    obs=self.obs["observation"].copy()
                    obs[:3]-=self.env.goal
                    obs[:3]/=self.position_bound
                    obs[3:]/=self.velocity_bound
                    #obs[3:]=opt.x_0[:,3:]
                    # Evaluate Boundary condition/not necessary here as we use SafeOpt
                    if i %10==0:
                        self.at_boundary, self.Fail, params = opt.check_rollout(state=obs.reshape(1,-1), action=params)

                    if self.Fail:
                        print("FAILED                  ",i,end=" ")
                        return 0, 0,0,state
                    elif self.at_boundary:
                        params = params.squeeze()
                        print(" Changed action to",i,params,end="")
                        param_a = self.set_params(params.squeeze())
                        self.Q = np.diag(param_a)

                        self.R=np.eye(3) / 100 * np.power(10, 3 * params[1])
                        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
                        K = scipy.linalg.inv(self.R) * (self.B.T * P)

                        K = np.asarray(K)

                        Kp = K[:, :3]
                        Kd = K[:, 3:]

                # Collect rollouts (not necessary here as we run safeopt)
                if i < self.rollout_limit:
                    obs=self.obs["observation"].copy()
                    obs[:3]=obs[:3]-self.env.goal
                    obs[:3] /= self.position_bound
                    obs[3:] /= self.velocity_bound
                    x=np.hstack([params[:2].reshape(1,-1),obs.reshape(1,1)])
                    state.append(x)
                bias = self.ID.g()


                J = self.ID.Jp(self.id)


                wM_des = np.dot(Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"]))-np.dot(Kd,self.obs["observation"][3:]-np.ones(3)*1/self.env.Tmax*(i<self.env.Tmax))
                u=-bias
                u += np.dot(J.T, wM_des)
                u = np.dot(self.N_bar, u)
                self.obs, reward, done, info = self.env.step(u)
                Objective+=reward
                constraint2 = np.maximum(np.linalg.norm(self.env.goal - self.obs["achieved_goal"]) - init_dist,
                                         constraint2)

                #constraint2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint2)





        return Objective/self.T,constraint2/init_dist,eigen_value,state


    def reset(self,x0=None):
        '''
        Reset environmnent for next experiment
        '''
        self.obs = self.env.reset()
        #self.init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
        self.Fail=False
        self.at_boundary=False

        if x0 is not None:
            x0*=self.position_bound
            self.env.goal=self.obs["observation"][:3]-x0[:3]




    def set_params(self, params):
        '''
        Update parameters for controller
        '''
        q1 = np.repeat(np.power(10, 6*params[0]),3) #param is between -1 and 1
        q2 = np.sqrt(q1)*0.1
        updated_params = np.hstack((q1.squeeze(), q2.squeeze()))
        return updated_params


if __name__ == '__main__':
    params = 1*np.asarray([0.6,0.6,0.6])
    sys = System(position_bound=0.15, velocity_bound=0.6, rollout_limit=500)
    sys.simulate(params=params, render=True, update=True)

from __future__ import print_function, division, absolute_import

# import GPy
# import gymnasium as gym
import gym
#import GPyOpt
import numpy as np 
import copy 
# from lin_sys import lin_sys
# import safeopt
# from diffusion_optimization import DiffOpt
from manipulator import System
# from IPython import embed as IPS
import matplotlib.pyplot as plt
import matplotlib
from tikzplotlib import save as tikz_save
import time

class safe_opt:

	def __init__(self, noise_var=1e-4, upper_overshoot=0.08, upper_eigenvalue=-10, lengthscale=0.7, ARD=True):
		self.noise_var, self.upper_overshoot, self.upper_eigenvalue = noise_var, upper_overshoot, upper_eigenvalue
		q = 4/6
		r = -1
		self.params = np.asarray([q, r])
		self.failures_safeopt = 0
		self.failures_diffopt = 0
		self.failure_overshoot_safeopt = 0
		self.failure_overshoot_diffopt = 0
		self.mean_reward = -0.33
		self.std_reward = 0.14
		self.eigen_value_std = 21
		# Set constants
		self.h = 0.5
		self.delta = 0.01
		self.l = 1.75
		# self.sys = lin_sys()
		# Set up system and optimizer
		self.sys = System(rollout_limit=0, position_bound=0.5,
						  velocity_bound=7,
						  upper_eigenvalue=self.upper_eigenvalue)
		f, g1, g2, state = self.sys.simulate(self.params, update=True)
		g2 = self.upper_eigenvalue - g2
		g2 /= self.eigen_value_std
		g1 -= self.upper_overshoot
		g1 = -g1 / self.upper_overshoot
		f -= self.mean_reward
		f /= self.std_reward
		f = np.asarray([[f]])
		f = f.reshape(-1, 1)
		g1 = np.asarray([[g1]])
		g1 = g1.reshape(-1, 1)
		L = [lengthscale / 6, lengthscale / 6]
		x=self.params.reshape(1,-1)
		KERNEL_f = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
		KERNEL_g = GPy.kern.sde_Matern32(input_dim=x.shape[1], lengthscale=L, ARD=ARD, variance=1)
		gp0 = GPy.models.GPRegression(x[0, :].reshape(1, -1), f, noise_var=self.noise_var, kernel=KERNEL_f)
		gp1 = GPy.models.GPRegression(x[0, :].reshape(1, -1), g1, noise_var=self.noise_var, kernel=KERNEL_g)
		# self.state_bounds = self.sys.state_limits
		# Bounds for the cart-pole system
		bounds = [[1 / 3, 1], [-1, 1]]
		parameter_set = safeopt.linearly_spaced_combinations(bounds, num_samples=1000)
		self.opt = safeopt.SafeOpt([gp0, gp1], parameter_set, fmin=[-np.inf, 0], beta=3.5)

		self.diff_opt = DiffOpt(self.h, self.delta, self.l, [-np.inf, 0], parameter_set, 1, 0, self.noise_var)

		self.time_safeopt_recorded = []
		self.time_diffopt_recorded = []
		# Collect more initial policies for S_0
		self.simulate_data()

	def simulate_data(self):
		'''
		Collect more initial policies
		'''
		p = [5/6,1]
		d = [-0.9,-2/3]

		for i in range(2):
			self.params = np.asarray([p[i],d[i]])
			f, g1,g2,state = self.sys.simulate(self.params,update=True)
			g2 = self.upper_eigenvalue - g2
			g2 /= self.eigen_value_std
			g1 -= self.upper_overshoot
			g1 = -g1/self.upper_overshoot
			f -= self.mean_reward
			f /= self.std_reward
			print(f, g1, g2)
			y = np.array([[f], [g1]])
			y = y.squeeze()
			self.opt.add_new_data_point(self.params.reshape(1, -1), y)
			self.diff_opt.add_new_data_point(self.params.reshape(1, -1), y)
		
	def setup_optimization(self,num_iter):
		self.num_it = num_iter
		# The statistical model of our objective function
		y_init = self.obj_fun(self.param)
		self.gp = GPy.models.GPRegression(self.param, self.obj_fun(self.param)[:,0,None], self.kernel, noise_var=self.noise_var)
		self.gp_constr = GPy.models.GPRegression(self.param, self.obj_fun(self.param)[:,1,None], self.kernel_constr, noise_var=self.noise_var)
		self.gp_constr_2 = GPy.models.GPRegression(self.param, self.obj_fun(self.param)[:,1,None], self.kernel_constr, noise_var=self.noise_var)
		# The optimization routine
		self.opt = safeopt.SafeOpt([self.gp, self.gp_constr, self.gp_constr_2], self.parameter_set, [-np.inf, -10, -10], beta=3)
		self.diff_opt = DiffOpt(self.h, self.delta, self.l, [-np.inf, 0, 0], self.parameter_set, len(self.state_bounds), self.threshold, self.noise_var)
		self.diff_opt.add_new_data_point(self.param, y_init)
		self.sys.reset()
		
	def optimization(self, run_safeopt=True, run_diffopt=True):
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
			f, g1, g2, state = self.sys.simulate(param, update=True)
			# Update and add collected data to GP
			g2 = self.upper_eigenvalue - g2
			g2 /= self.eigen_value_std
			g1 -= self.upper_overshoot
			g1 = -g1/self.upper_overshoot
			f -= self.mean_reward
			f /= self.std_reward

			y = np.array([[f], [g1]])
			y = y.squeeze()
			self.opt.add_new_data_point(param.reshape(1, -1), y)
			constraint_satisified = g1 >= 0 and g2 >= 0
			if not constraint_satisified:
				self.failure_overshoot_safeopt += -(g1*self.upper_overshoot) + self.upper_overshoot
			self.failures_safeopt += constraint_satisified
			print("\n SafeOpt")
			print(f, g1, g2, constraint_satisified)

		if run_diffopt:
			start_time = time.time()
			param = self.diff_opt.optimize()
			self.time_diffopt_recorded.append(time.time() - start_time)
			print(param, end="")
			# Run experiment with parameter
			f, g1, g2, state = self.sys.simulate(param, update=True)
			# Update and add collected data to GP
			g2 = self.upper_eigenvalue - g2
			g2 /= self.eigen_value_std
			g1 -= self.upper_overshoot
			g1 = -g1/self.upper_overshoot
			f -= self.mean_reward
			f /= self.std_reward

			y = np.array([[f], [g1]])
			y = y.squeeze()
			self.diff_opt.add_new_data_point(param.reshape(1, -1), y)
			constraint_satisified = g1 >= 0 and g2 >= 0
			if not constraint_satisified:
				self.failure_overshoot_diffopt += -(g1*self.upper_overshoot) + self.upper_overshoot
			self.failures_diffopt += constraint_satisified
			print("\n DiffOpt")
			print(f, g1, g2, constraint_satisified)

	def obj_fun(self, param):
		self.sys.reset()
		err = 0
		constr = np.inf*np.ones(len(self.state_bounds))
		reward_st = 0
		try:
			ctrl = np.hstack((param, np.array([[1, 1]])))
		except:
			ctrl = np.hstack((param.reshape(1, -1), np.array([[1, 1]])))
		for _ in range(100):
			action = ctrl@self.state.reshape(-1, 1)
			self.state, reward, _, _, _ = self.sys.step(action.flatten())
			reward_st += reward
			# action = self.sys.k@self.state
			# err += (self.state.T@self.state + action.T@action).item()
			dist_constr = [bound[1] - np.abs(self.state[idx]) for idx, bound in enumerate(self.state_bounds)]
			constr = [np.min([constr[idx], dist_constr[idx]]) for idx in range(len(self.state_bounds))]
			if np.min(constr) < 0:
				print('safety constraint violated!!!')
			# if np.min(dist_constr) < constr:
			#     constr = np.min(dist_constr)
		constr.insert(0, reward_st)
		return np.array(constr).reshape(1, -1)

	def show(self):
		x_eval = safeopt.linearly_spaced_combinations(self.bounds_param, 100)
		plt.subplot(2, 1, 1)
		plt.title('SafeOpt estimate')
		gp_mean, gp_std = self.gp.predict(x_eval)
		new_order = np.argsort(x_eval[:, 0].flatten())
		x_eval_plot = x_eval[:, 0].flatten()[new_order]
		gp_mean = gp_mean.flatten()[new_order]
		gp_std = gp_std.flatten()[new_order]
		plt.plot(x_eval_plot, gp_mean)
		plt.fill_between (x_eval_plot, gp_mean + 3*gp_std, gp_mean - 3*gp_std, color='lightgray')

		plt.subplot(2, 1, 2)
		plt.title('Diffusion-based learning')
		NW_est, NW_bound = self.diff_opt.NW_estimate(x_eval, 1)  
		new_order = np.argsort(x_eval[:, 0].flatten())
		x_eval_plot = x_eval[:, 0].flatten()[new_order]
		NW_est = NW_est.flatten()[new_order]
		NW_bound = NW_bound.flatten()[new_order]
		plt.plot(x_eval_plot, NW_est)
		plt.fill_between (x_eval_plot, NW_est + NW_bound, NW_est - NW_bound, color='lightgray')
		plt.savefig('comparison.png')
		plt.show()

def run_optimization():
	iterations = 401
	div = 5
	runs = 1
	# Ploting of safe set
	plot=True
	Reward_data_safeopt = np.zeros([int(iterations/div) + 1, runs])
	Reward_data_diffopt = np.zeros([int(iterations/div) + 1, runs])
	Overshoot_summary_safeopt = np.zeros([2, runs])
	Overshoot_summary_diffopt = np.zeros([2, runs])
	for r in range(runs):
		j = 0
		opt = safe_opt()
		if r > 0:
			# Stop ploting safe set after 1 full run
			plot=False
		for i in range(iterations):
			if i%div == 0:
				# Record optimum found after every 5 iterations
				maximum, f = opt.opt.get_maximum()
				f, g1, g2,dummy = opt.sys.simulate(maximum, update=True)
				f -= opt.mean_reward
				f /= opt.std_reward
				Reward_data_safeopt[j, r] = f
				maximum, f = opt.diff_opt.get_maximum()
				f, g1, g2,dummy = opt.sys.simulate(maximum, update=True)
				f -= opt.mean_reward
				f /= opt.std_reward
				Reward_data_diffopt[j, r] = f
				j += 1
				if plot and i%20 == 0:
					# Plot safe set after every 20 iterations
					
					# Define parameter space and determine lower bounds
					q=np.linspace(-1,1,25)
					r_cost=np.linspace(-1,1,25)
					a = np.asarray(np.meshgrid(q, r_cost)).T.reshape(-1, 2)
					input=a
					mean, var = opt.opt.gps[1].predict(input)
					std=np.sqrt(var)
					l_x0 = mean -opt.opt.beta(opt.opt.t)*std
					safe_idx=np.where(l_x0>=0)[0]
					values=np.zeros(a.shape[0])
					values[safe_idx]=1

					mean, var = opt.opt.gps[0].predict(input)
					l_f = mean - opt.opt.beta(opt.opt.t) * std

					safe_l_f=l_f[safe_idx]
					safe_max=np.where(l_f==safe_l_f.max())[0]
					optimum_params=a[safe_max,:]
					optimum_params=optimum_params.squeeze()
					q=np.reshape(a[:,0],[25,25])
					r_cost = np.reshape(a[:, 1], [25, 25])
					values = values.reshape([25, 25])
					colours = ['r', 'g']
					plt.rcParams['text.usetex'] = True
					fig = plt.figure(figsize=(12, 12))
					left, bottom, width, height = 0.15, 0.15, 0.8, 0.8
					ax = fig.add_axes([left, bottom, width, height])
					ax.set_xlabel(r'$q$', fontsize=50)
					ax.set_ylabel(r'$r$', fontsize=50)
					plt.xticks(fontsize=50)
					plt.yticks(fontsize=50)
					cs = ax.contourf(q*6, r_cost*3, values)
					ax.scatter(q*6, r_cost*3, c=values, cmap=matplotlib.colors.ListedColormap(colours))
					ax.scatter(optimum_params[0]*6, optimum_params[1]*3, marker="<", color="b", s=np.asarray([200]))
					# if iterations - i > 20:
					# 	ax.set_title("Safe Set Belief SafeOpt, iter "+str(i))
					ax.set_ylim([-3.1, 3.1])
					ax.set_xlim([-6.1, 6.1])

					# if iterations - i < 20:
					# 	tikz_save('Safeset_safeopt.tex')
					# else:
					plt.savefig('Safeset_safeopt_time' + str(i) +'.png', dpi=300)
					plt.close()

					NW_est, NW_bound = opt.diff_opt.NW_estimate(input, 1)
					l_x0 = NW_est - NW_bound
					safe_idx=np.where(l_x0>=0)[0]
					values=np.zeros(a.shape[0])
					values[safe_idx] = 1

					NW_est, NW_bound = opt.diff_opt.NW_estimate(input, 0)
					l_f = NW_est - NW_bound

					safe_l_f=l_f[safe_idx]
					safe_max=np.where(l_f==safe_l_f.max())[0]
					optimum_params=a[safe_max,:]
					optimum_params=optimum_params.squeeze()
					q=np.reshape(a[:,0],[25,25])
					r_cost = np.reshape(a[:, 1], [25, 25])
					values = values.reshape([25, 25])
					colours = ['r', 'g']
					plt.rcParams['text.usetex'] = True
					fig = plt.figure(figsize=(12, 12))
					left, bottom, width, height = 0.15, 0.15, 0.8, 0.8
					ax = fig.add_axes([left, bottom, width, height])
					ax.set_xlabel(r'$q$', fontsize=50)
					ax.set_ylabel(r'$r$', fontsize=50)
					plt.xticks(fontsize=50)
					plt.yticks(fontsize=50)
					cs = ax.contourf(q*6, r_cost*3, values)
					ax.scatter(q*6, r_cost*3, c=values, cmap=matplotlib.colors.ListedColormap(colours))
					ax.scatter(optimum_params[0]*6, optimum_params[1]*3, marker="<", color="b", s=np.asarray([200]))
					# if iterations - i > 20:
					# 	ax.set_title("Safe Set Belief DiffOpt, iter "+str(i))
					ax.set_ylim([-3.1, 3.1])
					ax.set_xlim([-6.1, 6.1])
					# if iterations - i < 20:
					# 	tikz_save('safeset_diffopt.tex')
					# else:
					plt.savefig('Safeset_diffopt_time' + str(i) +'.png', dpi=300)
					plt.close()

			
			# Optimization step
			opt.optimization()
			print(i)
		print("Safety SafeOpt")
		print(opt.failures_safeopt / iterations)
		print("Safety DiffOpt")
		print(opt.failures_diffopt / iterations)
		# Measure failures (constraint violation) and magnitude of violation
		Overshoot_summary_safeopt[0, r] = opt.failures_safeopt / iterations
		Overshoot_summary_diffopt[0, r] = opt.failures_diffopt / iterations
		failure_safeopt = np.maximum(1e-3, iterations - opt.failures_safeopt)
		failure_diffopt = np.maximum(1e-3, iterations - opt.failures_diffopt)
		Overshoot_summary_safeopt[1, r] = opt.failure_overshoot_safeopt / (failure_safeopt)
		Overshoot_summary_diffopt[1, r] = opt.failure_overshoot_diffopt / (failure_diffopt)
		print("Reward SafeOpt")
		print(Reward_data_safeopt[int(iterations/div), r], end=" ")
		print("\n Reward DiffOpt")
		print(Reward_data_diffopt[int(iterations/div), r], end=" ")
		print(r)
	# Save recorded data on rewards and constraint violation
	np.savetxt('SafeOpt_Overshoot.csv', Overshoot_summary_safeopt, delimiter=',')
	np.savetxt('SafeOpt_Reward.csv', Reward_data_safeopt, delimiter=',')
	np.savetxt('DiffOpt_Overshoot.csv', Overshoot_summary_diffopt, delimiter=',')
	np.savetxt('DiffOpt_Reward.csv', Reward_data_diffopt, delimiter=',')

	print("Safety SafeOpt")
	print(opt.failures_safeopt/iterations)
	print("Safety DiffOpt")
	print(opt.failures_diffopt/iterations)
	max_safeopt, f_safeopt = opt.opt.get_maximum()
	max_diffopt, f_diffopt = opt.diff_opt.get_maximum()
	#max=[2.00179108e+00,4.13625539e+00, 3.34599393e+00, 7.41304209e-01,2.81500345e-01, 3.13137132e-03]
	time_recorder = np.asarray(opt.time_safeopt_recorded)
	print("SafeOpt")
	print("Time mean and std:", time_recorder.mean(), time_recorder.std())
	print("Time last iteration:", time_recorder[-1])
	print("maximum", max_safeopt)  
	time_recorder = np.asarray(opt.time_diffopt_recorded)
	print("DiffOpt")
	print("Time mean and std:", time_recorder.mean(), time_recorder.std())
	print("Time last iteration:", time_recorder[-1])
	print("maximum", max_diffopt)    
	plt.show()
	plt.plot(opt.time_safeopt_recorded)
	plt.plot(opt.time_diffopt_recorded)
	tikz_save('time_comparison.tex')
	plt.show()
	# safe = safe_opt()
	# # safe.setup_optimization(num_iter=100)
	# safe.optimization()
	# safe.show()

if __name__ == '__main__':
	# run_optimization()
	sys = System(rollout_limit=0, position_bound=0.5,velocity_bound=7, upper_eigenvalue=-10)
	q = 4/6
	r = -1
	params = np.asarray([q, r])
	f, g1, g2, state = sys.simulate(params, update=True, render=True)

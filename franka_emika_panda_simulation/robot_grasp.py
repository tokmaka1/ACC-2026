import sys
import os
# import franka_real_time
import numpy as np
import time
#error_bound=0.2, initial policy: 0.6,0.6,0.6
class robot_grasp(object):
	'''
	Robot_arm class used to communicate with the robot arm and move in a oval trajectory
	'''
	def __init__(self,IP="192.168.0.1",w=np.pi/5000,sampling_time=1/200,reward_factor=1):
		'''
		Inputs:
		IP: address to connect with the robot
		w:	Freuquency of rotation
		sampling_time: How fast we want to sample

		Objects:
		robot: C++ interface with the robot to communicate with backend
		r: radius of the circle
		ellipse_factor: scaling of the y-direction of the circle to have an elliptic/oval shape
		center: ndarray, center of the circle
		default_impedance: ndarray, impedance recommended by libfranka examples
		safe_impedance: ndarray, initial safe impedance, can be used as a backup policy
		w: frequency of rotation
		sampling_time: Time we would like to receive the next sample within
		time_last_sample: Last time we received a sample from the robot
		num_samples: number of samples we have collected so far.


		Remark: receive(),send(), receive_and_send() are function calls of the C++ interface used to synchronize the back and front end.
		receive() synchronizes the states of backend with the states of front end.
		send() snychronizes the targets e.g. target_positions, impedances etc.
		'''
        # Connect with robot
		self.robot=franka_real_time.Robot(IP)
		self.robot.start()
		self.reset()
        # Define trajectory parameters
		self.r=0.3
		self.ellipse_factor=1.25
		self.state=[]
		self.robot.receive()
		self.center=self.robot.get_position()
		self.center[2]-=self.r
		self.default_impedance = self.robot.get_translation_impedance()
		self.safe_impedance = self.default_impedance.copy()*0.64	
		#self.go_to_start()
		self.w=w
		self.sampling_time = sampling_time
		self.time_last_sample = 0
		self.num_samples=0
		self.frequency_diff=0
		self.reward_factor=reward_factor
		
        # Define wall parameters
		self.width=0.12
		self.wall_height=0.38
		self.scale=self.width/self.wall_height
		self.tol=0.11
		#self.M[2]=0.1
		self.robot.move_to_reference()

	def setup_experiment(self):
		'''
		Moves robot to the desired joint configuration
		'''

		# Give robot some time after experiments to slow down
		for i in range(500):
			self.robot.receive_and_send()
		# Set robot joints to reference position
		self.reset()
		# Go to start position
		#self.go_to_start()
		self.num_samples=0
		self.frequency_diff=0

	def reset(self):
		'''
		Moves robot to the center of the circle around which we move, sets all the default parameters
		'''
		# Move to reference joint positions
		#self.robot.move_to_reference()
		# Reset all controller values, i.e. target_positions, orientations etc.
		self.robot.set_default()
		self.robot.receive()
		self.robot.send()
		self.state=[]
		
	def go_to_start(self):
		'''
		Puts robot on the perimeter of the circle from where we start the experiments
		'''
		# Define reference point to be at the edge of the circle
		reference=self.center.copy()
		Next_target=self.center.copy()
		reference[1]=self.ellipse_factor*self.r+self.center[1]

		# Linearly reach the reference point 
		traj_time=2000
		i=0
		while(i<traj_time):
			# Update target linearly
			Next_target=self.center+i*1/traj_time*(reference-self.center)
			self.robot.set_target_position(Next_target)
			self.robot.receive_and_send()
			if not self.robot.get_late():
				# Only update the iterator if target was updated, makes sure target_position and current position are close-by.
				# Avoids large control actions
				i+=1
		# Give some extra time to the robot to reach and stabilize at the target
		for i in range(int(0.25*traj_time)):
			self.robot.receive_and_send()
	
	def step(self,k):
		'''
		Takes in iteration step k, updates the targets for the robot to track

		Inputs
		k: Current iteration step

		Outputs:
		late: bool, whether the targets could be updated for the robot
		state: ndarray, current position (relative to target) and velocity of the robot. Empty if not sampled
		reward: - norm(error_pos)
		dist: norm(error_pos)

		'''
		state=[]
		reward=0
		dist_constraint=np.inf
		T=time.time()

		# For k==0 (experiment start) sample state and rewards etc.
		if k==0:
			# Update time to last sample
			self.time_last_sample=T
			# Receive robot state information
			self.robot.receive_and_send()
			# Get relative position and velocities
			current_pos=self.robot.get_position()
			pos=current_pos-self.robot.get_target_position()
			vel=self.robot.get_velocity()
			#print(current_pos[2])
			dist=np.linalg.norm(pos)
			dist_constraint=np.maximum(self.scale*np.abs(current_pos[2]-self.wall_height/2),np.abs(current_pos[1]))
			reward=-self.reward_factor*dist
			state=np.vstack((pos.reshape(-1,1),vel.reshape(-1,1)))
			self.state.append(state)
			self.num_samples+=1
			
		# If time since last sample >= sampling_time, sample states
		if (T-self.time_last_sample>=self.sampling_time*0.7):
			self.frequency_diff+=T-self.time_last_sample
			# Update time since last sample
			self.time_last_sample=T
			self.robot.receive_and_send()
			current_pos=self.robot.get_position()
			pos=current_pos-self.robot.get_target_position()
			vel=self.robot.get_velocity()
			#print(pos)
			dist=np.linalg.norm(pos)
			dist_constraint=np.maximum(self.scale*np.abs(current_pos[2]-self.wall_height/2),np.abs(current_pos[1]))
			#print(dist_constraint)
			#print(current_pos[2])
			reward=-self.reward_factor*dist
			state=np.vstack((pos.reshape(-1,1),vel.reshape(-1,1)))
			self.state.append(state)
			self.num_samples+=1

		angle=k*self.w+np.pi/2
		remainder=angle%(2*np.pi)
		Next_target=self.center.copy()
		if remainder>=0 and remainder<np.pi:
			# Update targets for the robot
			Next_target[2]=self.center[2]+self.r*np.sin(k*self.w+np.pi/2)
			Next_target[1]=self.center[1]+self.ellipse_factor*self.r*np.cos(k*self.w+np.pi/2)
		else:
			# Update targets for the robot
			Next_target[2]=self.center[2]-self.r*np.sin(k*self.w+np.pi/2)
			Next_target[1]=self.center[1]+self.ellipse_factor*self.r*np.cos(k*self.w+np.pi/2)
		# Set targets
		self.robot.set_target_position(Next_target)
		self.robot.receive_and_send()
		return self.robot.get_late(),state,reward,dist_constraint
        
        


	

	def change_impedance(self,impedance):
		'''
		Changes impedance of the robot control 
		'''
		
		late=True
		while late:
			self.robot.set_translation_impedance(impedance)
			self.robot.receive_and_send()
			late=self.robot.get_late()

		
		
		




	

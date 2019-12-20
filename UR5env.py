import torch
import gym
import numpy as np
from simulation import vrep
import time
import os

class UR5env(gym.Env):
    
    def __init__(self, sim_client,terminalstep,threshold,_max_episode_steps,objfilepath):
        super(UR5env,self).__init__()
        self.sim_client=sim_client
        self.action_space = gym.spaces.Box(low=-0.3*np.ones(6),high=0.3*np.ones(6),dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-360*np.ones(9),high=360*np.ones(9),dtype=np.float32)
        self.UR5_joint_handle = np.zeros((6,), dtype=np.int)
        self.jointnum = 6
        self.steps = 0
        self.terminalstep = terminalstep
        self.threshold = threshold
        self._max_episode_steps = _max_episode_steps
        self.obj_loc = np.zeros(3)
        self.obj_mesh_dir = os.path.abspath(objfilepath)

    def step(self, action):

        self.movejoint(action)
        observation = self.getjointpos()
        reward = self.getreward()
        done = self.endcheck(self.threshold)

        return observation, reward, done, 0

    def reset(self):
        
        
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        
        time.sleep(1)
        
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        self.add_objects()
        time.sleep(1)
        

        
        observation = self.getjointpos()
        return observation


    def movejoint(self, action):

        for ijoint in range(self.jointnum):
            sim_ret, self.UR5_joint_handle[ijoint] = vrep.simxGetObjectHandle(self.sim_client,'UR5_joint'+str(1+ijoint),vrep.simx_opmode_blocking)
            
        vrep.simxPauseCommunication(self.sim_client, 1)
        for ijoint in range(self.jointnum):
            vrep.simxSetJointTargetVelocity(self.sim_client, self.UR5_joint_handle[ijoint], action[ijoint], vrep.simx_opmode_oneshot)
            vrep.simxSetJointForce(self.sim_client, self.UR5_joint_handle[ijoint], 1000, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.sim_client, 0)
        
    def getjointpos(self):

        for ijoint in range(self.jointnum):
            sim_ret, self.UR5_joint_handle[ijoint] = vrep.simxGetObjectHandle(self.sim_client,'UR5_joint'+str(1+ijoint), vrep.simx_opmode_blocking)
        jointpos = np.zeros(9)
        for ijoint in range(self.jointnum):
            sim_ret, jointpos[ijoint] = vrep.simxGetJointPosition(self.sim_client, self.UR5_joint_handle[ijoint], vrep.simx_opmode_blocking)
        sim_ret, objective_handle = vrep.simxGetObjectHandle(self.sim_client,'Objectpoint', vrep.simx_opmode_blocking)
        sim_ret, self.obj_loc = vrep.simxGetObjectPosition(self.sim_client, objective_handle,-1, vrep.simx_opmode_blocking)
        jointpos[5:8] = self.obj_loc
        return jointpos

    def getreward(self):

        sim_ret, effhandle = vrep.simxGetObjectHandle(self.sim_client,'RG2_centerJoint1', vrep.simx_opmode_blocking)
        sim_ret, effpos = vrep.simxGetObjectPosition(self.sim_client, effhandle,-1, vrep.simx_opmode_blocking)
        reward = -np.log10(100*np.square(self.obj_loc-np.array(effpos)).mean())

        return reward

    def endcheck(self, threshold):
        
        sim_ret, effhandle = vrep.simxGetObjectHandle(self.sim_client,'RG2_centerJoint1', vrep.simx_opmode_blocking)
        sim_ret, effpos = vrep.simxGetObjectPosition(self.sim_client, effhandle,-1, vrep.simx_opmode_blocking)
        distance = np.sqrt(np.square(self.obj_loc-np.array(effpos)).mean())
        self.steps = self.steps+1
        
        if distance < threshold:
            return 1
        else:
            if self.steps >= self.terminalstep:
                self.steps = 0
                return 1
            else:
                return 0

    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        # self.object_handles = []
        mesh_list = os.listdir(self.obj_mesh_dir)
        curr_mesh_file = os.path.join(self.obj_mesh_dir, mesh_list[6])
        curr_shape_name = 'Objectpoint'
        workspace_limits = np.asarray([[-0.8, -0.2], [-0.8, 0.8], [-0.0001, 0.4]])
        drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
        drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
        object_position = [drop_x, drop_y, 0.15]
        object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
        object_color = [89.0, 161.0, 79.0]
        ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
        

if __name__=="__main__":
    vrep.simxFinish(-1) # Just in case, close all opened connections
    sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
    if sim_client == -1:
        print('Failed to connect to simulation (V-REP remote API server). Exiting.')
        exit()
    else:
        print('Connected to simulation.')
    env = UR5env(sim_client,np.array([1,1,1]))


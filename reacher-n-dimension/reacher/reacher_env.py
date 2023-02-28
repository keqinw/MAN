import numpy as np
import random
import matplotlib.pyplot as plt
import copy

class ReacherEnv_v0():
    metadata = {'render.modes': ['human']}  
  
    def __init__(self,n):
        self.n = n # n is the number of dimension
        self.action_space = 3**n # (-1,0,1)
        self.observation_space = 2*n # position of current and target
        self.size = 20 # the size of the world
        self.direction = np.array([-1,0,1])
        self.target = None
        self.origin = None
        self.position_previous = None
        self.done = None
        self.c = None
        self.reset()
        
    def step(self, action):
        
        # decode the action into n-dimension
        direction = []
        for i in range(self.n,0,-1):
            dir, action = np.divmod(action, 3**(i-1))
            direction.append(dir)
        
        action_decode = self.direction[direction]
        
        # move
        self.position_current = action_decode + self.position_previous

        # use the mahhatton distance for convenience
        distance_previous = np.sum(abs(self.target - self.position_previous))
        distance_current = np.sum(abs(self.target - self.position_current))

        self.c += 1

        # get reward
        reward = (distance_previous - distance_current)

        # reach the goal
        if distance_current == 0:
            reward += 100
            self.done = True

        elif self.c == 40:
            self.done = True

        # move out of the maze:
        elif np.amax(self.position_current) > self.size-1 or np.amin(self.position_current) < 0:
            # self.position_current = self.position_previous
            reward -= 10
            self.done = True

        self.position_previous = self.position_current
        self.ob = np.hstack([self.position_current,self.target])
        
        return self.ob, reward, self.done, dict()

    def reset(self):
        pass
        # init the world 
        while 1:
            self.origin = np.random.choice(range(self.size),self.n)
            self.target = np.random.choice(range(self.size),self.n)
            distance = np.sum(abs(self.target - self.origin))
            if distance != 0:
                break

        # init the state
        self.done = False
        self.ob = np.hstack([self.origin,self.target])  # goal-condition
        self.position_previous = self.origin

        # init the count
        self.c = 0

        return self.ob


    def render(self,mode='human'):
        pass
        
        
    def close(self):
        pass

    def seed(self, seed=None): 
        pass

class ReacherEnv_v1():
    metadata = {'render.modes': ['human']}  
  
    def __init__(self,n):
        self.n = n # n is the action number
        self.action_space = [3]*n
        self.observation_space = 2*n
        self.direction = np.array([-1,0,1])
        self.size = 20
        self.target = None
        self.origin = None
        self.position_previous = None
        self.done = None
        self.c = None

        self.reset()
        
    def step(self, action):
        # get the direction   
        action_decode = self.direction[action]

        # move
        self.position_current = action_decode + self.position_previous
        # self.position_current = self.length[length_index] * np.array(self.direction[direction_index]) + self.position_previous

        # use the mahhatton distance for convenience
        distance_previous = np.sum(abs(self.target - self.position_previous))
        distance_current = np.sum(abs(self.target - self.position_current))

        self.c += 1

        # get reward
        reward = (distance_previous - distance_current)

        # reach the goal
        if distance_current == 0:
            reward += 100
            self.done = True

        elif self.c == 40:
            self.done = True

        # move out of the maze:
        elif np.amax(self.position_current) > self.size-1 or np.amin(self.position_current) < 0:
            reward = -10
            self.done = True

        self.position_previous = self.position_current
        self.ob = np.hstack([self.position_current,self.target])
        
        return self.ob, reward, self.done, dict()

    def reset(self):
        while 1:
            self.origin = np.random.choice(range(self.size),self.n)
            self.target = np.random.choice(range(self.size),self.n)
            distance = np.sum(abs(self.target - self.origin))
            if distance != 0:
                break

        # init the state
        self.done = False
        self.ob = np.hstack([self.origin,self.target])  # goal-condition
        self.position_previous = self.origin

        # init the count
        self.c = 0

        return self.ob

    def render(self,mode='human'):
        pass
        
    def close(self):
        pass
    
    def seed(self, seed=None): 
        pass

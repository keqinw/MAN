from re import T
import re
import pybullet as p
import os
import numpy as np
import matplotlib.pyplot as plt

height = [0.025,0.025,0.05,0.075]

class Cube:
    def __init__(self,client,index, position, orientation, mode):
        self.index = index
        self.client = client
        if mode == 'grasp':
            self.position = np.insert(position,2,height[index-1])
        if mode == 'stack':
            # self.position = np.insert(position,2,1)
            self.position = position
        self.orientation = [0,0, orientation]

        f_name = os.path.join(os.path.dirname(__file__),'cube',)
        if mode == 'grasp':
            self.cube = p.loadURDF(fileName = f_name + str(index) + '.urdf',
                                basePosition = self.position,
                                baseOrientation = p.getQuaternionFromEuler(self.orientation),
                                # useFixedBase = True,
                                flags = p.URDF_USE_SELF_COLLISION,
                                physicsClientId = client)
        if mode == 'stack':
            self.cube = p.loadURDF(fileName = f_name + str(index) + '.urdf',
                                basePosition = self.position,
                                baseOrientation = p.getQuaternionFromEuler(self.orientation),
                                # useFixedBase = True,
                                flags = p.URDF_USE_SELF_COLLISION,
                                useFixedBase=True,
                                physicsClientId = client)


    def get_posture(self):
        return np.hstack((self.position,self.orientation[-1]))

    def get_id(self):
        return self.cube

    def get_type(self):
        return self.index

    def get_height(self):
        return height[self.index-1]


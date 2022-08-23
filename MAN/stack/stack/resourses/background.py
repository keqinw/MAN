import pybullet as p
import os
import numpy as np

class Background:
    def __init__(self,client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__),'background.urdf')
        position = [0,1,0.48]
        orientation = [np.pi/2,0,0]

        self.robot = p.loadURDF(fileName = f_name,
                              basePosition = position,
                              baseOrientation = p.getQuaternionFromEuler(orientation),
                              useFixedBase= True,
                              physicsClientId = client)

import pybullet as p
import os

class Goal:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'goal.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], base[2]],
                   baseOrientation = p.getQuaternionFromEuler([0, 0, base[3]]),
                   useFixedBase = True,
                   physicsClientId=client)



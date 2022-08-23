import pybullet as p
import os


class Plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleplane.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[0, 0, -0.005],
                   useFixedBase= True,
                   flags = p.URDF_USE_SELF_COLLISION,
                   physicsClientId=client
                   )



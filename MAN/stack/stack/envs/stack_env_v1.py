import gym
import numpy as np
import pybullet as p
import pybullet_data
from stack.resourses.cube import Cube
from stack.resourses.background import Background
from stack.resourses.plane import Plane
from stack.resourses.goal import Goal
import matplotlib.pyplot as plt
import copy

height = [0.025,0.025,0.05,0.075]
# indices = [2,4,4,4,2,3,3,1,1,1,1]
cube_list = np.array([[0, 0.05, 0],
                      [0.05, 0.05, 0.05],
                      [0.1, 0.1, 0.1],
                      [0.15, 0.15, 0.15]])

class StackEnv_v1(gym.Env):
    def __init__(self):
        # set the observation & action space (grasp)

        # set the observation & action space (stack)
        self.action_space = gym.spaces.Discrete(14)
        self.observation_space = gym.spaces.box.Box(
            low=np.zeros(14,dtype = np.float32),high=np.ones(14,dtype = np.float32))

        # self.client = p.connect(p.GUI)
        self.client = p.connect(p.DIRECT)
        p.setTimeStep(1/30, self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        self.rendered_img_grasp = None
        self.rendered_img_stack = None
        self.num_cube = None
        self.cube_ids = None
        self.done = False # determind if the process end
        self.heights = None
        self.type_cube = [i for i in range(len(height))]
        self.cube_pool = None
        
        self.reset()

        # set cameras
        self.viewMatrix_stack = p.computeViewMatrix(cameraEyePosition=[0, 0, 0.3],
                                        cameraTargetPosition=[0, 1, 0.3],
                                        cameraUpVector=[0, 0, 1])
        self.projectionMatrix_stack = p.computeProjectionMatrixFOV(fov=50,
                                                            aspect=1.0,
                                                            nearVal=0.1,
                                                            farVal=100)

        self.viewMatrix_grasp = p.computeViewMatrix(cameraEyePosition=[0, 0, 1],
                                        cameraTargetPosition=[0, 0, 0],
                                        cameraUpVector=[0, 1, 0])
        self.projectionMatrix_grasp = p.computeProjectionMatrixFOV(fov=70,
                                                            aspect=1.0,
                                                            nearVal=0.1,
                                                            farVal=100)

    def step(self,action):
        ######################## stack action #####################

        # update the remianing cubes

        # type = self.cube[index].get_type()
        index,action = action
        self.cube_pool.remove(index)
        # print(self.cube_pool)
        type = index + 1
        

        previous_height = copy.copy(self.heights)
        # action is discreate
        # each colunm's width is 0.06
        position = (action+1) * 0.056 - 0.392 - 0.028
        # height
        if type == 1:
            if self.heights[action+1] == 0:
                L = 0
            else:
                L = 1

            # small cube never form holes
            O = 0
            # here O shows if cube fills a hole
            # if self.heights[action] > self.heights[action+1]  and self.heights[action+2] > self.heights[action+1]:
                # O -= 0.5

            self.heights[action+1] += height[type-1]
            Cube(self.client,type,[position,0.975,self.heights[action+1]],np.pi/2,'stack')
            self.heights[action+1] += height[type-1]
                   

        else:
            if np.sum(self.heights[action:action+3]) == 0:
                L = 0
                
            else: 
                L = 1
            self.heights[action:action+3] += height[type-1]
            # calculate the hole
            O = len(np.nonzero(np.max(self.heights[action:action+3])-self.heights[action:action+3])[0]) * 0.7
            self.heights[action:action+3] = np.max(self.heights[action:action+3])
            Cube(self.client,type,[position,0.975,self.heights[action+1]],np.pi/2,'stack')
            self.heights[action:action+3] += height[type-1]

        self.count += 1

        ob = self.heights[1:-1]

        # calculate the cube remain
        self.num_cube -= 1
        # if no cube remain, then done
        if self.num_cube == 0:
            self.done = True
        else:
            self.done = False
        # if height is too large, out of the frame then done
        if max(height) >= 0.85:
            self.done = True 

        # calculate the reward
        # here is the important one, which should be modified future
        if round(max(self.heights) - max(previous_height),3) < height[type-1]:
            H = 0
        else:
            H = 1

        # mean = np.mean(self.heights)
        # previous_mean = np.mean(previous_height)

        virance = 0
        for i in range(len(self.heights)-1):
            virance += abs(self.heights[i+1] - self.heights[i])

        previous_virance = 0
        for i in range(len(self.heights)-1):
            previous_virance += abs(previous_height[i+1] - previous_height[i])
            
        # virance = np.mean(abs(np.array(self.heights)-mean))
        virance = round(virance,5)
        previous_virance = round(previous_virance,5)
        # previous_virance = np.mean(abs(np.array(previous_height)-previous_mean))
        # print(previous_virance)
        # print(virance)
        # if virance == 0 :
            # B = 1
        if previous_virance > virance:
            B = 1
        elif  previous_virance == virance:
            B = 0
        else:
            B = -0.3

        # The stack should be encourage to fill the blank, ranther than stack upward
        # if there a hole is formed, then get negtive reward O
        
        reward = B - H - L - O # need to be modified


        # print(B,H,L,O)
        return ob, reward, self.done, dict()

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # load plane
        # planeId = Plane(self.client)
        # planeId = p.loadURDF("plane.urdf")

        # load background
        # backgroundId = Background(self.client)

        # load cubes
        self.num_cube = np.random.choice([7,8,9,10])
        # self.num_cube = 7
        self.count = 0
        self.heights = np.zeros((16))

        self.cube_pool = list(np.random.randint(0,4,self.num_cube))

        return (self.heights[1:-1])

    def reset_compare(self,cube_pool, num_cube):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # load plane
        # planeId = Plane(self.client)
        # planeId = p.loadURDF("plane.urdf")

        # load background
        # backgroundId = Background(self.client)

        # load cubes
        self.num_cube = num_cube
        # self.num_cube = 7
        self.count = 0
        self.heights = np.zeros((16))

        self.cube_pool = cube_pool

        return (self.heights[1:-1])

    def render(self, mode='human', close=False):

        # mode: human: show the image, but not return image
        #       rgb_array: not show the image, just return image
        frame_stack = None

        # get the satck image
        if self.rendered_img_stack is None:
            self.rendered_img_stack = np.zeros((224, 224, 4))

        _, _, rgb_stack, _, _ = p.getCameraImage(width=224, height=224,
                                              viewMatrix=self.viewMatrix_stack,
                                              projectionMatrix=self.projectionMatrix_stack)

        rgb_stack = np.reshape(rgb_stack, (224, 224, 4))
        frame_stack = rgb_stack[:,:,:3]

        # show the frames
        if mode == 'human':
            plt.imshow(frame_stack)
            # plt.pause(0.1)
            plt.show()

        return frame_stack

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None): 
        pass
if __name__ == '__main__':
    a = StackEnv_v1()
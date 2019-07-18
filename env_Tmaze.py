import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import cv2

class EnvTMaze(object):
    def __init__(self, len, if_up):
        self.len = len
        self.occupancy = np.zeros((5, len+2))
        for i in range(5):
            self.occupancy[i, 0] = 1
            self.occupancy[i, len+1] = 1
        for i in range(len+2):
            self.occupancy[0, i] = 1
            self.occupancy[4, i] = 1
        for i in range(len):
            self.occupancy[1, i] = 1
            self.occupancy[3, i] = 1
        self.if_up = if_up
        self.agt_pos = [2, 1]
        if if_up == 1:
            self.dest_pos = [1, len]
            self.wrong_pos = [3, len]
        else:
            self.dest_pos = [3, len]
            self.wrong_pos = [1, len]

    def reset(self, if_up):
        self.if_up = if_up
        self.agt_pos = [2, 1]
        if if_up == 1:
            self.dest_pos = [1, self.len]
            self.wrong_pos = [3, self.len]
        else:
            self.dest_pos = [3, self.len]
            self.wrong_pos = [1, self.len]

    def step(self, action):
        reward = 0
        if action == 0:     # up
            if self.occupancy[self.agt_pos[0] - 1][self.agt_pos[1]] != 1:  # if can move
                self.agt_pos[0] = self.agt_pos[0] - 1
        if action == 1:     # down
            if self.occupancy[self.agt_pos[0] + 1][self.agt_pos[1]] != 1:  # if can move
                self.agt_pos[0] = self.agt_pos[0] + 1
        if action == 2:     # left
            if self.occupancy[self.agt_pos[0]][self.agt_pos[1] - 1] != 1:  # if can move
                self.agt_pos[1] = self.agt_pos[1] - 1
        if action == 3:     # right
            if self.occupancy[self.agt_pos[0]][self.agt_pos[1] + 1] != 1:  # if can move
                self.agt_pos[1] = self.agt_pos[1] + 1
        if self.agt_pos == self.dest_pos:
            reward = 10
        if self.agt_pos == self.wrong_pos:
            reward = -10
        return reward

    def get_obs(self):
        obs = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                img_x = self.agt_pos[0] - 1 + i
                img_y = self.agt_pos[1] - 1 + j
                if img_x >= 0 and img_x < 5 and img_y >= 0 and img_y < 8:
                    if self.occupancy[img_x, img_y] == 0:
                        obs[i, j, 0] = 1.0
                        obs[i, j, 1] = 1.0
                        obs[i, j, 2] = 1.0
        if self.agt_pos == [2, 1]:
            if self.if_up==1:
                obs[1, 1, 0] = 0.0
                obs[1, 1, 1] = 1.0
                obs[1, 1, 2] = 0.0
            else:
                obs[1, 1, 0] = 0.0
                obs[1, 1, 1] = 0.0
                obs[1, 1, 2] = 1.0
        return obs

    def get_global_obs(self):
        obs = np.zeros((5, 6, 3))
        for i in range(5):
            for j in range(6):
                if self.occupancy[i, j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
        if self.if_up == 1:
            obs[2, 1, 0] = 0.0
            obs[2, 1, 1] = 1.0
            obs[2, 1, 2] = 0.0
        else:
            obs[2, 1, 0] = 0.0
            obs[2, 1, 1] = 0.0
            obs[2, 1, 2] = 1.0
        obs[self.agt_pos[0], self.agt_pos[1], 0] = 1.0
        obs[self.agt_pos[0], self.agt_pos[1], 1] = 0.0
        obs[self.agt_pos[0], self.agt_pos[1], 2] = 0.0
        return obs

    def render(self):
        obs = self.get_global_obs()
        enlarge = 10
        new_obs = np.ones((5*enlarge, (self.len + 2)*enlarge, 3))
        for i in range(5):
            for j in range(self.len + 2):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge),
                                  (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge),
                                  (0, 255, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge),
                                  (255, 0, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(10)

    def plot_scene(self):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[2, 2])
        ax1.imshow(self.get_global_obs())
        ax2.imshow(self.get_obs())
        plt.show()

if __name__ == '__main__':
    env = EnvTMaze(4, random.randint(0, 1))
    max_iter = 100000
    for i in range(max_iter):
        print("iter= ", i)
        env.plot_scene()
        # env.render()
        print('agent at', env.agt_pos, 'dest', env.dest_pos, 'wrong', env.wrong_pos)
        action = random.randint(0, 3)
        reward = env.step(action)
        print('reward', reward)
        if reward != 0:
            print('reset')
            env.reset(random.randint(0, 1))

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:29:55 2020

@author: 付说举于版筑之间
"""
"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example. The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 3  # grid width
car_centre_dict={}

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u1', 'd1', 'l1', 'r1']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        car_centre_dict[1] = np.array([20, 20])                   
        # create car1
        self.car1 = self.canvas.create_oval(
            car_centre_dict[1][0] - 15, car_centre_dict[1][1] - 15,
            car_centre_dict[1][0] + 15, car_centre_dict[1][1] + 15,
            fill='yellow')
        self.text1 = self.canvas.create_text(
                        car_centre_dict[1][0], car_centre_dict[1][1],text = '1',
                         fill='black', font=('Times', 15))
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.car1)
        self.canvas.delete(self.text1)
        origin = np.array([20, 20])
        self.car1 = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='yellow')
        self.text1 = self.canvas.create_text(
                        origin[0], origin[1],text = '1',
                         fill='black', font=('Times', 15))
        # return observation
        return self.canvas.coords(self.car1)

    def step(self, action):
        
        s = self.canvas.coords(self.car1)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.car1, base_action[0], base_action[1])  # move agent
        self.canvas.move(self.text1, base_action[0], base_action[1])
        s1_ = self.canvas.coords(self.car1)  # next state get 4 coords
        # to ease the case, try to get the center
        # reward function
        if s1_ == [5, 125, 35, 155]:#x0 y0 x1 y1
            reward = 1
            done = True
            s1_ = 'terminal'
        else:
            reward = 0
            done = False

        return s1_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:59:57 2020

@author: 付说举于版筑之间
"""

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

org_str='000364105020'
aim_str='000123456000'

car_centre_dict={}
car_display_loc={}
for i in range(len(org_str)): 
    if org_str[i]!='0':
        car_centre_dict[org_str[i]]=[20+i%3*40,20+i//3*40]
        car_display_loc[org_str[i]]=[20+i%3*40-15,20+i//3*40-15,20+i%3*40+15,20+i//3*40+15]

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u1', 'd1', 'l1', 'r1','u2', 'd2', 'l2', 'r2']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
    def str2arr(self):
        pass
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
        car_centre_dict['1'] = np.array([20, 20])  
        car_centre_dict['2'] = np.array([60, 20])                 
        # create car1
        self.car1 = self.canvas.create_oval(
            car_centre_dict['1'][0] - 15, car_centre_dict['1'][1] - 15,
            car_centre_dict['1'][0] + 15, car_centre_dict['1'][1] + 15,
            fill='yellow')
        self.text1 = self.canvas.create_text(
                        car_centre_dict['1'][0], car_centre_dict['1'][1],text = '1',
                         fill='black', font=('Times', 15))
         # create car2
        self.car2 = self.canvas.create_oval(
            car_centre_dict['2'][0] - 15, car_centre_dict['2'][1] - 15,
            car_centre_dict['2'][0] + 15, car_centre_dict['2'][1] + 15,
            fill='yellow')
        self.text2 = self.canvas.create_text(
                        car_centre_dict['2'][0], car_centre_dict['2'][1],text = '2',
                         fill='black', font=('Times', 15))
        
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.car1)
        self.canvas.delete(self.text1)
        self.canvas.delete(self.car2)
        self.canvas.delete(self.text2)
        car1_origin = np.array([20, 20])
        car2_origin = np.array([60, 20])
        self.car1 = self.canvas.create_oval(
            car1_origin[0] - 15, car1_origin[1] - 15,
            car1_origin[0] + 15, car1_origin[1] + 15,
            fill='yellow')
        self.text1 = self.canvas.create_text(
                        car1_origin[0], car1_origin[1],text = '1',
                         fill='black', font=('Times', 15))
        self.car2 = self.canvas.create_oval(
            car2_origin[0] - 15, car2_origin[1] - 15,
            car2_origin[0] + 15, car2_origin[1] + 15,
            fill='yellow')
        self.text2 = self.canvas.create_text(
                        car2_origin[0], car2_origin[1],text = '2',
                         fill='black', font=('Times', 15))
        
        
        
        # return observation
        return [self.canvas.coords(self.car1),self.canvas.coords(self.car2)]

    def step(self, action):
        
        s = [self.canvas.coords(self.car1),self.canvas.coords(self.car2)]
        base_action = np.array([0, 0])
        if action == 0:   # 1up
            if s[0][1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 1down
            if s[0][1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 1right
            if s[0][0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # 1left
            if s[0][0] > UNIT:
                base_action[0] -= UNIT
        elif action == 4:   # 2up
            if s[1][1] > UNIT:
                base_action[1] -= UNIT
        elif action == 5:   # 2down
            if s[1][1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 6:   # 2right
            if s[1][0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 7:   # 2left
            if s[1][0] > UNIT:
                base_action[0] -= UNIT
        # 怎样实现move两个车
        if action <4:
            self.canvas.move(self.car1, base_action[0], base_action[1])  # move agent(car1)
            self.canvas.move(self.text1, base_action[0], base_action[1])
        else:
            self.canvas.move(self.car2, base_action[0], base_action[1])  # move agent(car2)
            self.canvas.move(self.text2, base_action[0], base_action[1])
        s_ = [self.canvas.coords(self.car1),self.canvas.coords(self.car2)]# next state get 4 coords
        # to ease the case, try to get the center
        # reward function
        if s_ == [[5, 45, 35, 75],[45, 45, 75, 75]]:#x0 y0 x1 y1
            reward = 1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

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

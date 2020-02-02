"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
from maze_env_v0 import Maze
from RL_brain_v1 import QLearningTable
import matplotlib.pyplot as plt
import numpy as np
import pickle

def update():
    cost_his=[]
    obs_time=0
    for episode in range(10):
        # initial observation
        observation = env.reset()
        obs_time += 1
        print("第"+str(obs_time)+"次")
        acion_step=0 # before chosing action, step is 0.flash everytime before while loop
        # while loop is for one episode
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            acion_step +=1 # count actions
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            
            # break while loop when end of this episode
            if done:
                cost_his.append(reward)
                
                if reward ==1:
                    print("用了"+str(acion_step)+"步成功到达目标状态")
                else:
                    print("用了"+str(acion_step)+"步未能成功到达目标状态")
                break

    # end of game
    print('game over')
    RL.store()
    
    
    env.destroy()
    plt.plot(np.arange(len(cost_his)), cost_his)
    plt.ylabel('Reward')
    plt.xlabel('training steps')
    plt.show()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    with open('training_data.pickle', 'rb') as file:
        train_his =pickle.load(file)
        RL.q_table=train_his
    print(RL.q_table)   
    env.after(100, update)
    env.mainloop()
    
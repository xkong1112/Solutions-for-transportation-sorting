# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:47:17 2020
PPO2 ALGORITHM FOR TRANSPORTATION SORTING
@author: Xiangyu Kong
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mainEnv import Maze

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001 #LR应该是learn rate
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10 
C_UPDATE_STEPS = 10 # 可能是每十个ep更新一次ac
# 莫烦源代码里面写的是AC更新频率是一样的，不知道具体为啥
S_DIM, A_DIM = 1, 1 
# action就是随机一个随机数的样子-2到2，立钟摆，所以a-dim是1，s-dim是3，为啥呢？
# 字符串的维度应该是1，state和action都是一个12位的字符串，np.array([str]).ndim=1所以维度都是1,这里面的中括号不能省略。
epsilon = 0.2
class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        # critic # self.v 是啥？v应该就是value，对应就是梯度，但在这个里面就是对应的神经网络
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
        # actor 从正态分布里面找一个动作
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False) # 记录参数，更新参数，没这个网络跑不起来
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):                     # 在每一次更新完actor和critic的同时更新old-pi
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')  # 初始一个tensor action 来传递action
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage') # 初始一个tensor 来记录 advantage
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'): # 定义一个surrogate用来计算surr objective，也就是那个期望，也就是clip的前一项
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
           
              # clipping method, find this is better
            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-epsilon, 1.+epsilon)*self.tfadv))
                    # 加一个负号，minimize就变成了maximize

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    # 没有返回值，返回了一个tensor, 输入值是序列t所有的action，state和折减后的rewards
    def update(self,    s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    # 把环境的动作合集输入tensorflow中，怎么选？       
    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable) # why *2?
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable) # sigma
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params
    
    # choose action要返回一个字符串！，get_v是啥，value？？？
    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        #[np.newaxis, :]的作用就是给原来的数组加一个[]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
# env改成输入一个字符串，主要是输入数据和输出数据的类型要替换一下，主体结构和框架不变

ppo = PPO()
all_ep_r=[]
step = []#用一个列表来记录每次episode实现的步数，最小化这个步数
#循环一千次
for ep in range(EP_MAX):
    s = Maze.reset()
    buffer_s, buffer_a, buffer_r ,buffer_step = [], [], [],[]
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        # Maze.render() render需要用显卡，但其实没必要
        a = ppo.choose_action(s)
        s_, r, done, step = Maze.step(a) # 从step传来的就只有三个数
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        buffer_step.append(step)
        s = s_
        ep_r += r # r是reward的总数，是否必要？是，没reward怎么训练？

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
# 用vstack和[:, np.newaxis]把原来的数组变成了列向量
# 因为discount rate append的是一个数，而其他的append的是ndarray，ndarray本身就带[],所以vstack来合并
            bs, ba, br ,bstep= np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br) # 把bs, ba, br传入update中
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
# 出一个图
plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.plot(np.arange(len(all_ep_r)), min_step)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

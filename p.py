# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:13:37 2020
@author: 98212
"""
"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from main_env import Maze

EP_MAX = 1000 # 1000个ep
EP_LEN = 200 #?一个episode200步
GAMMA = 0.9
A_LR = 0.0001 #LR应该是learn rate
C_LR = 0.0002
BATCH = 32 # 每32步更新一次
A_UPDATE_STEPS = 1 #?
# 多少步之内可以收敛，目前知道的是20步，目标就是20步以内找到合理的答案，原来的钟摆是10
C_UPDATE_STEPS = 1 #?
# 莫烦源代码里面写的是AC更新频率是一样的，不知道具体为啥
S_DIM, A_DIM = 1, 1 # action就是随机一个随机数的样子-2到2，立钟摆，所以a-dim是1，s-dim是3，为啥呢？字符串的维度应该是1，state和action都是一个字符串，所以维度都是1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization
# [0] 代表kl散度 [1] 代表clip

class PPO(object):

    def __init__(self):         
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        # self.v 是啥？v应该就是value，对应就是梯度，但在这个里面就是对应的神经网络
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1) # 前一个l1是input，1是units也就是神经元的个数（有时候也叫维数）,最后一层输出多少个神经元就代表最终有多少个类别输出啊，然后再做softmax
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r') # ？tf.placeholder(dtype,shape,name)其中[None, 3]表示列是3，行不定,此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor 从正态分布里面找一个动作
            # 从build a_net(一个正态分布的网络)
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
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
            if METHOD['name'] == 'kl_pen': # method 是一个字典
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
                    # 加一个负号，minimize就变成了maximize # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
                    # 这里面就是把ratio限制在1-epsilon和1+epsilon之间，然后乘以advantage
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    # 返回的是atrainop和ctrainop，其中op应该是operation
    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)] # 在python语言中，返回的tensor是numpy ndarray对象。
        # 后面的那个dict就是feeddict，也就是输入，前面的是输出，self.atrain_op
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        # 后面的那个dict就是feeddict，也就是输入，前面的是输出，self.ctrain_op
        # 我们都知道feed_dict的作用是给使用placeholder创建出来的tensor赋值
    # 应该没得问题，字符串应该也可以训练
    # Tensor就是一个n维数组，0维数组表示一个数(scalar)，1维数组表示一个向量(vector)，二维数字表示一个矩阵(matrix)
    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params
    
    # choose action要返回一个字符串！，get_v是啥，value？？？
    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2) # 也就是说clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
    #用clip防止溢出？
    def get_v(self, s):# 输入的s不是str,list,int,应该必须是np array，字符串要输入tf需要转化成为np array
        if s.ndim < 2: s = s[np.newaxis, :] # [np.newaxis, :]转变成一个行向量，[：，np.newaxis]转变成一个列向量
        # ndim就是Number of array dimensions.
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    # 输入是state,输出是value
    
# env改成输入一个字符串，主要是输入数据和输出数据的类型要替换一下，主体结构和框架不变

ppo = PPO()
all_ep_r = []
#循环一千次
for ep in range(EP_MAX):
    s = Maze.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        # Maze.render() render需要用显卡，但其实没必要
        a = ppo.choose_action(s)
        s_, r, done, _ = Maze.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:# 每一个batch更新一次PPO，也就是32次更新一次, %取模，返回除法的余数，一个ep有300也就更新7次，然后外面再循环1000，也就是更新了7000次
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )
# 出一个图
plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

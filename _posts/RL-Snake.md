---
layout: post
title: Deep Reinforcement Learning: DQN explained with Snake game.
---

!["Snake Img"](../images/snake.gif?style=centerme) 



The core idea of reinforcement learning is to use rewards in a way that the AI agent can learn how to perform well by maximizing it's expected rewards. To solve an RL problem, the AI agent forms a policy that represents what action to take at all the possible states of the environment.
Classic RL approaches were limited in solving high-dimensional problems since they mostly relied on hand-crafted linear features in order to represent this policy. However, in recent years, the representation power of deep neural networks have been used in the RL problems.
An RL problem consists of one or more agents interacting with an environment. These interactions are usually considered to be episodic. So at each time step, the agents choose an action from its action space. The agent makes this decision based on the policy the agent follows and its current state. After taking action, the agents transits to a new state and receives a reward. The goal of an RL problem is to find an optimal policy that maximizes the expected overall reward. To solve an RL problem, we almost always need to calculate a value function, which approximates the value of a state or taking a specific action at that state.
TD learning is a family of algorithms that, like the Monte Carlo method, do not require the transition model of the environment. Like dynamic programming, these algorithms do not have to wait for an experience to terminate in order to update value functions. Q-learning is an off-policy variant of TD learning that follows the following update rule in order to have a good approximation of the state-action values (Q):


<img src="https://tex.s2cms.ru/svg/Q(S_t%2CA_t)%20%5Cleftarrow%20Q(S_t%2CA_t)%20%2B%5Calpha%5BR_%7Bt%2B1%7D%20%2B%5Cgamma%20%5Cmax_%7Ba%7D%20Q(S_%7Bt%2B1%7D%2Ca)%20-%20Q(S_t%2CA_t)%5D." alt="Q(S_t,A_t) \leftarrow Q(S_t,A_t) +\alpha[R_{t+1} +\gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)]." />




Deep Q-learning uses deep neural networks to approximate Q-value function from raw data. This neural network is called Deep Q-Network (DQN).

DQN is trained by minimizing the following loss for a batch of transitions:


<img src="https://tex.s2cms.ru/svg/%5Cmathcal%7BL%7D%20%3D%20%5Cfrac%7B1%7D%7B%7CB%7C%7D%5Csum_%7B(s%2C%20a%2C%20s'%2C%20r)%20%5C%20%5Cin%20%5C%20B%7D%20%5Cmathcal%7BL%7D(%5Cdelta)" alt="\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)" />

where delta is the temporal defrence error, defined as:

<img src="https://tex.s2cms.ru/svg/%5Cdelta%20%3D%20Q(s%2C%20a)%20-%20(r%20%2B%20%5Cgamma%20%5Cmax_a%20Q(s'%2C%20a))%20" alt="\delta = Q(s, a) - (r + \gamma \max_a Q(s', a)) " />

Snake-RL is a python environment to train RL agents for the game of Snake! It comes with the implementation of DQN agent.

Creating your agent is as easy as:
```python
from RL_Snake import BaseAgent
import random

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
NA = 'None'
ACTIONS = {UP:0,DOWN:1,LEFT:2,RIGHT:3,NA:4}


class RandomAgent(BaseAgent):
    """
    Concrete agent class that take actions randomly
    """
    def take_action(self):
        """
        take actions randomly
        Returns:
            int or None: represent the action
        """
        state = self.get_state()
        if state is None:
            return ACTIONS[NA]
        return random.sample(ACTIONS.values(),1)
        
        
```
The state is two consecutive settings of the board stacked together.
The provided DQN agent uses CNN to process the state and outputs the state-action values.
```python
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
        """
        """
        def __init__(self, h, w, outputs):
            """
            Args:
                h: height of the board
                w: width of the board
                outputs: number of actions
            """
            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(32)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=3, stride=1):
                return (size - (kernel_size - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            """
            Args:
                x: Tensor representation of input states
            Returns:
                list of int: representing the Q values of each state-action pair
            """
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))
```
you can take a closer look at the [code](https://github.com/mkhoshpa/Snake-RL) if you're intrested.

References: 

[ Wonderful pytorch blog post about DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

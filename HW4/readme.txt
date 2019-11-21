https://github.com/mnarasim/ML7641/tree/master/HW4

1) There are three files that contain all the models and algorithms:

Reinforcement.py: this file includes Value iteration, Policy iteration and Q-learning.
To run first Open AI gym needs to be installed. This would work with Frozen Lake and Taxi.

For Forest Management pymdtoolbox needs to be installed. However, I modified the mdp.py that's installed as part of pymdtoolbox. So once pymdtoolbox installed please replace that mdp.py with what's in my GitHub.

2) Custom.py: This just has the custom MDP.

In Custom.py the Q-learning function uses the following parameters: 
P, R, discount, learning rate, epsilon, min epsilon, max epsilon, decay rate
mdptoolbox.mdp.QLearning(P, R, discount,0.3,0.5,0.01,0.3,-0.00005)

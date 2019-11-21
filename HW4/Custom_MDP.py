#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:03:14 2019

@author: Mani
"""

import numpy as np
import random as rand 
from matplotlib import pyplot as plt	  
import random
from timeit import default_timer as timer

class mdp(object):
    
    def __init__(self, states=10,actions=2,iterations=100,discount=0.9):
        
        self.states = states
        self.actions = actions
        self.iterations = iterations
        self.discount = discount
        
     
        self.sMatrix=np.empty((states,actions))
        for i in range(self.states):
            for j in range(self.actions):
                
                self.sMatrix[i][j] = j * (j/2) * (j**2)
                
        
        self.sMatrix[states-1,actions-1] = 1000
        self.sMatrix[0,0] = -1
    
  
        
      
        self.p = {s : {a : [np.random.random_sample(), np.random.randint(0,self.states), 0,'_'] for a in range(self.actions)} for s in range(self.states)}
        s = {}
        for x in range(self.states):
            a = {}
            for y in range(self.actions):
                t1 = []
                for z in range(3):
                    t1.append(np.random.random_sample())
           
                for z in range(3):
                    t1[z] = t1[z]/np.sum(t1)
                t1[-1] = 1.0 - np.sum(t1[0:-1])
               
                temp=[]
                for z in range(3):
                    if x > 0 and x < states-1:
                        t = np.random.choice([1,2,3], p=[0.4,0.3,0.3])
                        if t==1:
                            nextState = x
                        elif t==2:
                            nextState = x - 1
                        else:
                            nextState = x + 1
                        
                    elif x ==0:
                        t = np.random.choice([1,2], p=[0.5,0.5])
                        if t==1:
                            
                            nextState = x
                        else:
                            nextState = x+1
                    else:
                        t = np.random.choice([1,2], p=[0.5,0.5])
                        if t==1:
                            
                            nextState = x
                        else:
                            nextState = x-1
                    if x == nextState:
                        re=0
                    elif x<nextState:
                        re=0
                    else:
                        re=1
                    temp.append((t1[z], nextState, self.sMatrix[x][y],'_'))
            
                a[y] = temp
      
            s[x] = a
        self.p = s



    def run_episode(self,policy, gamma = 0.9, render = False):
 

        total_reward = 0
        step_idx = 0

     
        for i in range(int(self.states)):
            a = np.argmax(policy[i])
            reward = self.sMatrix[i][a]
            
            total_reward += (self.discount ** step_idx * reward)
            step_idx += 1
        
        return total_reward
            
    def evaluate_policy(self,policy, gamma = 0.9):
        n=1000
        scores = [self.run_episode(policy, gamma, False) for _ in range(n)]
        return np.mean(scores)
                
    def extract_policy(self, v):
 
        policy = np.zeros(self.states)
        for s in range(self.states):
            q_sa = np.zeros(self.actions)
            for a in range(self.actions):
                for prob,next_state,reward,done in self.p[s][a]:
               
            
                    q_sa[a] += (prob * (reward + self.discount * v[next_state]))
            policy[s] = np.argmax(q_sa)
        return policy
        
    


    def value_iteration(self):
        
        eps = 1e-10
        self.v = np.zeros(self.states)
    
        decision = True
        iters = 0
        while decision == True:
            self.prev_v = np.copy(self.v)
            iters = iters + 1
        
     
            for s in range(self.states):
           
            
                self.q = np.zeros(self.actions)
                for a in range(self.actions):
      
          
                    for i in range(len(self.p[s][a])):
                        prob = self.p[s][a][i][0]
                        next_state = self.p[s][a][i][1]
                        reward = self.p[s][a][i][2]
                        don = self.p[s][a][i][3]
                       
                        self.q[a] = self.q[a] + prob * (reward + self.discount * self.v[next_state])
                
                self.v[s] = max(self.q)
           
            if (np.sum(np.fabs(self.prev_v - self.v)) <= eps):
                print('Value iteration converged at ', iters)
                decision = False
            if (iters == self.iterations):
                print('No convergence - value iteration, max iteration reached')
                decision = False
        return self.v
    
    

    def compute_policy(self, policy):

        self.v = np.zeros(self.states)
        eps = 1e-10
        decision = True
        iters = 0
   
        while decision:
            iters = iters + 1
            self.prev_v = np.copy(self.v)
            delta =0
            for s in range(self.states):
                v_old = 0.0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state,reward,done in self.p[s][a]:
                        v_old = action_prob*prob * (reward + self.discount * self.prev_v[next_state])
                
                delta = max(delta, np.abs(v_old - self.v[s]))
               
                self.v[s] = v_old
               
            if (delta <= eps):
                print ('Policy evaluated in', iters)
                decision = False
            if (iters == self.iterations):
                print('No convergence - policy evaluation, max iteration reached')
                decision = False
        
        return self.v
    
    def policy_iteration(self):

        self.policy = np.ones([self.states, self.actions])/self.actions

        
        iters = 0
        decision = True
        while decision == True:
            iters = iters + 1
            stable_policy = True
            self.eval_policy = self.compute_policy(self.policy)
            
  
            for s in range(self.states):
                current_action = np.argmax(self.policy[s])
           
            
                policy_stable = True
                q = np.zeros(self.actions)
                for a in range(self.actions):
    
          
                    for prob,next_state,reward,done in self.p[s][a]:
                      
                        q[a] = q[a] + prob * (reward + self.discount * self.eval_policy[next_state])
                        
        
                best_action = np.argmax(q)
                
                if current_action != best_action:
                    policy_stable = False
                self.policy[s] = np.eye(self.actions)[best_action]
            
            if policy_stable == True:
                print('Policy iteration converged in ', iters)
                decision = False
                return self.policy     
  
           
            
    def QLearner(self,num_states, num_actions):
    
        total_episodes = 150 
        learning_rate = 1.0         
        max_steps = 2000     
        gamma = 0.95                 
        epsilon = 0.5                 
        max_epsilon = 2.0            
        min_epsilon = 0.01            
        decay_rate = -0.00005
        qtable = np.zeros([num_states,num_actions])	  
        rewards = []
        for episode in range(total_episodes):
            state = 0
            step = 0
            done = False
            total_rewards = 0
    
            for step in range(max_steps):
                exp_exp_tradeoff = random.uniform(0, 1)
        
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(qtable[state,:])

                else:
                    action = rand.randint(0, num_actions-1) 

            
                if action == 0:
                    new_state = state
                elif action == 1 and state>0:
                    new_state = state - 1
                elif action == 1 and state==0:
                    new_state = state
                elif action==2 and state<num_states-1:
                    new_state = state + 1
                else:
                    new_state = state
                reward = self.sMatrix[state][action] 
                if state == num_states - 1:
                    done = True

                qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
                total_rewards += reward
                state = new_state
        
                if done == True: 
                    break
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
            rewards.append(total_rewards)
        solution_policy = np.argmax(qtable,axis=1)
        print ("Score over time: " +  str(sum(rewards)/total_episodes))
        plt.title('Q Learner - OPtimal policy ')
        plt.plot(solution_policy)
        plt.xlabel('States')
        plt.ylabel('Actions')
        plt.figure()
    
        for episode in range(5):
            state = 0
            step = 0
            done = False
     
            for step in range(max_steps):
                action = np.argmax(qtable[state,:])
             
                if action == 0:
                    new_state = state
                elif action == 1 and state>0:
                    new_state = state - 1
                elif action == 1 and state==0:
                    new_state = state
                elif action==2 and state<num_states-1:
                    new_state = state + 1
                else:
                    new_state = state
                reward = self.sMatrix[state][action] 
             
                if state == num_states - 1:
                    done = True
        
                if done:
                    print("Number of steps", step)
                    break
                state = new_state
                
np.random.seed(1000)         

if __name__ == '__main__':
    max_iterations = 1000
    num_states= 400
    num_actions=3
    gamma = 0.9
    agent = mdp(num_states,num_actions,max_iterations,gamma)
    start_time = timer()
    optimal_v = agent.value_iteration()
    end_time = timer() - start_time
    print('Value time', end_time)
    policy = agent.extract_policy(optimal_v)
    scores = agent.evaluate_policy(policy, gamma)
    print('Value iteration - Average reward ', np.mean(scores))
    
    start_time = timer()
    optimal_p = agent.policy_iteration()
    end_time = timer() - start_time
    print('Policy Time ', end_time)
    scores = agent.evaluate_policy(optimal_p, gamma)
    print('Policy iteration - Average reward ', np.mean(scores))
    start_time = timer()
    agent.QLearner(num_states,num_actions)
    q_time = timer() - start_time
    print('Q Learner time ', q_time)
   
    #################################
    plt.title('Value iteration - Optimal Policy')
    plt.plot(policy)
    plt.xlabel('States')
    plt.ylabel('Actions')
    plt.figure()
    plt.title('Policy iteration - Optimal Policy')
    plt.plot(optimal_p)
    plt.xlabel('States')
    plt.ylabel('Actions')
    plt.figure()
    
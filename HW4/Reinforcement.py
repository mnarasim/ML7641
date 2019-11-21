#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:30:04 2019

@author: Mani
"""

import gym    
import numpy as np
import mdptoolbox.example
 
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import random

np.random.seed(10)  
def value_iteration(env, discount = 1.0, fname=0):
    Names = ['Frozen Lake','Taxi']
    max_iteration = 10000
    eps = 1e-10
    v = np.zeros(env.nS)
    
    decision = True
    iters = 0
    max_val = []
    while decision == True:
        prev_v = np.copy(v)
        iters = iters + 1
        
     
        for s in range(env.nS):
           
            
            q = np.zeros(env.nA)
            for a in range(env.nA):
               
                for prob, next_state, reward, done in env.P[s][a]:
                    q[a] = q[a] + prob * (reward + discount * v[next_state])
            v[s] = max(q)
        max_val.append(np.mean(v))    
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print('Converged at ', iters)
            plt.title('Value iteration - average value per iteration ' + Names[i])
            plt.plot(max_val)
            plt.xlabel('Iterations')
            plt.ylabel('Average value')
            plt.figure()
            decision = False
        if (iters == max_iteration):
            print('No convergence, max iteration reached')
            decision = False
            
    return v
            
def run_episode(env, policy, gamma = 1.0, render = False):
 
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward
            
def evaluate_policy(env, policy, gamma = 1.0, n = 1000):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, discount =1.0,fname=0):
    Names = ['Frozen Lake','Taxi']
    v = np.zeros(env.nS)
    eps = 1e-10
    decision = True
    iters = 0
    max_iteration = 10000
    max_val =[]
    while decision:
        iters = iters + 1
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
            
            
        max_val.append(np.mean(v))    
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            decision = False
            plt.title('Policy iteration - average value per iteration ' + Names[i])
            plt.plot(max_val)
            plt.xlabel('Iterations')
            plt.ylabel('Average values')
      
           
            print('Policy evaluated in ', iters)
        if (iters == max_iteration):
            print('No convergence, max iteration reached')
            decision = False
        
    return v

def policy_iteration(env, discount = 1.0,fname=0):

    policy = np.random.choice(env.nA, size=(env.nS))  
    max_iteration = 10000
    decision = True
    iters = 0
    
    while decision == True:
        iters = iters + 1
        old_policy_v = compute_policy_v(env, policy, discount)
        new_policy = extract_policy(old_policy_v, discount)
        if (np.all(policy == new_policy)):
            print ('No. of policies evaluated', iters)
            plt.figure()
            decision = False
        if (iters == max_iteration):
            print('No convergence, max iteration reached')
            decision = False
        policy = new_policy
       
    return policy     

def QLearner(env,num_states, num_actions,i):
    Names = ['Frozen Lake','Taxi']
    total_episodes = 1500  
    learning_rate = 1.0          
    max_steps = 1000            
    gamma = 0.90                 
    epsilon = 0.5               
    max_epsilon = 0.3            
    min_epsilon = 0.01            
    decay_rate = -0.00005
    qtable = np.zeros([num_states,num_actions])	  
    rewards = []
    max_val =[]
    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
    
        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1)
        
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])

            else:
                action = env.action_space.sample()
        
            new_state, reward, done, info = env.step(action)

            qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
            total_rewards += reward
            state = new_state
        
            if done == True: 
                break
        max_val.append(np.mean(qtable))
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
        rewards.append(total_rewards)
    

    print ("Q-Learning score over time: " +  str(sum(rewards)/total_episodes))
 
    solution_policy = np.argmax(qtable,axis=1)
    
    plt.plot(solution_policy)
    plt.title('Q Learner Optimal Policy - ' + Names[i])
    plt.xlabel('States')
    plt.ylabel('Actions')
    plt.figure()
    plt.title('Q Learner - average Q values ' + Names[i])
    
    plt.plot(max_val)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Q values')
    plt.figure()
    
    for episode in range(5):
            state = env.reset()
            step = 0
            done = False
    
            for step in range(max_steps):
                action = np.argmax(qtable[state,:])
        
                new_state, reward, done, info = env.step(action)
        
                if done:
                    print("Number of steps", step)
                   
                    break
                state = new_state
    print('-----------------------------------------------')
    
def mdp_forest():
     nStates = 1000
     r1=10
     r2 = 100
     p1 = 0.5
     discount = 0.90
     max_episodes = 10
     rewards = []
     max_val =[]
     P, R = mdptoolbox.example.forest(nStates,r1,r2,p1)
     vi = mdptoolbox.mdp.ValueIteration(P, R, discount,1e-10)
     vi.run()
     pi= mdptoolbox.mdp.PolicyIteration(P, R,discount,eval_type=1)
     pi.run()
     
     for i in range(max_episodes):
         qlearner = mdptoolbox.mdp.QLearning(P, R, discount,0.3,0.5,0.01,0.3,-0.00005)
         start_time = timer()
         qlearner.run()
         q_forest_time = timer() - start_time
         rewards.append(qlearner.rewards)
         max_val.append(np.mean(qlearner.Q))
     plt.title('Forest - Value Iteration optimal policy')
     plt.plot(vi.policy,label='Value Iteration Policy')
     plt.xlabel('States')
     plt.ylabel('Actions')
     plt.figure()
  
     plt.title('Forest, Value iteration, average values')
     plt.xlabel('Iterations')
     plt.ylabel('Average values')
     plt.plot(vi.mean_val)
     plt.figure()
     plt.title('Forest - Policy Iteration')
     plt.plot(pi.policy, label ='Policy Iteration Policy')
     plt.xlabel('States')
     plt.ylabel('Actions')
     plt.figure()
     plt.title('Forest, Policy iteration, average values')
     plt.xlabel('Iterations')
     plt.ylabel('Average values')
     plt.plot(pi.mean_val)
     plt.figure()
     
     plt.title('Forest - Q Learning')
     plt.plot(qlearner.policy, label='Q Learner')
     plt.xlabel('States')
     plt.ylabel('Actions')
     plt.figure()
     plt.title('Q Learner, Forest Management - average Q values')
     plt.plot(max_val)
     plt.xlabel('Episodes')
     plt.ylabel('Average Q values')
     

     print(vi.iter, ' Value iterations, ', np.round(vi.time,3), 'VI time')
     print(pi.iter, ' Policy iterations, ', np.around(pi.time,3), 'PI time')
     print(np.mean(vi.V), 'Value iteration average score')
     print(np.mean(pi.V), 'Policy iteration average score')
     print(np.sum(rewards)/max_episodes, 'Q Learner rewards')
     print(q_forest_time, 'Q Learner time - forest')
    
if __name__ == '__main__':
    scenarios = ['FrozenLake8x8-v0','Taxi-v3']
    Names = ['Frozen Lake','Taxi']
    
    for i in range(len(scenarios)):
        
        env = gym.make(scenarios[i])
       
        gamma = 0.9
        start_time = timer()
        optimal_v = value_iteration(env, gamma, i)
        value_time = timer() - start_time
        
        extract_v_policy = extract_policy(optimal_v,gamma)
        scores = evaluate_policy(env, extract_v_policy, gamma)
      
        
        print('Value iteration - Average reward for ' + str(Names[i]) +' ', np.round(np.mean(scores),3))
        print('Value iteration - time for ' + str(Names[i]) + ' ', np.round(value_time,3))
        #
        #
        print('---------------------------------------------------------')
        start_time = timer()
        optimal_policy = policy_iteration(env, gamma,i)
        policy_time = timer() - start_time
        scores = evaluate_policy(env, optimal_policy, gamma)
        print('Policy iteration - Average reward for ' + str(Names[i]) + ' ', np.round(np.mean(scores),3))
        print('Policy iteration - time for ' + str(Names[i]) + ' ', np.round(policy_time,3))
       
        print('----------------------------------------------------------')
       
     
        plt.title('Value Iteration - Optimal policy ' + str(Names[i]))
        plt.plot(extract_v_policy, label='Policy from value iteration')
        plt.xlabel('States')
        plt.ylabel('Actions')
        plt.figure()
        plt.title('Policy Iteration - Optimal policy '+ str(Names[i]))
        plt.plot(optimal_policy, label = 'Policy from policy iteration')
        plt.xlabel('States')
        plt.ylabel('Actions')
        plt.figure()
 
        start_time = timer()
        QLearner(env,env.nS, env.nA,i)
        q_time = timer() - start_time
        print('Q time ' + Names[i], q_time)
        
        
    mdp_forest()
  
       
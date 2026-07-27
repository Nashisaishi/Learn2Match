import math

import numpy as np
def stable_matching(agent,arm,u_agent,u_arm,N,K):
    #print(agent)
    pull = np.zeros(N, int) - 1
    optimal = np.zeros(N, int)
    estimation = np.zeros((N, K), int)
    match=np.zeros((N),int)
    for i in agent:
        estimation[i, :] = u_agent[i, :].argsort()[::-1]
        optimal[i] = 0
    #print(estimation)
    for t in range(N*N):
        for i in agent:
            pull[i]=estimation[i,optimal[i]]
        for i in agent:
            if collision(N,K,agent,arm,pull,u_arm)[i]==1:
                optimal[i]=optimal[i]+1
    for i in range(N):
        match[i]=estimation[i,optimal[i]]
    return match
def GS(agent,arm,u_agent,u_arm,ue_arm,N_arm,N,K,reward,t,regret,stable_matching):
    pull = np.zeros(N, int) - 1
    optimal = np.zeros(N, int)
    estimation = np.zeros((N, K), int)
    match = np.zeros((N), int)
    for i in agent:
        estimation[i, :] = u_agent[i, :].argsort()[::-1]
        optimal[i] = 0
    # print(estimation)
    for t_0 in range(N * N):
        for i in agent:
            pull[i] = estimation[i, optimal[i]]
        for i in agent:
            if collision(N, K, agent, arm, pull, ue_arm)[i] == 1:
                optimal[i] = optimal[i] + 1
        ue_arm,N_arm,reward,t,regret=update_arm(K,agent,arm,pull,u_agent,u_arm,ue_arm,N_arm,reward,t,regret,stable_matching)
    for i in range(N):
        match[i] = estimation[i, optimal[i]]
    return match,ue_arm,N_arm,reward,t,regret
def collision(N,K,agent,arm,pull,ue_arm):
    C=np.zeros(N)
    match = np.zeros(K, int) - 1
    for k in arm:
        MAX = -1
        for j in agent:
            if pull[j] == k:
                if ue_arm[k][j] > MAX:
                    MAX = ue_arm[k][j]
                    match[k] = j
    for j in agent:
        if j==match[pull[j]]:
            C[j]=0
        else:
            C[j]=1
    return C
def update(K,agent,arm,pull,u_agent,ue_agent,N_agent,u_arm,ue_arm,N_arm,reward,t,regret,stable_matching):
    match = np.zeros(K, int) - 1
    for k in arm:
        MAX = -1
        for j in agent:
            if pull[j] == k:
                if ue_arm[k][j] > MAX:
                    MAX = ue_arm[k][j]
                    match[k] = j
    for k in arm:
        if match[k] != -1:
            ue_arm[k][match[k]] = (ue_arm[k][match[k]] * N_arm[k][match[k]] + Bernoulli(u_arm[k][match[k]])) / (
                        N_arm[k][match[k]] + 1)
            N_arm[k][match[k]] = N_arm[k][match[k]] + 1
    for k in arm:
        if match[k] != -1:
            ue_agent[match[k]][k]=(ue_agent[match[k]][k]*N_agent[match[k]][k]+Bernoulli(u_agent[match[k]][k]))/(N_agent[match[k]][k]+1)
            N_agent[match[k]][k]=N_agent[match[k]][k]+1
            reward[match[k]] = reward[match[k]] + u_agent[match[k]][k]
    t=t+1
    if (t)%(1000)==0:
        for j in agent:
            regret[j,int((t)/(1000))]=t*u_agent[j,stable_matching[j]]-reward[j]
    return ue_agent,N_agent,ue_arm,N_arm,reward,t,regret
def Bernoulli(p):
    x = np.random.uniform(0,1)
    if x < p:
        return 1
    else:
        return 0
def update_arm(K,agent,arm,pull,u_agent,u_arm,ue_arm,N_arm,reward,t,regret,stable_matching):
    match=np.zeros(K,int)-1
    for k in arm:
        MAX=-1
        for j in agent:
            if pull[j]==k:
                if ue_arm[k][j]>MAX:
                    MAX=ue_arm[k][j]
                    match[k]=j
    for k in arm:
        if match[k]!=-1:
            ue_arm[k][match[k]]=(ue_arm[k][match[k]]*N_arm[k][match[k]]+Bernoulli(u_arm[k][match[k]]))/(N_arm[k][match[k]]+1)
            N_arm[k][match[k]]=N_arm[k][match[k]]+1
    for k in arm:
        if match[k]!=-1:
            reward[match[k]]=reward[match[k]]+u_agent[match[k]][k]
    t=t+1
    if (t) % (1000) == 0:
        for j in agent:
            regret[j, int((t) / (1000))] = t * u_agent[j, stable_matching[j]] - reward[j]
    return ue_arm,N_arm,reward,t,regret

def update_UCB(K, agent, arm, pull, u_agent, ue_agent, N_agent,  c,t_win, u_arm, ue_arm, N_arm, reward, t, regret, stable_matching):
    match = np.zeros(K, int) - 1
    N=len(agent)
    pull_agent=np.zeros((K,N),int)
    ucb=np.zeros((K,N))+10000000000
    t_0=math.log(5000000,2)
    for k in arm:
        for j in agent:
            if N_arm[k][j]!=0:
                ucb[k][j]=ue_arm[k][j] + (3 * t_0 / (2 * N_arm[k][j])) ** (1 / 2)
    for k in arm:
        MAX = -1
        for j in agent:
            if pull[j] == k:
                pull_agent[k][j]=1
                if ucb[k][j] > MAX:
                    MAX = ucb[k][j]
                    match[k] = j
    for k in arm:
        if match[k] != -1:
            ue_arm[k][match[k]] = (ue_arm[k][match[k]] * N_arm[k][match[k]] + Bernoulli(u_arm[k][match[k]])) / (
                    N_arm[k][match[k]] + 1)
            N_arm[k][match[k]] = N_arm[k][match[k]] + 1
    for k in arm:
        if match[k] != -1:
            ue_agent[match[k]][k] = (ue_agent[match[k]][k] * N_agent[match[k]][k] + Bernoulli(u_agent[match[k]][k])) / (
                        N_agent[match[k]][k] + 1)
            N_agent[match[k]][k] = N_agent[match[k]][k] + 1
            reward[match[k]] = reward[match[k]] + u_agent[match[k]][k]
    for j in agent:
        for j_1 in agent:
            if pull[j_1] == pull[j]:
                k=pull[j]
                if j!=j_1:
                    c[j][j_1][k] = c[j][j_1][k] + 1
                    if match[k] == j:
                        t_win[j][j_1][k] = t_win[j][j_1][k] + 1
    t = t + 1
    if (t) % (1000) == 0:
        for j in agent:
            regret[j, int((t) / (1000))] = t * u_agent[j, stable_matching[j]] - reward[j]
    return ue_agent, N_agent, ue_arm, N_arm, c, t_win, reward, t, regret,match
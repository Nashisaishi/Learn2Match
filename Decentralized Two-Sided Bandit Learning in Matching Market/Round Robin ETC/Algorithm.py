import math
import numpy as np
import Pre
def index_assignment(N,K,agent,arm,u_agent,u_arm,ue_arm,N_arm,reward,t_0,regret,stable_matching):
    pull=np.zeros(N,int)
    pai=np.zeros(N)
    index=np.zeros(N,int)
    for t in range(N):
        for j in agent:
            pull[j] = pai[j]
        collision=Pre.collision(N,K,agent, arm, pull, ue_arm)
        for j in agent:
            if collision[j] == 0:
                if pai[j] == 0:
                    index[j] = t
                    pai[j]=pai[j]+1
        ue_arm,N_arm,reward,t_0,regret=Pre.update_arm(K,agent,arm,pull,u_agent,u_arm,ue_arm,N_arm,reward,t_0,regret,stable_matching)
    return index,ue_arm,N_arm,reward,t_0,regret
def exploration(D,K,N,index,agent,arm,u_agent,ue_agent,N_agent,u_arm,ue_arm,N_arm,reward,t,T,regret,stable_matching):
    t_0=math.ceil(math.log(T,2))
    K_1=len(arm)
    pull=np.zeros(N,int)
    success=np.zeros(N,int)+1
    arm=np.array(list(arm))
    for t_1 in range(K_1*(K**2)*t_0):
        for j in agent:
            k=(index[j]+t)%K_1
            pull[j]=arm[k]
        ue_agent, N_agent, ue_arm, N_arm, reward, t,regret=Pre.update(K,agent,arm,pull,u_agent,ue_agent,N_agent,u_arm,ue_arm,N_arm,reward,t,regret,stable_matching)
    for j in agent:
        for k_1 in arm:
            for k_2 in arm:
                if k_1 != k_2:
                    if ue_agent[j][k_1] +(2)* (  t_0 / (N_agent[j][k_1])) ** (1 / 2) > ue_agent[j][k_2] - (2)*(
                            t_0 / (N_agent[j][k_2])) ** (1 / 2):
                        if ue_agent[j][k_1] - (2)*( t_0 / (N_agent[j][k_1])) ** (1 / 2) < ue_agent[j][k_2] + (2)*(
                                 t_0 / (N_agent[j][k_2])) ** (1 / 2):
                            success[j] = 0
    return success,ue_agent,N_agent,ue_arm,N_arm,reward,t,regret
def COMM(index,agent,arm,success,u_agent,u_arm,ue_arm,N_arm,N,K,reward,t,regret,stable_matching):
    K_1=len(arm)
    N_1=len(agent)
    pull=np.zeros(N,int)
    arm = np.array(list(arm))
    if N_1==1:
        return success,ue_arm,N_arm,t,regret
    for j_1 in range(N_1):
        for j in agent:
            if index[j]==j_1:
                transmitter=j
        for j_2 in range(N_1):
            if j_1!=j_2:
                for j in agent:
                    if index[j] == j_2:
                        receiver = j
                for k in range(K_1):
                    for j in agent:
                        pull[j]=arm[(k+1)%K_1]
                        if j==transmitter:
                            if success[j]==0:
                                pull[j]=arm[k]
                        if j==receiver:
                            pull[j]=arm[k]
                    ue_arm, N_arm, reward, t, regret = Pre.update_arm(K, agent, arm, pull, u_agent, u_arm, ue_arm,
                                                                        N_arm, reward, t, regret, stable_matching)
                    for j in agent:
                        if j==receiver:
                            if Pre.collision(N,K,agent,arm,pull,ue_arm)[j]==1:
                                success[j]=0
    return success,ue_arm,N_arm,t,regret
def update(index,agent,arm,success,u_agent,u_arm,ue_arm,N_arm,N,K,N_1,arm_1,reward,t,regret,stable_matching):
    pull=np.zeros(N,int)
    K_1=len(arm_1)
    arm_2 = list()
    arm_1=np.array(list(arm_1))
    for a in arm_1:
        if a not in arm_2:
            arm_2.append(a)
    for n in range (N_1):
        j_0=-1
        for j in agent:
            if index[j]==n:
                j_0=j #j_0's turn to explore
        N_2 = N_1
        #print(arm_2)
        for m in range(K_1):
            for j in agent:
                pull[j]=arm_1[(m+1)%K_1]
                if j==j_0:
                    pull[j]=arm_1[m]
            ue_arm, N_arm, reward, t, regret = Pre.update_arm(K, agent, arm, pull, u_agent, u_arm, ue_arm,
                                                              N_arm, reward, t, regret, stable_matching)
            for j in agent:
                if j==j_0:
                    if Pre.collision(N,K,agent,arm,pull,ue_arm)[j]==1:
                        N_2=N_2-1
                        if arm_1[m] in arm_2:
                            arm_2.remove(arm_1[m])
                        #print(1)
        if j_0 in agent:
            if N_2 != len(agent):
                print("fail")
            #print(N_2, arm, arm_2)
    index,ue_arm,N_arm,reward,t,regret=index_assignment(N,K,agent,arm,u_agent,u_arm,ue_arm,N_arm,reward,t,regret,stable_matching)
    return index,ue_arm,N_arm,reward,t,regret
def Algorithm(agent,arm,u_agent,u_arm,T,D):
    N=len(agent)
    K=len(arm)
    dot=int((T)/1000)+1
    regret=np.zeros((N,dot))
    ue_arm=np.zeros((K,N))
    ue_agent=np.zeros((N,K))
    N_agent=np.zeros((N,K),int)
    N_arm=np.zeros((K,N),int)
    stable_matching=Pre.stable_matching(agent,arm,u_agent,u_arm,N,K)
    reward=np.zeros(N)
    t_1=0
    index,ue_arm,N_arm,reward,t,regret=index_assignment(N,K,agent,arm,u_agent,u_arm,ue_arm,N_arm,reward,0,regret,stable_matching)
    success, ue_agent, N_agent, ue_arm, N_arm, reward, t, regret = exploration(D, K, N, index, agent, arm, u_agent,
                                                                               ue_agent, N_agent, u_arm, ue_arm, N_arm,
                                                                               reward, t, T, regret,
                                                                               stable_matching)
    t_1 = t_1 + 1
    success, ue_arm, N_arm, t, regret = COMM(index, agent, arm, success, u_agent, u_arm, ue_arm, N_arm, N, K, reward, t,
                                             regret, stable_matching)
    opt,ue_arm,N_arm,reward,t,regret=Pre.GS(agent,arm,u_agent,u_arm,ue_arm,N_arm,N,K,reward,t,regret,stable_matching)
    N_1 = len(agent)
    arm_1 = list()
    for a in arm:
        if a not in arm_1:
            arm_1.append(a)
    for j in range(N):
        if success[j] == 1:
            if j in agent:
                agent.remove(j)
                arm.remove(opt[j])
                if opt[j] != stable_matching[j]:
                    print("fail1")
    while(len(agent)>0):
        index, ue_arm, N_arm, reward, t, regret = update(index, agent, arm, success, u_agent, u_arm, ue_arm, N_arm, N,
                                                         K, N_1, arm_1, reward, t, regret, stable_matching)
        success,ue_agent,N_agent,ue_arm,N_arm,reward,t,regret=exploration(D,K, N, index, agent, arm, u_agent, ue_agent, N_agent, u_arm, ue_arm, N_arm, reward, t, T, regret,
                    stable_matching)
        t_1=t_1+1
        success, ue_arm, N_arm, t, regret=COMM(index,agent,arm,success,u_agent,u_arm,ue_arm,N_arm,N,K,reward,t,regret,stable_matching)
        opt,ue_arm,N_arm,reward,t,regret=Pre.GS(agent,arm,u_agent,u_arm,ue_arm,N_arm,N,K,reward,t,regret,stable_matching)
        N_1=len(agent)
        arm_1=list()
        for a in arm:
            if a not in arm_1:
                arm_1.append(a)
        for j in range(N):
            if success[j]==1:
                if j in agent:
                    if opt[j] != stable_matching[j]:
                        print("fail1")
                    x = int(t / 1000)
                    for i in range(x,dot):
                        regret[j,i]=t*u_agent[j][opt[j]]-reward[j]
                    agent.remove(j)
                    arm.remove(opt[j])
    regret_0=0
    for j in range(N):
        regret_0=regret_0+regret[j]
    return regret_0

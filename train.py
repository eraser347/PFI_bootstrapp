#!/user/wk2389/.conda/envs/SMDP/bin/python
import numpy as np
import scipy


## For quick update of Vinv
def sherman_morrison(X, V, w=1):
    result = V-(w*np.einsum('ij,j,k,kl -> il', V, X, X, V))/(1.+w*np.einsum('i,ij,j ->', X, V, X))
    return result

def mbeta(t, d, L, p, delta, lam, S): #Theoretical bounds are too conservative
    beta = (S*np.sqrt(lam)+np.sqrt(2*d*np.log((5*L)/delta))/p)/10
    return(beta)

def m(y1,y2):
    dist = min(y2-y1)
    return(max(0,dist))

def M(y1,y2,eps):
    dist = max(y1+eps-y2)
    return(max(0,dist))

'''
PFIwR
'''
class PFIwR:
    def __init__(self, contexts, L, hypers, seed=0):
        ## Hyperparameters in kwargs
        for x,y in hypers.items():
            setattr(self, x, y)

        ##Initialization
        self.seed = seed
        self.K = len(contexts)
        self.L = L
        self.d = len(contexts[0]) #real data Y is not linear to X
        self.contexts = contexts #np.eye(self.K) #np.array(contexts)
        self.t = 1
        self.A = [k for k in range(self.K)]
        self.P = []


        self.lam = 1+np.log(self.d*L/self.delta) #Theoretical Bound are too conservative
        self.Vinv = np.eye(self.d)/self.lam
        self.DR_A = self.lam*np.eye(self.d)
        self.theta_hat = np.zeros((self.d,L))
        # Exploration set = [K]
        X = np.array(self.contexts)
        self.v = np.min(np.linalg.eigvalsh(X.T @ X))/self.K

        #For imputation estimators
        self.Ainv = np.eye(self.d)/(self.p*self.lam)
        self.xy = np.zeros((self.d,L))
        self.ridge_xy = np.zeros((self.d,L))
        self.exp_X_indexes = []
        self.exp_Y = []

    def select(self):
        #print('t={}, E_t={}'.format(self.t, (max(32*np.sqrt(3), 8/(1-self.p))/self.v)*np.log(self.d*self.t**2/self.delta)/64))
        self.exploration = len(self.exp_Y) < (max(32*np.sqrt(3), 8/(1-self.p))/self.v) * np.log(self.d*self.t**2/self.delta)/10
        if self.exploration: #theoretical bounds are too conservative.
            self.a_t = np.random.choice(self.K)
        else:
            y_hat = np.array(self.contexts) @ self.theta_hat
            pareto_index = [k for k in self.A]
            for i in self.A:
                for j in self.A:
                    if np.max(y_hat[i] - y_hat[j]) < 0:
                        pareto_index.remove(i)
                        break
            self.a_t = np.random.choice(pareto_index)

        # update Xs
        self.t = self.t + 1
        self.DR_A += np.outer(self.contexts[self.a_t],self.contexts[self.a_t])/self.p
        for k in range(self.K):
            self.Vinv = sherman_morrison(self.contexts[k], self.Vinv)
        #self.Vinv = np.eye(self.K)/(self.t+self.lam)
        return(self.a_t)

    def update(self, reward):
        self.ridge_xy += np.outer(self.contexts[self.a_t], reward)/self.p
        #Data mixup
        if not self.exploration:
            perturb_index = np.random.choice(len(self.exp_X_indexes))
            while self.exp_X_indexes[perturb_index] == self.a_t:
                perturb_index = np.random.choice(len(self.exp_X_indexes))
            perturb_weight1 = np.random.uniform(-1,1)*3**(1/2)
            perturb_weight2 = np.random.uniform(-1,1)*3**(1/2)
            X_a_t = perturb_weight1*np.array(self.contexts)[self.a_t]+perturb_weight2*np.array(self.contexts)[self.exp_X_indexes[perturb_index]]
            Y = perturb_weight1*reward + perturb_weight2*self.exp_Y[perturb_index]
        else:
            X_a_t = self.contexts[self.a_t]
            Y = reward
            self.exp_X_indexes.append(self.a_t)
            self.exp_Y.append(Y)

        #Imputation estimator
        self.xy += np.outer(X_a_t,Y)
        self.Ainv = sherman_morrison(X_a_t, self.Ainv)
        impute = self.Ainv @ self.xy

        #DR estimator
        self.theta_hat = impute + self.Vinv @ (self.ridge_xy - self.DR_A @ impute)

        # Updating sets
        y_hat = np.array(self.contexts) @ self.theta_hat
        normalized_norms = {str(i):np.sqrt(np.dot(self.contexts[i],self.Vinv @ self.contexts[i])) for i in self.A}

        b = mbeta(self.t, self.d, self.L, self.p, self.delta, self.lam, np.linalg.norm(self.theta_hat)/self.L)
        #print(max([norms for norms in normalized_norms.values()])*b)
        # Update A and P
        C = [k for k in self.A]
        for i in C:
            for j in C:
                #if len(self.A) <=3:
                #    print(y_hat[self.A])
                #print('{} > {}'.format(m(y_hat[i,:],y_hat[j,:]), (normalized_norms[str(i)]+normalized_norms[str(j)])*b))
                if m(y_hat[i,:],y_hat[j,:]) > (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    self.A.remove(i)
                    break
        P = [k for k in self.A]
        for i in self.A:
            for j in self.A:
                if j != i and M(y_hat[i,:],y_hat[j,:],self.epsilon) <= (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    P.remove(i)
                    break
        P1 = P
        for j in P1:
            for i in list(set(self.A)-set(P1)):
                if M(y_hat[i,:],y_hat[j,:],self.epsilon) <= (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    P.remove(j)
                    break
        self.P = list(set(self.P) | set(P))
        self.A = list(set(self.A) - set(P))
        return()



'''
MultiPFI (Auer et al., 2016)
'''
class MultiPFI:
    def __init__(self, contexts, L, hypers, seed=0):
        ## Hyperparameters in kwargs
        for x,y in hypers.items():
            setattr(self, x, y)

        ##Initialization
        self.seed = seed
        self.t=0
        self.P = []
        self.K = len(contexts)
        self.L = L
        self.A = [k for k in range(self.K)]
        self.arm = 0
        self.estimate = False
        self.Srewards = np.zeros((self.K,L))
        self.SSrewards = np.zeros((self.K,L))
        self.N_t = np.ones(self.K)

    def select(self):
        self.a_t = self.A[self.arm]
        self.arm += 1
        self.t += 1
        return(self.a_t)

    def update(self, reward):
        # Update estimators
        self.N_t[self.a_t] += 1
        self.Srewards[self.a_t,:] += reward
        self.SSrewards[self.a_t,:] += reward**2
        if self.arm == len(self.A):
            self.arm = 0
            y_hat = self.Srewards / (np.tile(self.N_t.T, (self.L, 1))).T
            sigma_hat = self.SSrewards / (np.tile(self.N_t.T, (self.L, 1))).T - y_hat ** 2 + np.outer(np.sqrt(4*np.log(self.L*self.K*self.N_t/self.delta)/self.N_t) , np.array(self.V))
            b = np.array([np.sqrt(2 * sigma_hat[i,:] * np.log(self.L*self.K*self.N_t[i]/self.delta) / self.N_t[i]) for i in range(self.K)])
            # Update A and P
            C = [k for k in self.A]
            for i in C:
                for j in C:
                    if m(y_hat[i,:],y_hat[j,:]) > np.sqrt(max(b[i]**2 + b[j]**2)):
                        self.A.remove(i)
                        break
            P = [k for k in self.A]
            #self.P = list(set(self.P) | set(self.A))
            for i in self.A:
                for j in self.A:
                    if j != i and M(y_hat[i,:],y_hat[j,:],self.epsilon) <= np.sqrt(max(b[i]**2 + b[j]**2)):
                        P.remove(i)
                        break
            P1 = P
            for j in P1:
                for i in list(set(self.A)-set(P1)):
                    if M(y_hat[i,:],y_hat[j,:],self.epsilon) <= np.sqrt(max(b[i]**2 + b[j]**2)):
                        P.remove(j)
                        break
            self.P = list(set(self.P) | set(P))
            self.A = list(set(self.A) - set(P))
        return()


'''
PFImix
'''
class PFImix:
    def __init__(self, contexts, L, hypers, seed=0):
        ## Hyperparameters in kwargs
        for x,y in hypers.items():
            setattr(self, x, y)

        ##Initialization
        self.seed = seed
        self.K = len(contexts)
        self.L = L
        self.d = len(contexts[0]) #real data Y is not linear to X
        self.contexts = contexts #np.eye(self.K) #np.array(contexts)
        self.t = 1
        self.A = [k for k in range(self.K)]
        self.P = []


        self.lam = 1+np.log(self.d*L/self.delta) #Theoretical Bound are too conservative
        self.Vinv = np.eye(self.d)/self.lam
        self.DR_A = self.lam*np.eye(self.d)
        self.theta_hat = np.zeros((self.d,L))
        # Exploration set = [K]
        X = np.array(self.contexts)
        self.v = np.min(np.linalg.eigvalsh(X.T @ X))/self.K

        #For imputation estimators
        self.Ainv = np.eye(self.d)/(self.p*self.lam)
        self.xy = np.zeros((self.d,L))
        self.ridge_xy = np.zeros((self.d,L))
        self.exp_X_indexes = []
        self.exp_Y = []

    def select(self):
        #print('t={}, E_t={}'.format(self.t, (max(32*np.sqrt(3), 8/(1-self.p))/self.v)*np.log(self.d*self.t**2/self.delta)/64))
        self.exploration = len(self.exp_Y) < (max(32*np.sqrt(3), 8/(1-self.p))/self.v) * np.log(self.d*self.t**2/self.delta)/10
        if self.exploration: #theoretical bounds are too conservative.
            self.a_t = np.random.choice(self.K)
        else:
            y_hat = np.array(self.contexts) @ self.theta_hat
            pareto_index = [k for k in self.A]
            for i in self.A:
                for j in self.A:
                    if np.max(y_hat[i] - y_hat[j]) < 0:
                        pareto_index.remove(i)
                        break
            self.a_t = np.random.choice(pareto_index)

        # update Xs
        self.t = self.t + 1
        #self.Vinv = np.eye(self.K)/(self.t+self.lam)
        return(self.a_t)

    def update(self, reward):
        #Data mixup
        if not self.exploration:
            perturb_index = np.random.choice(len(self.exp_X_indexes))
            while self.exp_X_indexes[perturb_index] == self.a_t:
                perturb_index = np.random.choice(len(self.exp_X_indexes))
            perturb_weight1 = np.random.uniform(-1,1)*3**(1/2)
            perturb_weight2 = np.random.uniform(-1,1)*3**(1/2)
            X_a_t = perturb_weight1*np.array(self.contexts)[self.a_t]+perturb_weight2*np.array(self.contexts)[self.exp_X_indexes[perturb_index]]
            Y = perturb_weight1*reward + perturb_weight2*self.exp_Y[perturb_index]
        else:
            X_a_t = self.contexts[self.a_t]
            Y = reward
            self.exp_X_indexes.append(self.a_t)
            self.exp_Y.append(Y)

        #Imputation estimator
        self.xy += np.outer(X_a_t,Y)
        self.Ainv = sherman_morrison(X_a_t, self.Ainv)
        self.theta_hat = self.Ainv @ self.xy

        # Updating sets
        y_hat = np.array(self.contexts) @ self.theta_hat
        normalized_norms = {str(i):np.sqrt(np.dot(self.contexts[i],self.Ainv @ self.contexts[i])) for i in self.A}

        b = mbeta(self.t, self.d, self.L, self.p, self.delta, self.lam, np.linalg.norm(self.theta_hat)/self.L)
        #print(max([norms for norms in normalized_norms.values()])*b)
        # Update A and P
        C = [k for k in self.A]
        for i in C:
            for j in C:
                #if len(self.A) <=3:
                #    print(y_hat[self.A])
                #print('{} > {}'.format(m(y_hat[i,:],y_hat[j,:]), (normalized_norms[str(i)]+normalized_norms[str(j)])*b))
                if m(y_hat[i,:],y_hat[j,:]) > (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    self.A.remove(i)
                    break
        P = [k for k in self.A]
        for i in self.A:
            for j in self.A:
                if j != i and M(y_hat[i,:],y_hat[j,:],self.epsilon) <= (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    P.remove(i)
                    break
        P1 = P
        for j in P1:
            for i in list(set(self.A)-set(P1)):
                if M(y_hat[i,:],y_hat[j,:],self.epsilon) <= (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    P.remove(j)
                    break
        self.P = list(set(self.P) | set(P))
        self.A = list(set(self.A) - set(P))
        return()

'''
PFIridge
'''
class PFIridge:
    def __init__(self, contexts, L, hypers, seed=0):
        ## Hyperparameters in kwargs
        for x,y in hypers.items():
            setattr(self, x, y)

        ##Initialization
        self.seed = seed
        self.K = len(contexts)
        self.L = L
        self.d = len(contexts[0]) #real data Y is not linear to X
        self.contexts = contexts #np.eye(self.K) #np.array(contexts)
        self.t = 1
        self.A = [k for k in range(self.K)]
        self.P = []


        self.lam = 1+np.log(self.d*L/self.delta) #Theoretical Bound are too conservative
        self.theta_hat = np.zeros((self.d,L))
        # Exploration set = [K]
        X = np.array(self.contexts)
        self.v = np.min(np.linalg.eigvalsh(X.T @ X))/self.K

        #For imputation estimators
        self.Ainv = np.eye(self.d)/(self.p*self.lam)
        self.ridge_xy = np.zeros((self.d,L))

    def select(self):
        #print('t={}, E_t={}'.format(self.t, (max(32*np.sqrt(3), 8/(1-self.p))/self.v)*np.log(self.d*self.t**2/self.delta)/64))
        self.exploration = len(self.exp_Y) < (max(32*np.sqrt(3), 8/(1-self.p))/self.v) * np.log(self.d*self.t**2/self.delta)/10
        if self.exploration: #theoretical bounds are too conservative.
            self.a_t = np.random.choice(self.K)
        else:
            y_hat = np.array(self.contexts) @ self.theta_hat
            pareto_index = [k for k in self.A]
            for i in self.A:
                for j in self.A:
                    if np.max(y_hat[i] - y_hat[j]) < 0:
                        pareto_index.remove(i)
                        break
            self.a_t = np.random.choice(pareto_index)

        # update Xs
        self.t = self.t + 1
        #self.Vinv = np.eye(self.K)/(self.t+self.lam)
        return(self.a_t)

    def update(self, reward):
        X_a_t = self.contexts[self.a_t]
        Y = reward
        #ridge estimator
        self.xy += np.outer(X_a_t,Y)
        self.Ainv = sherman_morrison(X_a_t, self.Ainv)
        self.theta_hat = self.Ainv @ self.xy

        # Updating sets
        y_hat = np.array(self.contexts) @ self.theta_hat
        normalized_norms = {str(i):np.sqrt(np.dot(self.contexts[i],self.Ainv @ self.contexts[i])) for i in self.A}

        b = mbeta(self.t, self.d, self.L, self.p, self.delta, self.lam, np.linalg.norm(self.theta_hat)/self.L)
        #print(max([norms for norms in normalized_norms.values()])*b)
        # Update A and P
        C = [k for k in self.A]
        for i in C:
            for j in C:
                #if len(self.A) <=3:
                #    print(y_hat[self.A])
                #print('{} > {}'.format(m(y_hat[i,:],y_hat[j,:]), (normalized_norms[str(i)]+normalized_norms[str(j)])*b))
                if m(y_hat[i,:],y_hat[j,:]) > (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    self.A.remove(i)
                    break
        P = [k for k in self.A]
        for i in self.A:
            for j in self.A:
                if j != i and M(y_hat[i,:],y_hat[j,:],self.epsilon) <= (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    P.remove(i)
                    break
        P1 = P
        for j in P1:
            for i in list(set(self.A)-set(P1)):
                if M(y_hat[i,:],y_hat[j,:],self.epsilon) <= (normalized_norms[str(i)]+normalized_norms[str(j)])*b:
                    P.remove(j)
                    break
        self.P = list(set(self.P) | set(P))
        self.A = list(set(self.A) - set(P))
        return()

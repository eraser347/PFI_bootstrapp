#!/user/wk2389/.conda/envs/SMDP/bin/python

import numpy as np
from data import process_data
from itertools import product
import time, json, itertools, os
from train import PFIwR, MultiPFI
from tqdm import trange


def eval_real(model_name, data_name, K, hypers, N, seed=0):
    # model = model_name in str
    # hypers = {hyper_name: [hyper_value_1,...,hyper_value_n]}
    # evaluate model
    #names = ['sort_256','llvm_input','noc_CM_log']
    with open('./real/{}.txt'.format(data_name)) as infile:
        data_dict = json.load(infile)
    X = np.array(data_dict['X'])
    cluster_labels = np.array(data_dict['cluster_labels'])
    contexts = []
    for k in range(K):
        contexts.append(X[np.where(cluster_labels==k)[0][0],:].tolist())
    Y = np.array(data_dict['Y'])
    Y_means = np.array(data_dict['cluster_means'])[:,-2:]
    pareto_true = data_dict['pareto']
    results = []
    del(data_dict)
    #hyper_set = dict(zip(hypers.keys(),hypers))
    for hyper in product(*hypers.values()):
        hyper_set = dict(zip(hypers.keys(),hyper))
        # Identify true Pareto Front
        pareto = [i for i in range(16)]
        for i in range(16):
            for j in pareto:
                if np.max(Y_means[i,:] - Y_means[j,:]) < -hyper_set['epsilon']:
                    pareto.remove(i)
                    break
        #Initialize
        cumul_regret = []
        elapsed_time = []
        MSE = []
        P = []
        L = Y.shape[1]

        for n in range(N):
            print("{} Simulation {}, data={}, hyper ={}".format(model_name, n+1, data_name, str(hyper_set)))
            # Call model
            exec("global model; model = {}(contexts={},L={},hypers={})".format(model_name, contexts, L, str(hyper_set)))
            model_regret = []
            model_time = []
            model_MSE = []
            lA = 0
            while len(model.A) > 0:
                np.random.seed(seed+(n+1)*model.t)
                start = time.time()
                a_t = model.select()
                model_regret.append(max([max(min(Y_means[i,:]-Y_means[a_t,:]),0) for i in pareto]))
                model.update(Y[np.random.choice(np.where(cluster_labels == a_t)[0]),:])
                model_time.append(time.time() - start)
                if lA != len(model.A):
                    print('t={}, len(A)={}'.format(model.t,len(model.A)))
                lA = len(model.A)
                #mse = np.linalg.norm(reward - contexts @ model.theta_hat)**2/(L*K)
                #model_MSE.append(mse)
                #print('t={}, MSE={}'.format(model.t,mse))
            cumul_regret.append(np.cumsum(model_regret).tolist())
            elapsed_time.append(model_time)
            #MSE.append(model_MSE)
            print('Pareto: {}, Epsilon Pareto: {}, P: {}'.format(np.sort(pareto_true), np.sort(pareto), np.sort(model.P)))
            #print('Pareto: {}, P: {}, MSE: {:.5f}'.format(pareto, model.P, mse))
            P.append(model.P)
        ##Save at dict
        results.append({'hypers':hyper_set,
                        'regret':cumul_regret,
                        'P':P,
                        #'MSE':MSE,
                        'time':elapsed_time})
    ##Save to txt file
    if os.path.exists('./results') == False:
        os.mkdir('./results')
    with open('./results/{}_{}_seed{}.txt'.format(data_name, model_name, seed), 'w') as outfile:
        json.dump(results, outfile)

def eval_simul(model_name, contexts, theta, hypers, N, R, seed=0):
    # model = model_name in str
    # hypers = {hyper_name: [hyper_value_1,...,hyper_value_n]}
    # X = (K,d) contexts
    # theta = (d,L) parameters
    # evaluate model
    Y = contexts @ theta
    # Identify true Pareto Front
    pareto = [i for i in range(Y.shape[0])]
    for i in range(Y.shape[0]):
        for j in pareto:
            if np.max(Y[i,:] - Y[j,:]) < 0:
                pareto.remove(i)
                break
    results = []
    hyper_set = dict(zip(hypers.keys(),hypers))
    for hyper in product(*hypers.values()):
        #Initialize
        cumul_regret = []
        elapsed_time = []
        est_err = []
        ridge_err = []
        P = []
        L = Y.shape[1]
        hyper_set = dict(zip(hypers.keys(),hyper))
        for n in range(N):
            print("{} Simulation {}, hyper ={}".format(model_name, n+1, str(hyper_set)))
            # Call model
            exec("global model; model = {}(contexts={},L={},hypers={})".format(model_name, contexts.tolist(), L, str(hyper_set)))
            model_regret = []
            model_time = []
            model_est = []
            model_ridge = []
            lA = 0
            while len(model.A) > 0:
                np.random.seed(seed+(n+1)*model.t)
                start = time.time()
                #contexts[-1,:] = contexts[-2,:]
                a_t = model.select()
                model_regret.append(max([min(Y[i,:]-Y[a_t,:]) for i in pareto]))
                reward = Y[a_t,:] + np.random.normal(0, R, size=L)
                model.update(reward)
                model_time.append(time.time() - start)
                est = np.linalg.norm(theta - model.theta_hat)/L
                if lA != len(model.A):
                    print('t={}, len(A)={}, Error={:.5f}'.format(model.t,len(model.A), est))
                lA = len(model.A)
                model_est.append(est)
                #model_ridge.append(np.linalg.norm(theta - model.ridge_hat))
                #print('t={}, MSE={}'.format(model.t,mse))
            cumul_regret.append(np.cumsum(model_regret).tolist())
            elapsed_time.append(model_time)
            est_err.append(model_est)
            #ridge_err.append(model_ridge)
            print('Pareto: {}, P: {}, Error: {:.5f}'.format(pareto, model.P, est))
            P.append(model.P)
        ##Save at dict
        results.append({'hypers':hyper_set,
                        'Pareto':pareto,
                        'regret':cumul_regret,
                        'P':P,
                        'est_err':est_err,
                        #'ridge_err':ridge_err,
                        'time':elapsed_time})
    ##Save to txt file
    if os.path.exists('./results') == False:
        os.mkdir('./results')
    with open('./results/simul_{}_seed{}.txt'.format(model_name, seed), 'w') as outfile:
        json.dump(results, outfile)


def eval_est(model_name, X, theta, hypers, N, R, T, seed=0):
    # model = model_name in str
    # hypers = {hyper_name: [hyper_value_1,...,hyper_value_n]}
    # X = (K,d) contexts
    # theta = (d,L) parameters
    # evaluate model
    Y = X @ theta
    # Identify true Pareto Front
    pareto = [i for i in range(Y.shape[0])]
    for i in range(Y.shape[0]):
        for j in pareto:
            if np.max(Y[i,:] - Y[j,:]) < 0:
                pareto.remove(i)
                break
    results = []
    hyper_set = dict(zip(hypers.keys(),hypers))
    for hyper in product(*hypers.values()):
        #Initialize
        cumul_regret = []
        elapsed_time = []
        est_err = []
        ridge_err = []
        P = []
        L = Y.shape[1]
        d = X.shape[1]
        K = X.shape[0]
        hyper_set = dict(zip(hypers.keys(),hyper))
        for n in range(N):
            print("{} Simulation {}, hyper ={}".format(model_name, n+1, str(hyper_set)))
            # Call model
            exec("global model; model = {}(d={},K={},L={},hypers={})".format(model_name, d, K, L, str(hyper_set)))
            model_regret = []
            model_time = []
            model_est = []
            model_ridge = []
            lA = 0
            while model.t <= T:
                np.random.seed(seed+(n+1)*model.t)
                start = time.time()
                a_t = model.select(contexts=X + np.random.normal(0, R/np.sqrt(d), size=(K,d)))
                model_regret.append(max([min(Y[i,:]-Y[a_t,:]) for i in pareto]))
                reward = Y[a_t,:] + np.random.normal(0, R, size=L)
                model.update(reward)
                model_time.append(time.time() - start)
                est = np.linalg.norm(theta - model.theta_hat)/L
                if lA != len(model.A):
                    print('t={}, len(A)={}, Error={:.5f}'.format(model.t,len(model.A), est))
                lA = len(model.A)
                if len(model.A) == 0:
                    model.A = [k for k in range(K)]
                model_est.append(est)
                #model_ridge.append(np.linalg.norm(theta - model.ridge_hat))
                #print('t={}, MSE={}'.format(model.t,mse))
            cumul_regret.append(np.cumsum(model_regret).tolist())
            elapsed_time.append(model_time)
            est_err.append(model_est)
            #ridge_err.append(model_ridge)
            print('Pareto: {}, P: {}, Error: {:.5f}'.format(pareto, model.P, est))
            P.append(model.P)
        ##Save at dict
        results.append({'hypers':hyper_set,
                        'Pareto':pareto,
                        'regret':cumul_regret,
                        'P':P,
                        'est_err':est_err,
                        #'ridge_err':ridge_err,
                        'time':elapsed_time})
    ##Save to txt file
    if os.path.exists('./results') == False:
        os.mkdir('./results')
    with open('./results/est_{}.txt'.format(model_name), 'w') as outfile:
        json.dump(results, outfile)

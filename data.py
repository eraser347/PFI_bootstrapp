import csv, time, json, itertools, os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def process_data(name, q, K):
    ## Generate datasets
    # input: dataset_names = ['sort_256','llvm_input','noc_CM_log'], Outlier quantile = q []
    # output: np.array [X(1),...,X(K)] K contexts within unit balls, K rewards [Y(1),...,Y(K)] and Pareto_arms

    # Import from csv
    results = []
    with open('./train_data/{}.csv'.format(name), newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=';')
        for row in rows:
            results.append(row)
    results = np.array(results[1:], dtype=np.float32)

    # Normalize
    Y = results[:,-2:]
    #Y = Y/np.max(np.abs(Y),axis=0)
    X = results[:,:-2]
    #X = X/(np.sqrt(X.shape[1]+1)*np.max(np.abs(X),axis=0))

    # First linear regression
    reg1 = LinearRegression().fit(X, Y[:,0])
    reg2 = LinearRegression().fit(X, Y[:,1])
    # Identifying and Removing outliers
    res1 = np.abs(Y[:,0]-reg1.predict(X))
    res2 = np.abs(Y[:,1]-reg2.predict(X))
    not_outliers = [res1[i] <= np.quantile(res1, q) and res2[i] <= np.quantile(res2, q) for i in range(res1.shape[0])]
    Y = Y[not_outliers,:]
    X = X[not_outliers,:]
    results = results[not_outliers,:]
    # Fitting linear regression and compute R^2
    reg1 = LinearRegression().fit(X, Y[:,0])
    reg2 = LinearRegression().fit(X, Y[:,1])
    print('R^2: {:.3f} for Y1, {:.3f} for Y2'.format(reg1.score(X,Y[:,0]),reg2.score(X,Y[:,1])))
    print('Number of arms: {}'.format(X.shape[0]))

    # kmeans clustering
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(results)
    cluster_means = kmeans.cluster_centers_
    Y_means = cluster_means[:,-2:]
    #print(Y_means)
    cluster_labels = kmeans.labels_

    # Identify true Pareto Front
    pareto_index = [i for i in range(K)]
    for i in range(K):
        for j in pareto_index:
            if np.max(Y_means[i,:] - Y_means[j,:]) < 0:
                pareto_index.remove(i)
                break
    #print(pareto_index)
    ##Save to txt file

    if os.path.exists('./real') == False:
        os.mkdir('./real')
    with open('./real/{}_{}clusters.txt'.format(name, K), 'w') as outfile:
        json.dump({'X':np.hstack((np.ones(X.shape[0]).reshape(X.shape[0],1)/np.sqrt(X.shape[1]+1),X)).tolist(),
         'Y':Y.tolist(), 'pareto':pareto_index, 'cluster_labels':cluster_labels.tolist(), 'cluster_means':cluster_means.tolist()}, outfile)
    return()


def comparison_contexts(K, d, seed=0):
    ## Generate contexts
    # input: j, K, d
    # output: np.array [X(1),...,X(K)], [c(1),...,c(K)]  with K contexts in [0,1]^d and costs in [0,1]
    # Contexts

    #opt_context = np.hstack((np.random.uniform(-0.1-0.01,-0.1+0.01,size=(d-1)),np.random.uniform(1-0.005+1+0.005)))
    np.random.seed(seed) # For reproducibility
    #opt_con = np.random.uniform(-0.1,0, size=d//2)#np.hstack(( ))
    #opt_context = np.hstack((opt_con, 2/d+opt_con))
    #contexts = np.tile(opt_context, (K,1)).T + np.hstack((np.random.uniform(-0.5,0.5, size=(d,K-1)), np.zeros((d,1)) ))
    #contexts = np.random.uniform(-1/d,1/d,size=(d,K))
    #contexts = np.vstack((np.random.uniform(-1/d,0,size=(d//2,K)),np.random.uniform(0,1/d,size=(d-d//2,K))))
    #contexts = np.vstack((np.random.uniform(0.05-0.1,0.05+0.1,size=(d//2,K)),np.ones((d-d//2,K))))
    #contexts = np.random.uniform(0,1,size=(d,K))

    opt_context = np.append(np.zeros(d//2), np.ones(d-d//2))
    errors = np.vstack((np.random.uniform(0,0.05, size=(d//2,K-1)), np.random.uniform(-0.05,0, size=(d-d//2,K-1))))
    contexts = np.tile(opt_context, (K,1)).T + np.hstack((errors, np.zeros((d,1)) ))



    #contexts[:,0] = np.hstack((np.random.uniform(0,1,2), np.zeros(d-3), np.random.uniform(0.8,1)))
    #contexts[:,-1] = np.append(np.zeros(d-1),np.random.uniform(0.9,1.1))
    #contexts[:,0] = np.append(np.zeros(d-1), np.random.uniform(0.8,1,size=1))
    #contexts[:,1] = np.append(np.zeros(d-1), np.random.uniform(0.95,1,size=1))
    #contexts[:,2] = np.append(np.zeros(d-1), np.random.uniform(0.95,1,size=1))
    #contexts[:,2] = np.append(np.zeros(d-1), 1-3/d)
    #for k in np.arange(K//10):
    #    sub_con = np.random.uniform(0, 1/d, size=d//2)
    #    contexts[:,k] = np.hstack((sub_con, np.random.uniform(0,1/d)+sub_con))
        #contexts[:,k] = np.append(np.random.uniform(0,1,size=d-1), np.random.uniform(1/(2*d),1))
        #contexts[:,K-k-2] = np.append(np.random.uniform(0,1/(2*d),size=d-1), np.random.uniform(0.9,1,size=1))

    #sub_con = np.random.uniform(0, 1/d, size=d//2)
    #contexts[:,-2] = np.hstack((sub_con, 1.5/d+sub_con))
    #contexts[:0] = np.append(np.random.uniform(0,1,size=d-1), np.random.uniform(0.9,1))
    #contexts[:,-2] = np.hstack((0.002*np.ones(d//2), 0.001*np.ones(d-d//2)))
    #contexts[:,-1] = opt_context
    '''
    opt_context = np.append(np.zeros(d-2),np.ones(2))
    np.random.seed(seed) # For reproducibility
    #contexts = np.vstack((np.random.uniform(0,1,size=(d-1,K)),np.random.uniform(0,1/d,K)))
    contexts = np.vstack((np.random.uniform(0,1,size=(d-2,K)),np.random.uniform(0,1/d, size = (2,K))))
    contexts[:,-1] = opt_context
    contexts[:,0] = np.append(np.zeros(d-2), np.random.uniform(1-1/d, 1, 2))
    #contexts[:,1] = np.append(np.zeros(d-1), 1-2/d)
    #contexts[:,2] = np.append(np.zeros(d-1), 1-3/d)
    #for k in np.arange(K//10):
    #    contexts[:,k] = np.append(np.zeros(d-1), 1-1/d)
    #contexts[:,-2] = np.append(np.random.uniform(0,1/(2*d),size=d-1), 1-1/d) #no effect!
    '''
    return(contexts)

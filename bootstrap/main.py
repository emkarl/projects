import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from scipy import stats
import time
from multiprocessing import Pool
import functools as ft
import os
beta = 0.9
ar1 = np.array([1, -beta])
ma1 = np.array([1])
B = 499
replications = 100000
alpha = 0.05
sample_size = (10, 14, 20, 28, 40, 56, 80, 113, 160, 226, 320, 452, 640, 905, 1280)
rejects_t = []
rejects_pair = []
AR = ArmaProcess(ar1,ma1)

############################################################

def timer(myfun):
    def wrap_func(*args, **kwargs):
        start = time.time()
        res = myfun(*args, **kwargs)
        end = time.time()
        print("Time taken for {}: {} seconds".format(myfun.__name__, end - start))
        return res
    return wrap_func

def solve(x, y):
    return np.dot(y,x) / np.dot(x,x)

def calc_StdErr(beta, x, y):
    err1 = y - beta*x
    err2 = x - np.mean(x)
    a = np.dot(err1, err1) / (err1.size - 1)
    b = np.dot(err2, err2)
    return np.sqrt(a / b)

def calc_p(t_hat, t_samp, B):
    t_hat_arr = np.repeat(t_hat,B)
    p = 2 * min(np.sum(np.less_equal(t_samp, t_hat_arr)) / B,
                np.sum(np.less(t_hat_arr, t_samp)) / B)
    return p

############################################################
def student_t_test(alpha, n):
    lower = stats.t.ppf(alpha/2, df = n-1)
    upper = stats.t.ppf(1 - alpha/2, df = n-1)

    data = AR.generate_sample(nsample = n + 1)
    x = data[0:(n-1)]
    y = data[1:n]

    slope = solve(x,y)
    stderr = calc_StdErr(slope,x,y)
    T = (slope - beta) / stderr

    if T < lower or T > upper:
        return 1
    return 0

@timer
def student_t_simulation(alpha, n, replications):
    results = [student_t_test(alpha, n) for i in range(replications)]
    r = ft.reduce(lambda a,b: a + b, results, 0)
    return r / replications

############################################################

def do_one_res_bootstrap(n, slope_hat, residuals):
    rand_res = lambda size: np.random.choice(residuals, size)
    ar = np.array([1, -slope_hat])
    ma = np.array([1])
    AR_res = ArmaProcess(ar,ma)
    data = AR_res.generate_sample(nsample = n + 1, scale = 1, distrvs = rand_res)

    x = data[0:(n-1)]
    y = data[1:n]

    slope = solve(x, y)
    stderr = calc_StdErr(slope, x, y)
    T = (slope - slope_hat) / stderr

    return T

def residual_bootstrap_void(alpha, n, B):
    data = AR.generate_sample(nsample = n + 1)
    return 0

def residual_bootstrap(alpha, n, B):
    data = AR.generate_sample(nsample = n + 1)
    x = data[0:(n-1)]
    y = data[1:n]

    slope = solve(x, y)
    stderr = calc_StdErr(slope, x, y)
    t_hat = (slope - beta) / stderr
    residuals = y - slope*x

    t_samp = np.array([residual_bootstrap_void(n, slope, residuals) for i in range(B)])

    p = calc_p(t_hat, t_samp, B)

    if(p < alpha):
        return 1
    return 0

@timer
def residual_bootstrap_simulation_old(alpha, n, replications, B):
    results = [residual_bootstrap(alpha, n, B) for i in range(replications)]
    r = ft.reduce(lambda a,b: a + b, results, 0)
    return(r / replications)

@timer
def residual_bootstrap_simulation(alpha, n, replications, B):
    pool = Pool()
    results = [pool.apply_async(residual_bootstrap, (alpha, n, B,)) for i in range(replications)]
    r = ft.reduce(lambda a,b: a + b.get(), results, 0)
    return(r / replications)
############################################################

def wild_bootstrap(alpha, n, B):
    reject = 0
    for i in range(replications):
        data = AR.generate_sample(nsample = n + 1)
        x = data[0:(n-1)]
        y = data[1:n]
        result = stats.linregress(x,y)

############################################################

def do_one_pairs_bootstrap_void(beta_hat, x, y, index):
    index_r = index
    slope = solve(x[index_r],y[index_r])
    stderr = calc_StdErr(slope,x[index_r],y[index_r])

    while stderr == 0:
        # this is for the case we have only selected two kind of row
        index_r = np.random.choice(index, len(index))
        slope = solve(x[index_r],y[index_r])
        stderr = calc_StdErr(slope,x[index_r],y[index_r])

    T = (slope - beta_hat) / stderr

    return T

def do_one_pairs_bootstrap(beta_hat, x, y, index):
    index_r = np.random.choice(index, len(index))
    slope = solve(x[index_r],y[index_r])
    stderr = calc_StdErr(slope,x[index_r],y[index_r])

    while stderr == 0:
        # this is for the case we have only selected two kind of row
        index_r = np.random.choice(index, len(index))
        slope = solve(x[index_r],y[index_r])
        stderr = calc_StdErr(slope,x[index_r],y[index_r])

    T = (slope - beta_hat) / stderr

    return T

def pairs_bootstrap_test(alpha, n, B):
    data = AR.generate_sample(nsample = n + 1)
    x = data[0:(n-1)]
    y = data[1:n]

    slope = solve(x,y)
    stderr = calc_StdErr(slope, x, y)
    t_hat = (slope - beta) / stderr

    index = np.arange(n-1)
    t_samp = np.array([do_one_pairs_bootstrap(slope, x, y, index) for i in range(B)])

    p = calc_p(t_hat, t_samp, B)
    if(p < alpha):
        return 1
    return 0 

@timer
def pairs_bootstrap_simulation(alpha, n, replications, B):
    pool = Pool()
    results = [pool.apply_async(pairs_bootstrap_test, (alpha, n, B,)) for i in range(replications)]
    r = ft.reduce(lambda a,b: a + b.get(), results, 0)
    return(r / replications)

############################################################
rejects_t = []
rejects_pair = []
rejects_residual = []
print("Config: replications = {}, B = {}, core = {}".format(replications,B,os.cpu_count()))
for n in sample_size:
    print("Running student_t_simulation for n = {}".format(n))
    rejects_t.append(student_t_simulation(alpha, n, replications))
for n in sample_size:
    print("Running pairs_bootstrap_simulation for n = {}".format(n))
    rejects_pair.append(pairs_bootstrap_simulation(alpha, n, replications, B))
for n in sample_size:
    print("Running residual_bootstrap_simulation for n = {}".format(n))
    rejects_residual.append(residual_bootstrap_simulation(alpha, n, replications, B))
print(rejects_t)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(sample_size, rejects_t, color = 'blue')
ax.plot(sample_size, rejects_pair, color = 'green')
ax.plot(sample_size, rejects_residual, color = 'yellow')
ax.plot(sample_size, [alpha]*len(sample_size), color = 'red')
plt.savefig('/app/figures/plot.png')

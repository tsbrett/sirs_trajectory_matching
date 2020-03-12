import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

tp = {"R0": 2, "g": 1./5, "S_0": 1000, "I_0": 2, "N": 1200, "b": 0.5}

# ODE solver. Whole bunch of parameters currently hardcoded
#todo: currently par is a dict. This is likely not compatible with optimizer..
def solve_SIR_model(par):
    def SIR_model(t, y, R0, g, N):
        return [-R0*g*y[0]*y[1]/N,
                R0*g*y[0]*y[1]/N - g*y[1]]

    sol = solve_ivp(fun=SIR_model, t_span=[1,100], y0=[par["S_0"], par["I_0"]],
                    t_eval=np.arange(1,100,1),
                    args=(par["R0"], par["g"], par["N"]))

    output = pd.DataFrame({"t": sol["t"],
                           "S": sol["y"][0],
                           "I": sol["y"][1]})

    return output


# apply reporting error
test_data = solve_SIR_model(tp)
test_data["cases"] = np.random.poisson(lam=tp["b"]*test_data["I"], size=None)

plt.plot(test_data["t"], np.log(test_data["I"]))
plt.plot(test_data["t"], (test_data["t"]-1)*(tp["R0"]*tp["S_0"]/tp["N"]-1)*tp["g"] +np.log(2))
plt.ylim(0,6)
plt.show()

plt.plot(test_data["t"], tp["R0"]*test_data["S"]/tp["N"])
plt.show()

# Likelihood function
from scipy.stats import poisson
# Apply reporting process
lik_arr = poisson.logpmf(k=test_data["cases"],mu=tp["b"]*test_data["I"])
ll = lik_arr.sum()

# Cost function for optimizer
def cost_function(data, par):
    o = solve_SIR_model(par)
    lik_arr = poisson.logpmf(k=data, mu=par["b"] * o["I"])
    return lik_arr.sum()

par_arr = pd.DataFrame(tp, index=[0])
par_arr = pd.concat([par_arr]*20)
par_arr["R0"] = np.linspace(1.5,3,20)
par_arr.index = par_arr["R0"]

ll_profile = par_arr.apply(lambda x: cost_function(test_data["cases"], x),
                           axis=1)


# numerical optimizer

from scipy.optimize import minimize
# Clunky wrapper, lifts true pars then changes R0
def ll_wrapper(x):
    pars = tp.copy()
    pars["R0"] = x[0]
    return -cost_function(test_data["cases"], pars)


# Numerical optimizer using L-BFGS-B algorithm:
opt = minimize(ll_wrapper, x0=5,
               method="L-BFGS-B",
               bounds=[(1,10)],
               options={"maxiter": 200,
                        "disp": False})


plt.plot(ll_profile)
plt.axvline(tp["R0"], cost_function(test_data["cases"],tp), linestyle="--")
plt.scatter(opt["x"][0], -opt["fun"])
plt.show()
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

tp = {"R0": 2, "g": 1./2.5, "w": 1./(365*3), "eta": 1., "b1": 0.2, "t0":0.2*365,
      "S_0": 1e6, "I_0": 2, "N": 2e6, "b": 0.01, "tmax":365}

# ODE solver. Whole bunch of parameters currently hardcoded
#todo: currently par is a dict. This is likely not compatible with optimizer..
def solve_SIRS_model(par):

    #todo: this is not robust if N < S_0 + I_0
    initial_cond = [par["S_0"], par["I_0"], par["N"] -par["S_0"]-par["I_0"]]

    def SIRS_model(t, y, R0, g, w, eta, N, b1, t0):
        bt = (1+ b1*np.cos( 2*np.pi*(t-t0)/365.))
        return [-R0*bt*g*y[0]*(y[1]+ eta)/N + w*y[2],
                R0*bt*g*y[0]*(y[1] + eta)/N - g*y[1],
                g*y[1] - w*y[2]]

    sol = solve_ivp(fun=SIRS_model, t_span=[-10*365,par["tmax"]],
                    y0=initial_cond,
                    t_eval=np.arange(0,par["tmax"],7),
                    args=(par["R0"], par["g"], par["w"],
                          par["eta"], par["N"], par["b1"],  par["t0"]))

    output = pd.DataFrame({"t": sol["t"],
                           "S": sol["y"][0],
                           "I": sol["y"][1]})

    return output


# apply reporting error
test_data = solve_SIRS_model(tp)
test_data["cases"] = np.random.poisson(lam=tp["b"]*test_data["I"], size=None)


fig, ax = plt.subplots(1, figsize=(4,3))
ax.plot(test_data["t"], test_data["cases"])
ax.set_ylim(0,5e2)
ax.set_xlabel("Day")
ax.set_ylabel("Weekly \"Cases\"")
sns.despine(trim=True)
fig.tight_layout()
fig.savefig("./simulated_timeseries.pdf")
plt.show()

#
# plt.plot(test_data["t"], np.log(test_data["I"]))
# plt.plot(test_data["t"], (test_data["t"]-1)*(tp["R0"]*tp["S_0"]/tp["N"]-1)*tp["g"] +np.log(2))
# plt.ylim(0,6)
# plt.show()

# plt.plot(test_data["t"], tp["R0"]*test_data["S"]/tp["N"])
# plt.show()

# Likelihood function
from scipy.stats import poisson
# Apply reporting process
lik_arr = poisson.logpmf(k=test_data["cases"],mu=tp["b"]*test_data["I"])
ll = lik_arr.sum()

# Cost function for optimizer
def cost_function(data, par):
    o = solve_SIRS_model(par)
    lik_arr = poisson.logpmf(k=data, mu=par["b"] * o["I"])
    return lik_arr.sum()

n_points=50
par_arr = pd.DataFrame(tp, index=[0])
par_arr = pd.concat([par_arr]*n_points)
par_arr["R0"] = np.linspace(1.1,10,n_points)
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
opt = minimize(ll_wrapper, x0=3,
               method="L-BFGS-B",
               bounds=[(1,10)],
               options={"maxiter": 200,
                        "disp": False})


fig, ax = plt.subplots(1, figsize=(4,3))
ax.plot(ll_profile)
ax.axvline(tp["R0"], cost_function(test_data["cases"],tp), linestyle="--")
ax.scatter(opt["x"][0], -opt["fun"])
ax.set_xlabel("$R_0$")
ax.set_ylabel("Log-likelihood")
sns.despine(trim=True)
fig.tight_layout()
fig.savefig("./likelihood_profile.pdf")

plt.show()


# apply reporting error
mle = tp
mle["R0"] = opt["x"][0]
mle_sample = solve_SIRS_model(mle)
mle_sample["cases"] = np.random.poisson(lam=mle["b"]*mle_sample["I"],
                                        size=None)


fig, ax = plt.subplots(1, figsize=(4,3))
ax.plot(test_data["t"], test_data["cases"], label="Data")
ax.plot(mle_sample["t"], mle_sample["cases"], label="Simulation from MLE")
ax.set_ylim(0,5e2)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Weekly reported \"cases\"")
sns.despine(trim=True)
fig.tight_layout()
fig.savefig("./simulation_from_mle.pdf")
plt.show()


fig, ax = plt.subplots(1, figsize=(4,3))
ax.plot(test_data["t"], test_data["cases"], label="Data")
ax.plot(mle_sample["t"], mle_sample["cases"], label="Simulation from MLE")
ax.set_ylim(0,5e2)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Weekly reported \"cases\"")
sns.despine(trim=True)
fig.tight_layout()
fig.savefig("./simulation_from_mle.pdf")
plt.show()

tp2 = tp.copy()
tp2["tmax"] = 2*365
mle2 = mle.copy()
mle2["tmax"] = 2*365
test_data2 = solve_SIRS_model(tp2)
test_data2["cases"] = np.random.poisson(lam=tp2["b"]*test_data2["I"], size=None)
mle_sample2 = solve_SIRS_model(mle2)
mle_sample2["cases"] = np.random.poisson(lam=mle2["b"]*mle_sample2["I"], size=None)

fig, ax = plt.subplots(1, figsize=(4,3))
ax.plot(test_data2["t"], test_data2["cases"], label="Data")
ax.plot(mle_sample2["t"], mle_sample2["cases"], label="Simulation from MLE")
ax.set_ylim(0,5e2)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Weekly reported \"cases\"")
sns.despine(trim=True)
fig.tight_layout()
fig.savefig("./simulation_from_mle2.pdf")
plt.show()

n_points=50
par_arr = pd.DataFrame(tp, index=[0])
par_arr = pd.concat([par_arr]*n_points)
par_arr["R0"] = np.linspace(1.1,10,n_points)
par_arr.index = par_arr["R0"]
for i, par in par_arr.iterrows():
    ts = solve_SIRS_model(par)
    ts["cases"] = np.random.poisson(lam=par["b"]*ts["I"],
                                            size=None)
    plt.plot(ts["t"], ts["cases"])
plt.show()



ili_clean = pd.read_csv("./data/ili_clean.csv")

# ili.clean %>%
#   filter(state.abb=="MA") %>%
#   filter(between(season,2016,2016)) -> df_ma
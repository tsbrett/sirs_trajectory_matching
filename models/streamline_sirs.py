import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import poisson
from scipy.optimize import minimize

tp = {"R0": 2, "g": 1./2.5, "w": 1./(365*3), "eta": 1., "b1": 0.2,
      "tpeak":0.2*365,
      "S_0": 1e6, "I_0": 2, "N": 2e6, "b": 0.01, "tmax":365}

# ODE solver. Whole bunch of parameters currently hardcoded
#todo: currently par is a dict. This is likely not compatible with optimizer..
def solve_SIRS_model(par, times=np.arange(0,365,7), t0=-10*365):

    #todo: this is not robust if N < S_0 + I_0
    initial_cond = [par["S_0"], par["I_0"], par["N"] -par["S_0"]-par["I_0"], 0]

    def SIRS_model(t, y, R0, g, w, eta, N, b1, tpeak):
        bt = (1+ b1*np.cos( 2*np.pi*(t-tpeak)/365.))
        return [-R0*bt*g*y[0]*(y[1]+ eta)/N + w*y[2],
                R0*bt*g*y[0]*(y[1] + eta)/N - g*y[1],
                g*y[1] - w*y[2],
                g*y[1]]

    sol = solve_ivp(fun=SIRS_model, t_span=[t0,par["tmax"]],
                    y0=initial_cond,
                    t_eval=times,
                    args=(par["R0"], par["g"], par["w"],
                          par["eta"], par["N"], par["b1"],  par["tpeak"]))
    #could I just pass args as a dict? or tuple(dict(...))?

    output = pd.DataFrame({"t": sol["t"],
                           "S": sol["y"][0],
                           "I": sol["y"][1],
                           "R": sol["y"][2],
                           "C": sol["y"][3]
                           })

    return output

# apply reporting error
test_data = solve_SIRS_model(tp)
test_data["cases"] = np.random.poisson(lam=tp["b"]*(test_data["I"]), size=None)

# Cost function for optimizer
def cost_function(data, par):
    o = solve_SIRS_model(par)
    lik_arr = poisson.logpmf(k=data, mu=par["b"] * o["I"])
    return lik_arr.sum()

# numerical optimizer
# Clunky wrapper, lifts true pars then changes R0
def get_optimum_parameters():
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
    return opt


# Likelihood function
# Apply reporting process
lik_arr = poisson.logpmf(k=test_data["cases"],mu=tp["b"]*test_data["I"])
ll = lik_arr.sum()

# Calculate profile likelihood
n_points=50
par_arr = pd.DataFrame(tp, index=[0])
par_arr = pd.concat([par_arr]*n_points)
par_arr["R0"] = np.linspace(1.1,10,n_points)
par_arr.index = par_arr["R0"]

ll_profile = par_arr.apply(lambda x: cost_function(test_data["cases"], x),
                           axis=1)

opt = get_optimum_parameters()

# apply reporting error
mle = tp
mle["R0"] = opt["x"][0]
mle_sample = solve_SIRS_model(mle)
mle_sample["cases"] = np.random.poisson(lam=mle["b"]*mle_sample["I"],
                                        size=None)

##### Plots

fig, ax = plt.subplots(1, figsize=(4,3))
ax.plot(test_data["t"], test_data["cases"])
ax.set_ylim(0,5e2)
ax.set_xlabel("Day")
ax.set_ylabel("Weekly \"Cases\"")
sns.despine(trim=True)
fig.tight_layout()
fig.savefig("./simulated_timeseries.pdf")
plt.show()


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

# n_points=50
# par_arr = pd.DataFrame(tp, index=[0])
# par_arr = pd.concat([par_arr]*n_points)
# par_arr["R0"] = np.linspace(1.1,10,n_points)
# par_arr.index = par_arr["R0"]
# for i, par in par_arr.iterrows():
#     ts = solve_SIRS_model(par)
#     ts["cases"] = np.random.poisson(lam=par["b"]*ts["I"],
#                                             size=None)
#     plt.plot(ts["t"], ts["cases"])
# plt.show()



ili_clean = pd.read_csv("./data/ili_clean.csv")

# ili.clean %>%
#   filter(state.abb=="MA") %>%
#   filter(between(season,2016,2016)) -> df_ma
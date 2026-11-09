# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Prepares network files with monte-carlo parameter sweeps for solving process.

Relevant Settings
-----------------

.. code:: yaml

    monte_carlo:
    options:
        add_to_snakefile: false # When set to true, enables Monte Carlo sampling
        samples: 9 # number of optimizations. Note that number of samples when using scipy has to be the square of a prime number
        sampling_strategy: "chaospy"  # "pydoe2", "chaospy", "scipy", packages that are supported
        seed: 42 # set seedling for reproducibilty
    uncertainties:
        loads_t.p_set:
          type: uniform
          args: [0, 1]
        generators_t.p_max_pu.loc[:, n.generators.carrier == "onwind"]:
          type: lognormal
          args: [1.5]
        generators_t.p_max_pu.loc[:, n.generators.carrier == "solar"]:
          type: beta
          args: [0.5, 2]

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`monte_cf`

Inputs
------
- ``networks/elec_s_10_ec_lcopt_Co2L-24H.nc``

Outputs
-------
- ``networks/elec_s_10_ec_lcopt_Co2L-24H_{unc}.nc``

e.g.    networks/elec_s_10_ec_lcopt_Co2L-24H_m0.nc
        networks/elec_s_10_ec_lcopt_Co2L-24H_m1.nc
        ...

Description
-----------
PyPSA-Earth is deterministic which means that a set of inputs give a set of outputs.
Parameter sweeps can help to explore the uncertainty of the outputs cause by parameter changes.
Many are familiar with the classical "sensitivity analysis" that can be applied by varying the
input of only one feature, while exploring its outputs changes. Here implemented is a
"global sensitivity analysis" that can help to explore the multi-dimensional uncertainty space
when more than one feature are changed at the same time.

To do so, the scripts is separated in two building blocks: One creates the experimental design,
the other, modifies and outputs the network file. Building the experimental design is currently
supported by the packages pyDOE2, chaospy and scipy. This should give users the freedom to
explore alternative approaches. The orthogonal latin hypercube sampling is thereby found as most
performant, hence, implemented here. Sampling the multi-dimensional uncertainty space is relatively
easy. It only requires two things: The number of *samples* (defines the number of total networks to
be optimized) and *features* (pypsa network object e.g loads_t.p_set or generators_t.p_max_pu). This
results in an experimental design of the dimension (samples X features).

The experimental design `lh` (dimension: sample X features) is used to modify the PyPSA
networks. Thereby, this script creates samples x amount of networks. The iterators comes from the
wildcard {unc}, which is described in the config.yaml and created in the Snakefile as a range from
0 to (total number of) SAMPLES.
"""
import os
import re
import chaospy
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging, create_logger
from add_electricity import load_costs
from pyDOE2 import lhs
from scipy.stats import beta, gamma, lognorm, norm, qmc, triang, truncnorm
from sklearn.preprocessing import MinMaxScaler
from solve_network import *

#logger = create_logger(__name__)
sns.set(style="whitegrid")
logger = logging.getLogger(__name__)

# CREAZIONE WILDCARD PER MONTE CARLO: SE GLOBAL SENSITIVITY FARA' NUMERO ITERAZIONI UGUALE AI SAMPLE DEFINITI NEL CONFIG
# SE SINGLE BEST IN WORST FARA' NUMERO ITERAZIONI UGUALE AL NUMERO DI STORES DEFINITI NEL CONFIG
def wildcard_creator(config): #, method=None):

    # Creates wildcard for monte-carlo simulations.

    method = config["monte_carlo"]["options"].get("method")
    """ 
    if method == "global_sensitivity":
        return [f"h{i}" for i in range(config["monte_carlo"]["options"]["samples"])]
 """
    if method == "single_best_in_worst":
        return [
            f"a{i}"
            for i in range(len(config["electricity"]["extendable_carriers"]["Store"]))
        ]
    if method == "MC":
        return [f"m{i}" for i in range(config["monte_carlo"]["options"]["samples"])]
    if method == "SBAW":
        uncertainties = config["monte_carlo"]["distributions"][method]["uncertainties"]

        def extract_group(name):
            m = re.search(r'carrier\s*==\s*"([^"]+)"', name)
            if m:
                return m.group(1)

            m = re.search(r'str\.contains\("([^"]+)"\)', name)
            if m:
                label = m.group(1)
                if "H2LT" in label:
                    return "H2LT"
                if "H2HT" in label:
                    return "H2HT"
                return label

            return name

        groups = list(dict.fromkeys(extract_group(k) for k in uncertainties.keys()))
        return [f"s{i}" for i in range(len(groups))]

""" if method == "SBAW":
        return [
            f"s{i}" for i in range(len(config["monte_carlo"]["distributions"][method]["uncertainties"]))
            #for i in range(len(config["electricity"]["extendable_carriers"]["Store"]))
        ]"""


def monte_carlo_sampling_pydoe2(
    N_FEATURES: int,
    SAMPLES: int,
    uncertainties_values: dict,
    random_state: int,
    criterion: str = None,
    iteration: int = None,
    correlation_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Creates Latin Hypercube Sample (LHS) implementation from PyDOE2 with
    various options. Additionally, all "corners" are simulated.

    Adapted from Disspaset: https://github.com/energy-modelling-toolkit/Dispa-SET/blob/master/scripts/build_and_run_hypercube.py
    Documentation on PyDOE2: https://github.com/clicumu/pyDOE2 (fixes latin_cube errors)
    """
    from pyDOE2 import lhs
    from scipy.stats import qmc

    # Generate a Nfeatures-dimensional latin hypercube varying between 0 and 1:
    lh = lhs(
        N_FEATURES,
        samples=SAMPLES,
        criterion=criterion,
        iterations=iteration,
        random_state=random_state,
        correlation_matrix=correlation_matrix,
    )

    lh = rescale_distribution(lh, uncertainties_values)

    return lh


def monte_carlo_sampling_chaospy(
    N_FEATURES: int,
    SAMPLES: int,
    uncertainties_values: dict,
    seed: int,
    rule: str = "latin_hypercube",
) -> np.ndarray:
    """
    Creates Latin Hypercube Sample (LHS) implementation from chaospy.

    Documentation on Chaospy: https://github.com/clicumu/pyDOE2 (fixes latin_cube errors)
    Documentation on Chaospy latin-hyper cube (quasi-Monte Carlo method): https://chaospy.readthedocs.io/en/master/user_guide/fundamentals/quasi_random_samples.html#Quasi-random-samples
    """
    import chaospy
    from scipy.stats import qmc

    # generate a Nfeatures-dimensional latin hypercube varying between 0 and 1:
    N_FEATURES = (
        "chaospy.Uniform(0, 1), " * N_FEATURES
    )  # NUMERO PARAMS AFFETTO DA UNCERTAINTY
    uniform_cube = eval(
        f"chaospy.J({N_FEATURES})"
    )  # writes Nfeatures times the chaospy.uniform... command)
    lh = uniform_cube.sample(SAMPLES, rule=rule, seed=seed).T  # ESTRAZIONE RANDOMICA

    lh = rescale_distribution(lh, uncertainties_values)

    return lh



def monte_carlo_sampling_chaospy2(N_FEATURES, SAMPLES, SEED, rule="latin_hypercube"):
    """
    Creates Latin Hypercube Sample (LHS) implementation from chaospy.

    Documentation on Chaospy: https://github.com/clicumu/pyDOE2 (fixes latin_cube errors)
    Documentation on Chaospy latin-hyper cube (quasi-Monte Carlo method): https://chaospy.readthedocs.io/en/master/user_guide/fundamentals/quasi_random_samples.html#Quasi-random-samples

    """
    # Generate a Nfeatures-dimensional latin hypercube varying between 0 and 1:
    N_FEATURES = "chaospy.Uniform(0, 1), " * N_FEATURES
    uniform_cube = eval(
        f"chaospy.J({N_FEATURES})"
    )  # writes N features times the chaospy.uniform... command)
    lh = uniform_cube.sample(SAMPLES, rule=rule, seed=SEED).T
    # discrepancy = qmc.discrepancy(lh)
    # logger.info("Hypercube discrepancy is:", discrepancy)

    return lh


def monte_carlo_sampling_scipy(
    N_FEATURES: int,
    SAMPLES: int,
    uncertainties_values: dict,
    seed: int,
    strength: int = 2,
    optimization: str = None,
) -> np.ndarray:
    """
    Creates Latin Hypercube Sample (LHS) implementation from SciPy with various
    options:

    - Center the point within the multi-dimensional grid, centered=True
    - Optimization scheme, optimization="random-cd"
    - Strength=1, classical LHS
    - Strength=2, performant orthogonal LHS, requires the sample to be square of a prime e.g. sq(11)=121

    Options could be combined to produce an optimized centered orthogonal array
    based LHS. After optimization, the result would not be guaranteed to be of strength 2.

    Documentation for Quasi-Monte Carlo approaches: https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
    Documentation for Latin Hypercube: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html#scipy.stats.qmc.LatinHypercube
    Orthogonal LHS is better than basic LHS: https://github.com/scipy/scipy/pull/14546/files, https://en.wikipedia.org/wiki/Latin_hypercube_sampling
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(
        d=N_FEATURES,
        strength=strength,
        optimization=optimization,
        seed=seed,
    )

    lh = sampler.random(n=SAMPLES)

    lh = rescale_distribution(lh, uncertainties_values)

    return lh


def report_discrepancy(latin_hypercube: np.ndarray):
    """
    Calculates the discrepancy of a Latin hypercube sample (LHS) using the
    `scipy.stats.qmc` module. The discrepancy is a measure of how uniformly
    the sample points are distributed in the multi-dimensional space.
    """
    # samples space needs to be from 0 to 1
    mm = MinMaxScaler(feature_range=(0, 1), clip=True)
    lht = mm.fit_transform(latin_hypercube)

    discrepancy = qmc.discrepancy(lht)
    
    logger.info(
    "Discrepancy is: %s more details in function documentation.",
    discrepancy,
)


def single_best_in_worst_list(worst_list, best_list):
    """
    Return list with single best value per list of worst values

    The goal is to create a list of scenarios where each scenario
    has one best value and the rest are worst cases. This is useful for
    assessing the impact of a single technology on the system.
    E.g. if a technology is not optimised in any best case, it is
    likely not relevant to include in the optimization or
    global sensitivity analysis.

    What is good and what is bad is determined by the user.
    See example below for clarification of use cases.

    Input:
    ------
    worst_list: 1D array, represents the worst value per technology
    best_list: 1D array, represents the best value per technology

    Output:
    -------
    new_list: 2D array

    Example:
    --------
    Let's create a scenario set with 3 technologies (creates 3 scenarios)
    # Scenario set when low values are good e.g. costs
    >>> single_best_in_worst_list([4, 4, 4], [1, 1, 1])
    [[1, 4, 4], [4, 1, 4], [4, 4, 1]]
    # Scenario set when high values are good e.g. efficiency
    >>> single_best_in_worst_list([1, 1, 1], [4, 4, 4])
    [[4, 1, 1], [1, 4, 1], [1, 1, 4]]
    """
    new_list = []
    for i in range(len(worst_list)):
        l = worst_list.copy()
        l[i] = best_list[i]
        new_list.append(l)

    return new_list

import numpy as np
import pandas as pd
from scipy.stats import gamma, truncnorm, lognorm, uniform


def get_distribution_object(dist_type, params):
    if dist_type == 'gamma':
        shape = params[0]
        loc = params[1] if len(params) > 1 else 0
        scale = params[2] if len(params) > 2 else 1
        return gamma(shape, loc=loc, scale=scale)

    elif dist_type == 'truncnorm':
        a = params[0]
        b = params[1]
        loc = params[2]
        scale = params[3]
        return truncnorm(a, b, loc=loc, scale=scale)

    elif dist_type == 'lognorm':
        s = params[0]
        loc = params[1] if len(params) > 1 else 0
        scale = params[2] if len(params) > 2 else 1
        return lognorm(s, loc=loc, scale=scale)

    elif dist_type == 'uniform':
        loc = params[0]
        scale = params[1]
        return uniform(loc=loc, scale=scale)

    else:
        raise ValueError(f"Distribuzione non supportata: {dist_type}")

"""def build_percentile_matrix_from_config(uncertainties_config):
    tech_names = list(uncertainties_config.keys())
    count = len(tech_names)
    matrix = np.zeros((count, count))
    
    for i in range(count):
        for j in range(count):
            tech_j = tech_names[j]
            dist_info = uncertainties_config[tech_j]
            dist_type = dist_info['type']
            params = dist_info['args']
            
            dist = get_distribution_object(dist_type, params)
            mean = dist.mean()
            
            percentile = 0.05 if i == j else 0.95
            ppf_value = dist.ppf(percentile)
            
            # Coefficiente moltiplicativo rispetto alla media
            k = ppf_value / mean if mean != 0 else np.nan
            matrix[i, j] = k
            #carrier_names = [re.search(r'"(.*?)"', t).group(1) for t in tech_names]
    
    scenarios_k = pd.DataFrame(matrix, index=(tech_names), columns=tech_names)
    scenarios_k = scenarios_k.values  
    return scenarios_k"""

def build_percentile_matrix_from_config(uncertainties_config):

    tech_names = list(uncertainties_config.keys())

    # ---- funzione interna per estrarre il gruppo tecnologico ----
    def extract_group(name):
        # caso: carrier == "VRX"
        m = re.search(r'carrier\s*==\s*"([^"]+)"', name)
        if m:
            return m.group(1)

        # caso: str.contains("VRX") o "H2LT charger"
        m = re.search(r'str\.contains\("([^"]+)"\)', name)
        if m:
            label = m.group(1)
            # unifica charger/discharger
            if "H2LT" in label:
                return "H2LT"
            if "H2HT" in label:
                return "H2HT"
            return label

        return name  # fallback

    # ---- mappa feature → gruppo ----
    feature_to_group = {k: extract_group(k) for k in tech_names}

    # ---- lista gruppi unici ----
    groups = list(dict.fromkeys(feature_to_group.values()))
    n_groups = len(groups)
    n_features = len(tech_names)

    matrix = np.zeros((n_groups, n_features))

    # ---- costruzione matrice ----
    for i, group in enumerate(groups):
        for j, tech_j in enumerate(tech_names):

            dist_info = uncertainties_config[tech_j]
            dist_type = dist_info['type']
            params = dist_info['args']

            dist = get_distribution_object(dist_type, params)
            mean = dist.mean()

            # 👇 QUI LA LOGICA CHE VOLEVI
            if feature_to_group[tech_j] == group:
                percentile =  0.05 #0.05   # BEST
            else:
                percentile =  0.95 #0.95   # WORST

            ppf_value = dist.ppf(percentile)

            k = ppf_value / mean if mean != 0 else np.nan
            matrix[i, j] = k

    scenarios_k = pd.DataFrame(matrix, index=groups, columns=tech_names)

    return scenarios_k.values

def build_percentile_matrix_from_config_v0(uncertainties_config):
    tech_names = list(uncertainties_config.keys())
    count = len(tech_names)
    matrix = np.zeros((count, count))
    
    for i in range(count):
        for j in range(count):
            tech_j = tech_names[j]
            dist_info = uncertainties_config[tech_j]
            dist_type = dist_info['type']
            params = dist_info['args']
            
            dist = get_distribution_object(dist_type, params)
            mean = dist.mean()
            
            percentile = 0.05 if i == j else 0.95
            ppf_value = dist.ppf(percentile)
            
            # Coefficiente moltiplicativo rispetto alla media
            k = ppf_value / mean if mean != 0 else np.nan
            matrix[i, j] = k
            #carrier_names = [re.search(r'"(.*?)"', t).group(1) for t in tech_names]
    
    scenarios_k = pd.DataFrame(matrix, index=(tech_names), columns=tech_names)
    scenarios_k = scenarios_k.values  
    return scenarios_k

# QUI RISCALA I VALORI COMPRENDENDOLI DA  A 1 CONSIDERATO CHE ESTRAE DA UNA DISTRIBUZIONE DIFFERENTE 
# DI VALORI DIFFERENTI CON MAX MIN DIVERSI
def rescale_distribution(
    latin_hypercube: np.ndarray, uncertainties_values: dict
) -> np.ndarray:
    """
    Rescales a Latin hypercube sampling (LHS) using specified distribution
    parameters.
    """
    from scipy.stats import beta, gamma, lognorm, norm, triang
    from sklearn.preprocessing import minmax_scale

    for idx, value in enumerate(uncertainties_values):
        dist = value.get("type")
        params = value.get("args")

        match dist:
            case "uniform":
                loc, scale = params
                latin_hypercube[:, idx] = minmax_scale(
                    latin_hypercube[:, idx],
                    feature_range=(loc, loc + scale),
                )

            case "normal":
                mean, std = params
                latin_hypercube[:, idx] = norm.ppf(
                    latin_hypercube[:, idx],
                    loc=mean,
                    scale=std,
                )

            case "lognormal":
                if len(params) == 1:
                    s = params[0]
                    loc = 0
                    scale = 1
                elif len(params) == 2:
                    s, loc = params
                    scale = 1
                elif len(params) == 3:
                    s, loc, scale = params
                else:
                    raise ValueError(
                        "lognormal must have [s], [s, loc], or [s, loc, scale]"
                    )

                latin_hypercube[:, idx] = lognorm.ppf(
                    latin_hypercube[:, idx],
                    s=s,
                    loc=loc,
                    scale=scale,
                )

            case "triangle":
                if len(params) == 1:
                    c = params[0]
                    loc = 0
                    scale = 1
                elif len(params) == 2:
                    c, loc = params
                    scale = 1
                elif len(params) == 3:
                    c, loc, scale = params
                else:
                    raise ValueError(
                        "triangle must have [c], [c, loc], or [c, loc, scale]"
                    )

                latin_hypercube[:, idx] = triang.ppf(
                    latin_hypercube[:, idx],
                    c=c,
                    loc=loc,
                    scale=scale,
                )

            case "beta":
                if len(params) == 2:
                    a, b = params
                    loc = 0
                    scale = 1
                elif len(params) == 3:
                    a, b, loc = params
                    scale = 1
                elif len(params) == 4:
                    a, b, loc, scale = params
                else:
                    raise ValueError(
                        "beta must have [a, b], [a, b, loc], or [a, b, loc, scale]"
                    )

                latin_hypercube[:, idx] = beta.ppf(
                    latin_hypercube[:, idx],
                    a=a,
                    b=b,
                    loc=loc,
                    scale=scale,
                )

            case "gamma":
                if len(params) == 1:
                    shape = params[0]
                    loc = 0
                    scale = 1
                elif len(params) == 2:
                    shape, scale = params
                    loc = 0
                elif len(params) == 3:
                    shape, loc, scale = params
                else:
                    raise ValueError(
                        "gamma must have [shape], [shape, scale], or [shape, loc, scale]"
                    )

                latin_hypercube[:, idx] = gamma.ppf(
                    latin_hypercube[:, idx],
                    a=shape,
                    loc=loc,
                    scale=scale,
                )

            case "truncated_normal":
                a, b, mean, std = params
                latin_hypercube[:, idx] = truncnorm.ppf(
                    latin_hypercube[:, idx],
                    a,
                    b,
                    loc=mean,
                    scale=std,
                )

            case _:
                raise ValueError(f"Unsupported distribution: {dist}")

    report_discrepancy(latin_hypercube)
    return latin_hypercube

def validate_parameters(
    sampling_strategy: str, samples: int, uncertainties_values: dict
) -> None:
    valid_strategy = ["chaospy", "scipy", "pydoe2"]
    valid_distribution = [
        "uniform",
        "normal",
        "lognormal",
        "triangle",
        "beta",
        "gamma",
        "truncated_normal",
    ]

    if samples is None:
        raise ValueError("assign a value to samples")
    elif isinstance(samples, (float, str)):
        raise ValueError("samples must be an integer")

    if sampling_strategy not in valid_strategy:
        raise ValueError(
            f"Invalid sampling strategy: {sampling_strategy}. Choose from {valid_strategy}"
        )

    for value in uncertainties_values:
        dist_type = value.get("type")
        param = value.get("args")

        if dist_type is None or param is None or len(param) == 0:
            raise ValueError("assign a list of parameters to distribution_params")

        if dist_type not in valid_distribution:
            raise ValueError(
                f"Unsupported Distribution: {dist_type}. Choose from {valid_distribution}"
            )

        if dist_type == "uniform":
            if len(param) != 2:
                raise ValueError("uniform must have 2 parameters: [loc, scale]")
            if param[1] <= 0:
                raise ValueError("uniform: scale must be strictly positive")

        elif dist_type == "normal":
            if len(param) != 2:
                raise ValueError("normal must have 2 parameters: [mean, std]")
            if param[1] <= 0:
                raise ValueError("normal: std must be strictly positive")

        elif dist_type == "lognormal":
            if len(param) not in [1, 2, 3]:
                raise ValueError(
                    "lognormal must have [s], [s, loc], or [s, loc, scale]"
                )
            if param[0] <= 0:
                raise ValueError("lognormal: s must be strictly positive")
            if len(param) == 3 and param[2] <= 0:
                raise ValueError("lognormal: scale must be strictly positive")

        elif dist_type == "triangle":
            if len(param) not in [1, 2, 3]:
                raise ValueError(
                    "triangle must have [c], [c, loc], or [c, loc, scale]"
                )
            if not 0 <= param[0] <= 1:
                raise ValueError("triangle: c must be between 0 and 1")
            if len(param) == 3 and param[2] <= 0:
                raise ValueError("triangle: scale must be strictly positive")

        elif dist_type == "beta":
            if len(param) not in [2, 3, 4]:
                raise ValueError(
                    "beta must have [a, b], [a, b, loc], or [a, b, loc, scale]"
                )
            if param[0] <= 0 or param[1] <= 0:
                raise ValueError("beta: a and b must be strictly positive")
            if len(param) == 4 and param[3] <= 0:
                raise ValueError("beta: scale must be strictly positive")

        elif dist_type == "gamma":
            if len(param) not in [1, 2, 3]:
                raise ValueError(
                    "gamma must have [shape], [shape, scale], or [shape, loc, scale]"
                )
            if param[0] <= 0:
                raise ValueError("gamma: shape must be strictly positive")
            if len(param) == 2 and param[1] <= 0:
                raise ValueError("gamma: scale must be strictly positive")
            if len(param) == 3 and param[2] <= 0:
                raise ValueError("gamma: scale must be strictly positive")

        elif dist_type == "truncated_normal":
            if len(param) != 4:
                raise ValueError(
                    "truncated_normal must have 4 parameters: [a, b, loc, scale]"
                )
            if param[3] <= 0:
                raise ValueError(
                    "truncated_normal: scale must be strictly positive"
                )

    return None

def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6.
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n

def capex_to_capital_cost(k, capex_value, n, costs):
    """
    Convert sampled CAPEX to annualized PyPSA capital_cost.

    capex_value:
      - EUR/kW for generators/links
      - EUR/kWh for stores

    output:
      - EUR/MW/a for generators/links
      - EUR/MWh/a for stores
    """

    def annualize(tech, capex_kw):
        lifetime = costs.at[tech, "lifetime"]
        fom = costs.at[tech, "FOM"] if "FOM" in costs.columns else 0.0
        discount_rate = costs.at[tech, "discount rate"]

        capex_mw = capex_kw * 1000.0

        return (
            calculate_annuity(lifetime, discount_rate)
            + fom / 100.0
        ) * capex_mw

    if 'carrier == "solar"' in k:
        return annualize("solar", capex_value)

    if 'carrier == "onwind"' in k:
        return annualize("onwind", capex_value)

    if 'carrier == "FPVinnerbasin"' in k:
        return annualize("FPVinnerbasin", capex_value)

    if 'carrier == "offwind-ac"' in k:
        base_total = n.generators.loc[
            n.generators.carrier == "offwind-ac",
            "capital_cost",
        ]
        base_core = costs.at["offwind", "capital_cost"]
        extra = base_total - base_core
        return annualize("offwind", capex_value) + extra

    if 'carrier == "offwind-FPVsea"' in k:
        base_total = n.generators.loc[
            n.generators.carrier == "offwind-FPVsea",
            "capital_cost",
        ]
        base_core = costs.at["offwind-FPVsea", "capital_cost"]
        extra = base_total - base_core
        return annualize("offwind-FPVsea", capex_value) + extra

    if 'carrier == "VRX"' in k and "stores" in k:
        return annualize("V-RX-store", capex_value)

    if "VRX" in k and "links" in k:
        return annualize("V-RX-bicharger", capex_value)

    if 'carrier == "CCair"' in k and "stores" in k:
        return annualize("CAES-store", capex_value)

    if "CCair" in k and "links" in k:
        return annualize("CAES-bicharger", capex_value)

    if "LIION" in k and "stores" in k:
        return annualize("LI-ION-store", capex_value)

    if "LIION" in k and "links" in k:
        return annualize("LI-ION-bicharger", capex_value)
    
    if 'carrier == "FW"' in k and "stores" in k:
        return annualize("FLYWHEEL-store", capex_value)

    if "FW" in k and "links" in k:
        return annualize("FLYWHEEL-bicharger", capex_value)
    
    if 'carrier == "H2LT"' in k and "stores" in k:
        return annualize("hydrogen storage tank LT", capex_value)

    if "H2LT charger" in k and "links" in k:
        return annualize("PEM ELY-charger", capex_value)

    if "H2LT discharger" in k and "links" in k:
        return annualize("PEM FC-discharger", capex_value)
    
    if 'carrier == "H2HT"' in k and "stores" in k:
        return annualize("hydrogen storage tank HT", capex_value)

    if "H2HT charger" in k and "links" in k:
        return annualize("SOEC ELY-charger", capex_value)

    if "H2HT discharger" in k and "links" in k:
        return annualize("SO FC-discharger", capex_value)

    if 'carrier == "nuclear"' in k:
        return annualize("nuclear", capex_value)


    return capex_value

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "monte_carlo",
            simpl="",
            clusters="6",
            ll="copt",
            opts="Co2L-3H",
            unc="m0",
        )
    
    configure_logging(snakemake)
    config = snakemake.config
    #n = pypsa.Network(snakemake.input[0])
    n = pypsa.Network(snakemake.input.network)
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.params.costs,
        snakemake.params.electricity,
        Nyears,
    )
    monte_carlo_config = snakemake.params.monte_carlo

    ### PREPARAZIONE VARIABLES PER MONTE CARLO GLOBAL SENSIIVITY
    method = monte_carlo_config["options"]["method"]
    distributions = monte_carlo_config.get("distributions", {})
    
    if method == "MC":
        uncertainties_config = distributions.get("MC", {}).get("uncertainties", {})
        MONTE_CARLO_PYPSA_FEATURES = [k for k in uncertainties_config if k]
        N_FEATURES = len(MONTE_CARLO_PYPSA_FEATURES)
        # COMMON PARAMETERS
        # Parametri comuni
        MONTE_CARLO_OPTIONS = monte_carlo_config["options"]
        SAMPLES = MONTE_CARLO_OPTIONS.get("samples")
        SAMPLING_STRATEGY = MONTE_CARLO_OPTIONS.get("sampling_strategy")
        SEED = MONTE_CARLO_OPTIONS.get("seed")
        UNCERTAINTIES_VALUES = uncertainties_config.values()
        # PARAMETERS VALIDATION
        # validates the parameters supplied from config file
        validate_parameters(SAMPLING_STRATEGY, SAMPLES, UNCERTAINTIES_VALUES)

    if method == "single_best_in_worst":
    # Sottocaso di MC → usa le bounds specifiche
        uncertainties_config = distributions.get("single_best_in_worst", {}).get("uncertainties", {})
        L_BOUNDS = [item[0] for item in uncertainties_config.values()]
        U_BOUNDS = [item[1] for item in uncertainties_config.values()]
         
    if method == "SBAW":
        # Sottocaso di MC → usa le bounds specifiche
        
        uncertainties_config = distributions.get("SBAW", {}).get("uncertainties", {})
        MONTE_CARLO_PYPSA_FEATURES = [k for k in uncertainties_config if k]
   
    

    if monte_carlo_config["options"]["method"] == "MC":
        if SAMPLING_STRATEGY == "pydoe2":
            lh = monte_carlo_sampling_pydoe2(
                N_FEATURES,
                SAMPLES,
                UNCERTAINTIES_VALUES,
                random_state=SEED,
                criterion=None,
                iteration=None,
                correlation_matrix=None,
            )
        if SAMPLING_STRATEGY == "scipy":
            lh = monte_carlo_sampling_scipy(
                N_FEATURES,
                SAMPLES,
                UNCERTAINTIES_VALUES,
                seed=SEED,
                strength=2,
                optimization=None,
            )
        if SAMPLING_STRATEGY == "chaospy":
            lh = monte_carlo_sampling_chaospy(
                N_FEATURES,
                SAMPLES,
                UNCERTAINTIES_VALUES,  # RESCALING AFTER SAMPLING
                seed=SEED,
                rule="latin_hypercube",
            )
    if monte_carlo_config["options"]["method"] == "global_sensitivity":
        if SAMPLING_STRATEGY == "pydoe2":
            lh = monte_carlo_sampling_pydoe2(
                N_FEATURES,
                SAMPLES,
                criterion=None,
                iteration=None,
                random_state=42,
                correlation_matrix=None,
            )
        if SAMPLING_STRATEGY == "scipy":
            lh = monte_carlo_sampling_scipy(
                N_FEATURES,
                SAMPLES,
                centered=False,
                strength=2,
                optimization=None,
                seed=42,
            )
        if SAMPLING_STRATEGY == "chaospy":
            lh = monte_carlo_sampling_chaospy2(
                N_FEATURES,
                SAMPLES,
                SEED,
                rule="latin_hypercube",
                # ONLY SAMPLING
            )
        scenarios = qmc.scale(lh, L_BOUNDS, U_BOUNDS)  # HERE RESCALING

    if monte_carlo_config["options"]["method"] == "single_best_in_worst":
            carrier_no = len(n.stores.carrier.unique())
            worst_list = U_BOUNDS * carrier_no
            best_list = L_BOUNDS * carrier_no
            scenarios = single_best_in_worst_list(
                worst_list, best_list
            )  # matrix of upper and lower bounds for each stores listed in config.yaml (fattori moltiplicativi)

    if monte_carlo_config["options"]["method"] == "SBAW":
            matrix = build_percentile_matrix_from_config(uncertainties_config)
 
    unc_wildcards = snakemake.wildcards[-1]
    i = int(unc_wildcards[1:])
    j = 0

    if monte_carlo_config["options"]["method"] == "MC":
            # for k, v in enumerate(MONTE_CARLO_PYPSA_FEATURES): #.items():
        for k in MONTE_CARLO_PYPSA_FEATURES:
                # this loop sets in one scenario each "i" feature assumption
                # k is the config input key "loads_t.p_set"
                # v is the lower and upper bound [0.8,1.3], that was used for lh_scaled
                # i, j interation number to pick values of experimental setup
                # Example: n.loads_t.p_set = n.loads_t.p_set * lh_scaled[0,0]
                # si muove per campionamento, quindi riscala tutta la riga per tutti i param affetti da uncertainty
                value = lh[i, j]

                if "capital_cost" in k:
                    value = capex_to_capital_cost(k, value, n, costs)

                exec(f"n.{k} = value")
                logger.info(f"Set n.{k} to value {value} in the {i} scenario")

                #exec(f"n.{k} = {lh[i,j]}")
                #logger.info(f"Set n.{k} to value {lh[i,j]} in the {i} scenario")
                j = j + 1

    if monte_carlo_config["options"]["method"] == "single_best_in_worst":
            for k, _ in uncertainties_config.items():
                type = k.split(".")[0]  # "stores", "generators", ...
                feature = k.split(".")[1]  # "capital_cost", "efficiency", ...

            # TODO: Generalize for other features. Currently this scales the whole storage-chain
            if type == "stores":
                # scales the whole storage-chain
                carrier_list = n.stores.carrier.unique()
                for c in carrier_list:
                    n.stores.loc[n.stores.carrier == c, feature] *= scenarios[i][j]
                    n.links.loc[n.links.carrier.str.contains("H2"), feature] *= scenarios[
                        i
                    ][j]
                    logger.info(
                        f"Scaled {feature} for carrier={c} of store and links by factor {scenarios[i][j]} in the {i} scenario"
                    )
                    j += 1
   
    if monte_carlo_config["options"]["method"] == "SBAW":
            for k in MONTE_CARLO_PYPSA_FEATURES:
                exec(f"n.{k} = n.{k} * {matrix[i,j]}")
                logger.info(f"Scaled n.{k} by factor {matrix[i,j]} in the {i} scenario")
                j = j + 1

    ### EXPORT AND METADATA
        
    if monte_carlo_config["options"]["method"] == "global_sensitivity":
        scenario_dict = (
                    pd.DataFrame(scenarios).rename_axis("iteration").add_suffix("_feature")
                ).to_dict()
        n.meta.update(scenario_dict)
        n.export_to_netcdf(snakemake.output[0]) 
            

    if monte_carlo_config["options"]["method"] == "MC":
        latin_hypercube_dict = (
                pd.DataFrame(lh).rename_axis("Nruns").add_suffix("_feature")
            ).to_dict()
        n.meta.update(latin_hypercube_dict)  # AGGIORNA I DATI DEL NETWORK
        n.export_to_netcdf(snakemake.output[0])
        

    if monte_carlo_config["options"]["method"] == "single_best_in_worst":
        scenario_dict = (
                pd.DataFrame(scenarios).rename_axis("iteration").add_suffix("_feature")
            ).to_dict()
        n.meta.update(scenario_dict)
        n.export_to_netcdf(snakemake.output[0])

    if monte_carlo_config["options"]["method"] == "SBAW":
        df = pd.DataFrame(matrix)
        df.columns = [f"{i}_feature" for i in range(df.shape[1])]
        df.index = range(len(df))  # opzionale se già fatto prima
        scenario_dict = df.rename_axis("iteration").to_dict()

        n.meta.update(scenario_dict)
        n.export_to_netcdf(snakemake.output[0])
from fgspectra import frequency as fgf
import matplotlib.pyplot as plt
import numpy as np
import sacc
import argparse
from scipy.optimize import minimize
import inspect


def load_data(sacc_file, **selection_kwargs):
    """
    Load data from a SACC file

    Parameters
    ----------
    sacc_file : str
        Path to the SACC file.
    **selection_kwargs : dict
        Selection criteria for the data.

    Returns
    -------
    s : sacc.Sacc
        The loaded and filtered Sacc object.
    """
    s = sacc.Sacc.load_fits(sacc_file)
    s.keep_selection(**selection_kwargs)

    tokeep = []
    for tr1, tr2 in s.get_tracer_combinations():

        idxEE = s.indices("cl_ee", tracers=(tr1, tr2))
        idxBB = s.indices("cl_bb", tracers=(tr1, tr2))
        tokeep.extend(idxEE)
        tokeep.extend(idxBB)
    s.keep_indices(tokeep)
    return s


def filter_tracers(s, tracers):
    """
    Filter a given sacc file to keep
    only tracers present in the tracers list.

    Parameters
    ----------
    s : sacc.Sacc
        The loaded Sacc object.
    tracers : list of str
        List of tracer names to keep.

    Returns
    -------
    s : sacc.Sacc
        The filtered Sacc object.
    """
    tokeep = []
    for tr1, tr2 in s.get_tracer_combinations():
        if (tr1 in tracers) and (tr2 in tracers):
            idxEE = s.indices("cl_ee", tracers=(tr1, tr2))
            idxBB = s.indices("cl_bb", tracers=(tr1, tr2))
            # if "SAT" in tr1 and "SAT" in tr2:
            #     continue
            tokeep.extend(idxEE)
            tokeep.extend(idxBB)
    tokeep = np.array(tokeep).astype(int)
    s.keep_indices(tokeep)
    return s


def compsep(s, components):
    """
    Compute the component-separated signal and covariance
    for a given set of components.

    Parameters
    ----------
    s : sacc.Sacc
        The loaded Sacc object.
    components : dict
        Dictionary of components with their frequency
        response functions and parameters.

    Returns
    -------
    lb : np.ndarray
        The multipole array.
    ps_ML : np.ndarray
        The maximum likelihood estimate of the component-separated signal.
    cov_ML : np.ndarray
        The covariance matrix of the maximum likelihood estimate.
    """

    ncomp = len(components)

    # print(s.get_data_types())
    n_bins = (len(s.mean)) // (len(s.get_tracer_combinations()) * len(s.get_data_types()))
    S = np.zeros((len(s.mean), (2 * ncomp + 2) * n_bins))

    # Order is CMB E, CMB B first and then other components in order
    for id_tr_pair, (tr1, tr2) in enumerate(s.get_tracer_combinations()):
        idxEE = s.indices("cl_ee", tracers=(tr1, tr2))
        lb = np.array(s._get_tags_by_index(["ell"], idxEE)[0])

        idxBB = s.indices("cl_bb", tracers=(tr1, tr2))
        nu1 = s.tracers[tr1].nu[1]
        nu2 = s.tracers[tr2].nu[1]

        for i in range(n_bins):
            S[idxEE[i], 0*n_bins + i] = 1.0   # CMB E
            S[idxBB[i], 1*n_bins + i] = 1.0   # CMB B

            for j, comp in enumerate(components):
                fn = components[comp]["fn"]
                kwargs = components[comp]["kwargs"]
                response_nu1 = np.asarray(fn(nu=nu1, **kwargs)).item()
                response_nu2 = np.asarray(fn(nu=nu2, **kwargs)).item()
                component_cl = response_nu1 * response_nu2
                # Component E
                S[idxEE[i], (2 + 2*j)*n_bins + i] = component_cl
                # Component B
                S[idxBB[i], (2 + 2*j + 1)*n_bins + i] = component_cl
    ps = s.mean
    cov = s.covariance.covmat
    invcov = np.linalg.inv(cov)

    cov_ML = np.linalg.inv(S.T @ invcov @ S)
    ps_ML = cov_ML @ S.T @ invcov @ ps
    return lb, ps_ML, cov_ML, S, ps, cov, invcov


def compsep_dict(s, components):
    """
    Call compsep and return the result as a dictionary.
    """
    lb, ps_ML, cov_ML, S, ps, cov, invcov = compsep(s, components)
    ML_dict = {
        "ell": lb,
        "cl_ee_cmb": ps_ML[0*len(lb):1*len(lb)],
        "cl_bb_cmb": ps_ML[1*len(lb):2*len(lb)],
        "clerr_ee_cmb": np.sqrt(cov_ML[0*len(lb):1*len(lb), 0*len(lb):1*len(lb)].diagonal()),
        "clerr_bb_cmb": np.sqrt(cov_ML[1*len(lb):2*len(lb), 1*len(lb):2*len(lb)].diagonal()),
        "clcov_bb_cmb": cov_ML[1*len(lb):2*len(lb), 1*len(lb):2*len(lb)],
        "clcov_ee_cmb": cov_ML[0*len(lb):1*len(lb), 0*len(lb):1*len(lb)]
    }
    for j, comp in enumerate(components):
        ML_dict[f"cl_ee_{comp}"] = ps_ML[(2 + j)*len(lb):(2 + j + 1)*len(lb)]
        ML_dict[f"cl_bb_{comp}"] = ps_ML[(2 + j + 1)*len(la):(2 + j + 2)* len( lb)]
        ML_dict[f"clerr_ee_{comp}"] = np.sqrt(cov_ML[(2 + j)* len( lb):(2 + j + 1)* len( lb), (2 + j)* len( lb):(2 + j + 1)* len( lb)].diagonal())
        ML_dict[f"clerr_bb_{comp}"] = np.sqrt(cov_ML[(2 + j + 1)* len( lb):(2 + j + 2)* len( lb), (2 + j + 1)* len( lb):(2 + j + 2)* len( lb)].diagonal())
        ML_dict[f"clcov_ee_{comp}"] = cov_ML[(2 + j)* len( lb):(2 + j + 1)* len( lb), (2 + j)* len( lb):(2 + j + 1)* len( lb)]
        ML_dict[f"clcov_bb_{comp}"] = cov_ML[(2 + j + 1)* len( lb):(2 + j + 2)* len( lb), (2 + j + 1)* len( lb):(2 + j + 2)* len( lb)]
    return ML_dict


def main(args):
    """
    """
    s = load_data(args.sacc_file, ell__gt=args.lmin, ell__lt=args.lmax)
    tracers = args.tracers.split(",")
    s = filter_tracers(s, tracers)

    dust = fgf.ModifiedBlackBody()
    sync = fgf.Synchrotron()

    dust_kwargs = {"nu_0": args.nu0_dust, "temp": args.temp_dust, "beta": args.beta_dust}
    sync_kwargs = {"nu_0": args.nu0_sync, "beta": args.beta_sync}

    components = {
        "dust": {"fn": dust, "kwargs": dust_kwargs},
        "sync": {"fn": sync, "kwargs": sync_kwargs}
    }
    requested_components = args.comps.split(",")
    for c in requested_components:
        if c not in components:
            raise ValueError(f"Requested component '{c}' is not defined.")
    components = {c: components[c] for c in requested_components}

    lb, ML_cl, ML_cov, _, _, _, _ = compsep(s, components)

    ML_dict = {
        "ell": lb,
        "cl_ee_cmb": ML_cl[0*len(lb):1*len(lb)],
        "cl_bb_cmb": ML_cl[1*len(lb):2*len(lb)],
        "clerr_ee_cmb": np.sqrt(ML_cov[0*len(lb):1*len(lb), 0*len(lb):1*len(lb)].diagonal()),
        "clerr_bb_cmb": np.sqrt(ML_cov[1*len(lb):2*len(lb), 1*len(lb):2*len(lb)].diagonal()),
        "clcov_bb_cmb": ML_cov[1*len(lb):2*len(lb), 1*len(lb):2*len(lb)],
        "clcov_ee_cmb": ML_cov[0*len(lb):1*len(lb), 0*len(lb):1*len(lb)]
    }
    for j, comp in enumerate(components):
        ML_dict[f"cl_ee_{comp}"] = ML_cl[(2 + j)*len(lb):(2 + j + 1)*len(lb)]
        ML_dict[f"cl_bb_{comp}"] = ML_cl[(2 + j + 1)*len(lb):(2 + j + 2)*len(lb)]
        ML_dict[f"clerr_ee_{comp}"] = np.sqrt(ML_cov[(2 + j)*len(lb):(2 + j + 1)*len(lb), (2 + j)*len(lb):(2 + j + 1)*len(lb)].diagonal())
        ML_dict[f"clerr_bb_{comp}"] = np.sqrt(ML_cov[(2 + j + 1)*len(lb):(2 + j + 2)*len(lb), (2 + j + 1)*len(lb):(2 + j + 2)*len(lb)].diagonal())
        ML_dict[f"clcov_ee_{comp}"] = ML_cov[(2 + j)*len(lb):(2 + j + 1)*len(lb), (2 + j)*len(lb):(2 + j + 1)*len(lb)]
        ML_dict[f"clcov_bb_{comp}"] = ML_cov[(2 + j + 1)*len(lb):(2 + j + 2)*len(lb), (2 + j + 1)*len(lb):(2 + j + 2)*len(lb)]

    comps_label = "_".join(requested_components)
    fname = f"MLBR_cls_lmin{args.lmin}_lmax{args.lmax}_{comps_label}"
    if "dust" in requested_components:
        fname += f"nu0d{args.nu0_dust}_Td{args.temp_dust}_Bd{args.beta_dust}"
    if "sync" in requested_components:
        fname += f"nu0s{args.nu0_sync}_Bs{args.beta_sync}"
    np.savez(f"{fname}.npz", lb=lb, **ML_dict, tracers=tracers)

    if (
        args.lensed_th_cl is not None
    ) and (
        args.tensor1_th_cl is not None
    ):

        # Dummy index to collect a bandpower
        # window function
        idx = s.indices(
            "cl_bb",
            tracers=(
                list(s.tracers.keys())[0],
                list(s.tracers.keys())[0],
            )
        )
        bpw = s.get_bandpower_windows(idx).weight.T
        ells, TT, EE, BB, TE = np.loadtxt(args.lensed_th_cl, unpack=True)
        _, TTr1, EEr1, BBr1, TEr1 = np.loadtxt(args.tensor1_th_cl, unpack=True)
        BB_prim = BBr1 - BB
        BBlens = BB
        th_fc = 1.0
        # First fit CMB bandpowers
        template_Alens = bpw @ (BBlens/ells/(ells+1)*2*np.pi)[:bpw.shape[-1]] * th_fc
        template_r = bpw @ (BB_prim/ells/(ells+1)*2*np.pi)[:bpw.shape[-1]] * th_fc
        T = np.column_stack((template_Alens, template_r))
        bb = ML_dict["cl_bb_cmb"]
        C_inv = np.linalg.inv(ML_dict["clcov_bb_cmb"])
        t = np.linalg.solve(T.T @ C_inv @ T, T.T @ C_inv @ bb)
        t_cov = np.linalg.inv(T.T @ C_inv @ T)
        print(t, np.sqrt(t_cov.diagonal()))

        cmb_pars = {
            "Alens": t[0],
            "r": t[1]
        }

        # Then loop over components ?
        def power_law_model(ell, A, alpha):
            return A * (ell/80)**alpha
        models = {
            "dust": power_law_model,
            "sync": power_law_model,
        }
        model_pars = {}
        for comp in components:

            def tomin(p):
                model = models[comp](ells, *p)
                model = bpw @ (model/ells/(ells+1)*2*np.pi)[:bpw.shape[-1]] * th_fc
                res = model - ML_dict[f"cl_bb_{comp}"]
                chi2 = res @ np.linalg.inv(ML_dict[f"clcov_bb_{comp}"]) @ res
                return chi2

            npars = len(inspect.signature(models[comp]).parameters) - 1
            res = minimize(tomin, [1.]*npars, method="Powell")

            print(comp, res.x)
            model_pars[comp] = res.x

        plt.figure(figsize=(8, 6))
        plt.plot(
            ells,
            cmb_pars["Alens"] * BBlens * th_fc + cmb_pars["r"] * BB_prim * th_fc,
            color="k"
        )

        plt.errorbar(
            lb,
            ML_dict["cl_bb_cmb"],
            yerr=ML_dict["clerr_bb_cmb"],
            color="navy",
            label="CMB",
            marker="o",
            markerfacecolor="white",
            markersize=4,
            ls="None",
            markeredgewidth=0.9,
            elinewidth=0.9,
            zorder=6
        )
        plt.xlim(args.lmin, args.lmax)
        plt.ylim(-1e-2, 7e-2)

        plt.tick_params(axis="both", which="both", labelsize=12)
        plt.xlabel(r"$\ell$", fontsize=16)
        plt.ylabel(r"$D_\ell^{BB}$", fontsize=16)
        # plt.gcf().subplots_adjust(
        #     left=0.12,
        #     right=0.995,
        #     bottom=0.09,
        #     top=0.92
        # )
        plt.savefig(f"{fname}.pdf")

        plt.figure(figsize=(8, 6))
        for comp in requested_components:
            l, = plt.plot(
                ells,
                models[comp](ells, *model_pars[comp]) * th_fc,
                label=comp
            )
            plt.errorbar(
                lb,
                ML_dict[f"cl_bb_{comp}"],
                yerr=ML_dict[f"clerr_bb_{comp}"],
                color=l.get_color(),
                marker="o",
                markerfacecolor="white",
                markersize=4,
                ls="None",
                markeredgewidth=0.9,
                elinewidth=0.9
            )
        plt.xlim(args.lmin, args.lmax)
        plt.yscale("log")
        plt.legend(fontsize=14, loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.0))
        plt.xlabel(r"$\ell$", fontsize=16)
        plt.ylabel(r"$D_\ell^{BB}$", fontsize=16)
        # plt.gcf().subplots_adjust(
        #     left=0.12,
        #     right=0.995,
        #     bottom=0.09,
        #     top=0.92
        # )
        plt.savefig(f"{fname}_components.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sacc-file",
        help="Path to the SACC file",
        type=str
    )
    parser.add_argument(
        "--tracers",
        help="Comma separated list of tracers to keep",
        type=str
    )
    parser.add_argument(
        "--lmin",
        help="Minimum multipole to consider",
        type=float
    )
    parser.add_argument(
        "--lmax",
        help="Maximum multipole to consider",
        type=float
    )
    parser.add_argument(
        "--comps",
        help="Comma separated list of components to consider",
        type=str
    )
    parser.add_argument(
        "--nu0-dust",
        help="Pivot frequency for dust SED",
        default=353.0,
        type=float
    )
    parser.add_argument(
        "--temp-dust",
        help="Temperature for dust SED",
        default=20.0,
        type=float
    )
    parser.add_argument(
        "--beta-dust",
        help="Dust SED spectral index",
        default=1.5,
        type=float
    )

    parser.add_argument(
        "--nu0-sync",
        help="Pivot frequency for synchrotron SED",
        default=23.0,
        type=float
    )
    parser.add_argument(
        "--beta-sync",
        help="Synchrotron SED spectral index",
        default=-3.0,
        type=float
    )
    parser.add_argument(
        "--lensed-th-cl",
        help="Lensed theory Cls",
        default=None
    )
    parser.add_argument(
        "--tensor1-th-cl",
        help="r=1 + lensed theory Cls",
        default=None
    )
    args = parser.parse_args()
    main(args)
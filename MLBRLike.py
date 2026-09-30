from fgspectra import frequency as fgf
import numpy as np
from cobaya import Likelihood
from mlbr import load_data, filter_tracers, compsep


class BandpowerLike(Likelihood):
    """
    """

    @classmethod
    def get_modified_defaults(cls, defaults, input_options=None):
        """Declare bandpower outputs from the configured SACC file."""
        defaults = super().get_modified_defaults(defaults, input_options)
        input_options = input_options or {}
        sacc_file = input_options.get("sacc_file")
        if sacc_file is None:
            return defaults

        s = load_data(
            sacc_file,
            ell__gt=input_options.get("lmin"),
            ell__lt=input_options.get("lmax"),
        )
        tracers = input_options.get("tracers")
        if tracers is not None:
            s = filter_tracers(s, tracers)
        n_bins = len(s.indices(
            "cl_bb",
            tracers=s.get_tracer_combinations()[0]
        ))

        defaults["params"] = {
            **{
                f"Cb_{i}": {
                    "derived": True,
                    "latex": f"C_{{b{i}}}",
                }
                for i in range(n_bins)
            },
            **{
                f"Cb_{i}_{comp}": {
                    "derived": True,
                    "latex": f"C_{{b{i}}}^{{{comp}}}",
                }
                for i in range(n_bins)
                for comp in ("dust", "sync")
            },
        }
        return defaults

    def get_can_support_params(self):
        """Declare the sampled inputs consumed by ``logp``."""
        return ["beta_dust", "beta_sync", "temp_dust"]

    # Default settings
    sacc_file: str
    lmin: float
    lmax: float
    nu0_dust: float
    nu0_sync: float
    tracers: list

    def initialize(self):
        """
        """
        self.s = load_data(
            self.sacc_file,
            ell__gt=self.lmin,
            ell__lt=self.lmax
        )
        self.s = filter_tracers(self.s, self.tracers)

        self.dust = fgf.ModifiedBlackBody()
        self.sync = fgf.Synchrotron()
        dust_kwargs = {"nu_0": self.nu0_dust, "temp": 20.0, "beta": 1.5}
        sync_kwargs = {"nu_0": self.nu0_sync, "beta": -3.0}
        self.components = {
            "dust": {"fn": self.dust, "kwargs": dust_kwargs},
            "sync": {"fn": self.sync, "kwargs": sync_kwargs}
        }
        lb, _, _, _, ps, cov, invcov = compsep(self.s, self.components)

        self.lb = lb
        self.n_bins = len(self.lb)
        self.cov = cov
        self.invcov = invcov
        self.ps = ps

    def logp(self, beta_dust, beta_sync, temp_dust, _derived=None):
        self.components["dust"]["kwargs"].update(
            {
                "beta": beta_dust,
                "temp": temp_dust
            }
        )
        self.components["sync"]["kwargs"].update({"beta": beta_sync})

        lb, ps_ML, cov_ML, S, ps, cov, invcov = compsep(
            self.s,
            self.components
        )
        ps_ML_real = np.random.multivariate_normal(ps_ML, cov_ML)

        # Select only b for chi2
        # and use ML solution to fit for betas
        idx = self.s.indices("cl_bb")
        chi2 = (ps - S @ ps_ML)[idx].T @ np.linalg.solve(
            cov[np.ix_(idx, idx)],
            (ps - S @ ps_ML)[idx]
        )
        # chi2 = (ps - S @ ps_ML_real).T @ invcov @ (ps - S @ ps_ML_real)
        # print(chi2)

        if _derived is not None:
            for i in range(self.n_bins):
                _derived[f"Cb_{i}"] = ps_ML_real[
                    1*self.n_bins:
                    2*self.n_bins][i]
                for idc, comp in enumerate(self.components):
                    _derived[f"Cb_{i}_{comp}"] = ps_ML_real[
                        (2 + 2*idc + 1)*self.n_bins:
                        (2 + 2*idc + 2)*self.n_bins][i]

        return -0.5 * chi2

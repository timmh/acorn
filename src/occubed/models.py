import numpyro.distributions as dist

from biolith.models import occu as _upstream_occu
from occubed.cs_calibrated import occu_cs_calibrated
from occubed.cs_models import occu_cs, occu_cs_init_strategy, simulate_cs
from occubed.priors import DEFAULT_LOGIT_COEF_PRIOR


def occu(
    *args,
    prior_beta: dist.Distribution = DEFAULT_LOGIT_COEF_PRIOR,
    prior_alpha: dist.Distribution = DEFAULT_LOGIT_COEF_PRIOR,
    **kwargs,
):
    """Run upstream biolith occupancy with the experiment coefficient prior."""

    return _upstream_occu(
        *args, prior_beta=prior_beta, prior_alpha=prior_alpha, **kwargs
    )


__all__ = [
    "occu",
    "occu_cs",
    "occu_cs_calibrated",
    "occu_cs_init_strategy",
    "simulate_cs",
]

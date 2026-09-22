from numpyro.infer import MCMC


def disable_numpyro_mcmc_caching() -> None:
    """Disable NumPyro MCMC function caches in the current process."""

    import numpyro.infer.mcmc as numpyro_mcmc
    import numpyro.util as numpyro_util

    if getattr(MCMC, "_occubed_caching_disabled", False):
        getattr(numpyro_mcmc._collect_and_postprocess, "_cache", {}).clear()
        return

    def _no_cached_by(*_args, **_kwargs):
        return lambda fn: fn

    class _NoCache(dict):
        def get(self, *_args, **_kwargs):
            return None

        def __setitem__(self, *_args, **_kwargs):
            pass

        def clear(self):
            pass

    numpyro_util.cached_by = _no_cached_by
    numpyro_mcmc.cached_by = _no_cached_by
    getattr(numpyro_mcmc._collect_and_postprocess, "_cache", {}).clear()

    original_init = MCMC.__init__

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._cache = _NoCache()
        self._init_state_cache = _NoCache()

    MCMC.__init__ = _patched_init
    MCMC._occubed_caching_disabled = True

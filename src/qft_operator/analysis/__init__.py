"""Post-hoc analysis: anomalous-dimension spectrum extraction and residual fits."""

from qft_operator.analysis.spectrum import (
    SpectrumReport,
    anomalous_dimension_from_correlator,
    effective_dimension_from_correlator,
    fit_log_slope,
    summarize_spectrum,
)

__all__ = [
    "SpectrumReport",
    "anomalous_dimension_from_correlator",
    "effective_dimension_from_correlator",
    "fit_log_slope",
    "summarize_spectrum",
]

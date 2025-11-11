import math
from typing import Tuple

DEFAULT_ALPHA = 0.9
DEFAULT_BETA = 0.6


def alpha_beta_from_xi_eta(xi: float, eta_sym: float) -> Tuple[float, float]:
    """
    Compute the daughter diameter scaling factors (alpha, beta) from the
    asymmetry parameters xi and eta (beta/alpha).

    Uses:
        alpha**xi + beta**xi = 1
        eta = beta / alpha
    """
    if xi is None or eta_sym is None:
        raise ValueError("Both xi and eta_sym must be provided to derive alpha/beta.")

    xi = float(xi)
    eta = float(eta_sym)

    if not math.isfinite(xi) or xi <= 0.0:
        raise ValueError(f"xi must be positive and finite, got {xi}.")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError(f"eta_sym must be positive and finite, got {eta}.")

    alpha = math.pow(1.0 / (1.0 + math.pow(eta, xi)), 1.0 / xi)
    beta = eta * alpha
    return float(alpha), float(beta)


def resolve_branch_scaling(alpha=None,
                           beta=None,
                           xi=None,
                           eta_sym=None,
                           *,
                           default_alpha: float = DEFAULT_ALPHA,
                           default_beta: float = DEFAULT_BETA,
                           tol: float = 1e-6) -> Tuple[float, float]:
    """
    Resolve alpha and beta scaling factors from explicit inputs or xi/eta_sym.
    """
    has_alpha_beta = (alpha is not None) and (beta is not None)
    has_xi_eta = (xi is not None) and (eta_sym is not None)

    if has_xi_eta:
        derived_alpha, derived_beta = alpha_beta_from_xi_eta(xi, eta_sym)
        if has_alpha_beta:
            if (not math.isclose(float(alpha), derived_alpha, rel_tol=tol, abs_tol=tol) or
                not math.isclose(float(beta), derived_beta, rel_tol=tol, abs_tol=tol)):
                raise ValueError(
                    "Provided alpha/beta do not satisfy the xi/eta_sym relationship.")
        return derived_alpha, derived_beta

    if has_alpha_beta:
        return float(alpha), float(beta)

    # handle partially specified inputs
    if (alpha is None) ^ (beta is None):
        raise ValueError("Both alpha and beta must be provided if using explicit scaling.")
    if (xi is None) ^ (eta_sym is None):
        raise ValueError("Both xi and eta_sym are required when deriving alpha/beta.")

    if default_alpha is None or default_beta is None:
        raise ValueError("Either alpha/beta or xi/eta_sym must be provided.")

    return float(default_alpha), float(default_beta)

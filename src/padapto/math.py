from math import exp, log, log1p


def logaddexp(x: float, y: float) -> float:
    """
    Compute `log(exp(x) + exp(y))` without underflow.

    This uses the well-known "log sum exp" trick, similar to numpy's
    `np.logaddexp` function.
    """
    if x == y:
        return x + log(2)

    diff = x - y

    if diff > 0:
        return x + log1p(exp(-diff))
    elif diff <= 0:
        return y + log1p(exp(diff))
    else:
        return diff

"""Independent combinatorial oracle; integer arithmetic, no upstream recurrence."""

from fractions import Fraction
from math import comb, erfc, sqrt


def validate_counts(counts):
    counts = tuple(counts)
    if len(counts) != 3 or any(type(x) is not int or x < 0 for x in counts):
        raise ValueError('Expected three nonnegative integer genotype counts')
    if not 1 <= sum(counts) <= 1000:
        raise ValueError('Educational oracle supports 1..1000 called diploids')
    return counts


def conditional_distribution(n, copies):
    """P(H=h | N=n, ALT copies=copies) by counting labeled allele placements."""
    if type(n) is not int or not 1 <= n <= 1000:
        raise ValueError('n must be an integer in 1..1000')
    if type(copies) is not int or not 0 <= copies <= 2 * n:
        raise ValueError('copies must be an integer in 0..2n')
    denominator = comb(2 * n, copies)
    return {
        h: Fraction(comb(n, h) * comb(n - h, (copies - h) // 2) * 2 ** h,
                    denominator)
        for h in range(copies % 2, min(copies, 2 * n - copies) + 1, 2)
    }


def exact_tests(counts):
    ref, het, alt = validate_counts(counts)
    law = conditional_distribution(ref + het + alt, het + 2 * alt)
    observed = law[het]
    return {'two_sided': sum((p for p in law.values() if p <= observed), Fraction()),
            'deficit': sum((p for h, p in law.items() if h <= het), Fraction()),
            'excess': sum((p for h, p in law.items() if h >= het), Fraction()),
            'observed_probability': observed}


def plug_in_expectations(counts):
    ref, het, alt = validate_counts(counts)
    n = ref + het + alt
    q = Fraction(het + 2 * alt, 2 * n)
    return n * (1 - q) ** 2, 2 * n * q * (1 - q), n * q ** 2


def pearson_test(counts):
    counts = validate_counts(counts)
    expected = plug_in_expectations(counts)
    if any(e == 0 for e in expected):
        return None, None
    statistic = sum((o - e) ** 2 / e for o, e in zip(counts, expected))
    return float(statistic), erfc(sqrt(float(statistic) / 2))


def calibration(n, copies, alpha=Fraction(1, 20)):
    """Exact null rejection mass, not Monte Carlo or a genome-wide error rate."""
    if not isinstance(alpha, Fraction) or not 0 < alpha < 1:
        raise ValueError('alpha must be a Fraction strictly between 0 and 1')
    law = conditional_distribution(n, copies)
    exact_mass = Fraction()
    asymptotic_mass = Fraction()
    rows = []
    for h, probability in law.items():
        alt = (copies - h) // 2
        counts = (n - h - alt, h, alt)
        p_exact = exact_tests(counts)['two_sided']
        _, p_chi = pearson_test(counts)
        if p_exact <= alpha:
            exact_mass += probability
        if p_chi is not None and p_chi <= float(alpha):
            asymptotic_mass += probability
        rows.append({'counts': counts, 'null_probability': probability,
                     'exact_p': p_exact, 'chi_square_p': p_chi})
    return {'exact_rejection_mass': exact_mass,
            'chi_square_rejection_mass': asymptotic_mass, 'configurations': rows}

import numpy as np
import astropy.units as u
from scipy.stats import beta
from scipy.integrate import quad
from legwork.utils import get_a_from_ecc


N_MERGER = [23, 15]
FIT = [(34.55387671901061, 74.89576244079865,
        19.688610898930914, 1084.718030730763),
       (6.98249469541844, 23.979991700875626,
       66.28116728466077, 401.3853748298194)]


def p_e(e):
    """Find the probability that a binary will have eccentricity e after an
    encounter. E.g. Leigh+18 Eq. 35

    Parameters
    ----------
    e : `float/array`
        Eccentricity

    Returns
    -------
    p(e) : `float.array`
        Probability of having eccentricity e
    """
    return e * (1 - e**2)**(-1/2)


def rejection_sampling_e(sample_size=100):
    """Produce a sample of eccentricities matching the PDF given in p_e using
    a rejection sampling algorithm

    Parameters
    ----------
    sample_size : `int`
        Required sample size

    Returns
    -------
    samples : `float array`
        Eccentricity sample
    """
    samples = np.array([])

    # work out the height of the uniform distribution that contains p_e
    LARGE_E = 1 - 1e-3
    fac = p_e(LARGE_E)

    # while we still don't have enough samples
    while len(samples) < sample_size:
        # draw a sample of eccentricities and probabilities
        sample_e = np.random.rand(sample_size)
        sample_prob = np.random.rand(sample_size) * fac

        # calculate the probability of drawing each eccentricity
        true_prob = p_e(sample_e)

        # add the samples where sampled probability is below true probability
        samples = np.concatenate((samples, sample_e[sample_prob < true_prob]))

    # shuffle and trim to take a random subset of correct length
    np.random.shuffle(samples)
    samples = samples[:sample_size]

    return samples


def sample_immigrant_mass(size=(1000,), gamma=1, m_min=5*u.Msun,
                          m_max=50*u.Msun):
    """Draw a sample of immigrant masses. Use given gamma and the bounds of the
    masses where gamma is p(m)->m^(-gamma)

    Parameters
    ----------
    size : `tuple`
        Desired sample size (should be array shape)

    gamma : `int`
        Exponent of mass probability distribution (either 1 or 2)

    m_min : `float`
        Minimum immigrant mass

    m_max : `float`
        Maximum immigrant mass

    Returns
    -------
    m_imm : `float/array`
        Sampled immigrant masses with shape ``size``
    """
    assert gamma == 1 or gamma == 2, "Gamma must be 1 or 2"

    if gamma == 1:
        m_imm = m_min * (m_max / m_min)**(np.random.random_sample(size=size))
    if gamma == 2:
        m_imm = ((1 / m_min) - np.random.random_sample(size=size)
                 * ((1 / m_min) - (1 / m_max)))**(-1)
    return m_imm


def merger_mass_loss(q, a_1, a_2):
    """Compute mass lost during a GW merger between two BHs. Expression from
    Tichy & Marronetti 2008.

    Parameters
    ----------
    q : `float/array`
        Mass ratio of binary
    a_1 : `float/array`
        Spin of primary BH
    a_2 : `float/array`
        Spin of secondary BH

    Returns
    -------
    loss : `float/array`
        Fraction of total binary mass lost during merger
    """
    v = q / (1 + q)**2
    loss = 0.2 * v + 0.208 * v**2 * (a_1 + a_2)
    return loss


def merger_final_spin(q, a_1, a_2):
    """Compute final BH spin after a GW merger between two BHs. Expression from
    Tichy & Marronetti 2008.

    Parameters
    ----------
    q : `float/array`
        Mass ratio of binary
    a_1 : `float/array`
        Spin of primary BH
    a_2 : `float/array`
        Spin of secondary BH

    Returns
    -------
    a_f : `float/array`
        Final spin of the BH after merger
    """
    v = q / (1 + q)**2
    a_f = 0.686 * (5.04 * v - 4.16 * v**2)\
        + 0.4 * ((a_1 / (0.632 + 1 / q)**2) + (a_2 / (0.632 + q)**2))
    return a_f


def fit_final_oligarch_mass(gamma, a_BH=1.0, size=500000):
    """Produce a fit for the final oligarch mass distribution

    Parameters
    ----------
    gamma : `int`
        Exponent of mass probability distribution (either 1 or 2)
    size : `int`, optional
        Size of sample to take for fitting, by default 500000

    Returns
    -------
    fit : `tuple`
        Fit for mass distribution

    bounds : `tuple`
        Minimum and maximum simulated masses
    """

    m_immigrants = sample_immigrant_mass(gamma=gamma,
                                         size=(N_MERGER[gamma - 1], size))

    m_oligarch = m_immigrants[0, :]
    aIMBH = np.repeat(a_BH, size)

    for i in range(1, N_MERGER[gamma - 1]):
        q = m_immigrants[i, :] / m_oligarch
        m_binary = m_oligarch + m_immigrants[i, :]
        m_oligarch = m_binary * (1 - merger_mass_loss(q, aIMBH, a_BH))
        aIMBH = merger_final_spin(q, aIMBH, a_BH)

    return beta.fit(data=m_oligarch.value), (np.min(m_oligarch.value),
                                             np.max(m_oligarch.value))


def a_from_t_merge(ecc_i, t_merge, beta):
    """Find the initial semi-major axis given the initial eccentricity and
    merger time. (Solve Peters 1964 Eq. 5.14 for c0 and convert to a_i using
    Eq. 5.11)

    Parameters
    ----------
    ecc_i : `float/array`
        Initial eccentricity

    t_merge : `float/array`
        Time until merger

    beta : `float/array`
        Beta constant from Peters 1964

    Returns
    -------
    a_i : `float/array`
        Initial semi-major axis

    c_0 : `float/array`
        c0 constant from Peters 1964
    """
    def intfunc(e):
        return (e**(29/19) * (1 + 121/304 * e**2)**(1181/2299))\
            / (1 - e**2)**(3/2)

    c_0 = np.array([((19/12 * t_merge[i] * beta[i]
                     / quad(intfunc, 0, ecc_i[i])[0])**(1/4)).to(u.AU).value
                   for i in range(len(ecc_i))]) * u.AU
    a_i = get_a_from_ecc(ecc=ecc_i, c_0=c_0)
    return a_i, c_0

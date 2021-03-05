import numpy as np

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
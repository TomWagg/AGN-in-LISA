import numpy as np
import astropy.units as u
from scipy.integrate import odeint
import legwork

from helpers import *


def simulate_AGN_rate(n_AGN=200, gamma=1, encounter_factor=10, t_obs=4*u.yr,
                      max_distance=1*u.Gpc, galaxy_density=4e6*u.Gpc**(-3),
                      AGN_fraction=0.01, snr_cutoff=7):

    # ensure the input is sensible
    assert gamma == 1 or gamma == 2, "Gamma must be 1 or 2"
    assert encounter_factor >= 1 and encounter_factor <= 10,\
        "Encounter factor must be between 1 and 10"

    # define the AGN lifetime as a megayear
    AGN_lifetime = 1 * u.Myr
    
    # define the number of encounters given gamma
    n_mergers = 23 if gamma == 1 else 15
    n_encounters = encounter_factor * n_mergers

    # work out average time between encounters
    encounter_timescale = (AGN_lifetime / n_encounters).to(u.yr)
    
    # randomly sample AGN age and other times
    AGN_age = np.random.rand(n_AGN) * AGN_lifetime
    t_encounter_to_merge = np.random.rand(n_AGN) * encounter_timescale
    t_since_encounter = np.random.rand(n_AGN) * t_encounter_to_merge
    
    # randomly sample distance to each AGN
    distance = np.random.rand(n_AGN)**(1/3) * max_distance
    
    # randomly sample final oligarch mass for each AGN
    a, b, loc, scale = FIT[gamma - 1]
    final_m_oligarch = (np.random.beta(a=a, b=b, size=n_AGN)
                        * scale + loc) * u.Msun
    
    # work out current oligarch mass based on growth at m->t^(3/2)
    m_oligarch = (final_m_oligarch / AGN_lifetime**(3/2)) * AGN_age**(3/2)
    
    # randomly sample immigrant mass for each AGN based on gamma
    m_immigrant = sample_immigrant_mass(size=(n_AGN,), gamma=gamma)

    m_c = legwork.utils.chirp_mass(m_oligarch, m_immigrant)
    beta = legwork.utils.beta(m_oligarch, m_immigrant)

    # draw eccentricity from Leigh+18 distribution
    e_enc = rejection_sampling_e(n_AGN)
    
    # calculate separation based on mass, eccentricity and inspiral times
    a_enc, c0_enc = a_from_t_merge(e_enc, t_encounter_to_merge, beta)

    e_evol = legwork.evol.evol_ecc(ecc_i=e_enc, beta=beta, a_i=a_enc,
                                   output_vars="ecc", t_evol=t_since_encounter,
                                   n_step=2)
    e_LISA = e_evol.T[-1]
    
    # convert to separation
    a_LISA = legwork.utils.get_a_from_ecc(e_LISA, c0_enc)
    forb_LISA = legwork.utils.get_f_orb_from_a(a_LISA, m_oligarch, m_immigrant)

    sources = legwork.source.Source(m_1=m_oligarch, m_2=m_immigrant,
                                    dist=distance, ecc=e_LISA, f_orb=forb_LISA)
    snr = sources.get_snr(verbose=True)

    sample_volume = max_distance**3
    n_AGN = galaxy_density * sample_volume * AGN_fraction
    fraction_GW_emission = (encounter_timescale / AGN_lifetime
                            * n_mergers).decompose()
    fraction_detectable = len(snr[snr > snr_cutoff]) / n_AGN

    n_detection = n_AGN * fraction_GW_emission * fraction_detectable
    
    return snr, n_detection

snr, n_detection = simulate_AGN_rate(5)
print(snr, n_detection)
import numpy as np
import astropy.units as u
import h5py as h5
import legwork as lw
import getopt
import sys
import os.path

from helpers import N_MERGER, FIT, a_from_t_merge, sample_immigrant_mass,\
    rejection_sampling_e


def simulate_LISA_AGN_rate(n_AGN=200, gamma=1, encounter_factor=10,
                           t_obs=4*u.yr, max_distance=1*u.Gpc,
                           galaxy_density=4e6*u.Gpc**(-3), AGN_fraction=0.01,
                           snr_cutoff=7):
    """Simulate the number of AGN migration trap binaries that LISA will see.

    Parameters
    ----------
    n_AGN : `int`, optional
        Number of AGNs to simulate, by default 200

    gamma : `int`, optional
        Exponent of black hole mass distribution p(m)->m^(-gamma), by default 1

    encounter_factor : `int`, optional
        Number of encounters each binary receives, by default 10

    t_obs : `float`, optional
        LISA mission length, by default 4 years

    max_distance : `float`, optional
        Maximum distance to simulate, by default 1 Gpc

    galaxy_density : `float`, optional
        Number density of galaxies in local universe, by default 4e6 / Gpc^3

    AGN_fraction : `float`, optional
        Fraction of galaxies that are currently AGNs, by default 0.01

    snr_cutoff : `int`, optional
        SNR threshold for detection, by default 7


    Returns
    -------
    results : `dict`
        Dictionary of results
    """
    # ensure the input is sensible
    assert gamma == 1 or gamma == 2, "Gamma must be 1 or 2"
    assert encounter_factor >= 1 and encounter_factor <= 10,\
        "Encounter factor must be between 1 and 10"

    # define the AGN lifetime as a megayear
    AGN_lifetime = 1e6 * u.yr

    # define the number of encounters given gamma
    n_encounters = encounter_factor * N_MERGER[gamma - 1]

    # work out average time between encounters
    encounter_timescale = (AGN_lifetime / n_encounters).to(u.yr)

    # randomly sample AGN age and other times
    AGN_age = np.random.rand(n_AGN) * AGN_lifetime
    t_encounter_to_merge = np.random.rand(n_AGN) * encounter_timescale
    t_since_encounter = np.random.rand(n_AGN) * t_encounter_to_merge

    # randomly sample distance to each AGN (uniform in volume)
    distance = np.random.rand(n_AGN)**(1/3) * max_distance

    # randomly sample final oligarch mass for each AGN
    a, b, loc, scale = FIT[gamma - 1]
    final_m_oligarch = (np.random.beta(a=a, b=b, size=n_AGN)
                        * scale + loc) * u.Msun

    # work out current oligarch mass based on growth at m->t^(3/2)
    m_oligarch = (final_m_oligarch / AGN_lifetime**(3/2)) * AGN_age**(3/2)

    # randomly sample immigrant mass for each AGN based on gamma
    m_immigrant = sample_immigrant_mass(size=(n_AGN,), gamma=gamma)

    # draw eccentricity from Leigh+18 distribution
    e_enc = rejection_sampling_e(n_AGN)

    # calculate separation based on mass, eccentricity and inspiral times
    beta = lw.utils.beta(m_oligarch, m_immigrant)
    a_enc, c0_enc = a_from_t_merge(e_enc, t_encounter_to_merge, beta)

    e_evol = lw.evol.evol_ecc(ecc_i=e_enc, beta=beta, a_i=a_enc,
                              output_vars="ecc", t_evol=t_since_encounter,
                              n_step=2)
    e_LISA = e_evol.T[-1]

    # convert to separation
    a_LISA = lw.utils.get_a_from_ecc(e_LISA, c0_enc)
    f_orb_LISA = lw.utils.get_f_orb_from_a(a_LISA, m_oligarch, m_immigrant)

    sources = lw.source.Source(m_1=m_oligarch, m_2=m_immigrant,
                               dist=distance, ecc=e_LISA, f_orb=f_orb_LISA,
                               sc_params={"t_obs": t_obs})
    snr = sources.get_snr(t_obs=t_obs, verbose=True)
    max_snr_harmonic = sources.max_snr_harmonic
    del sources

    sample_volume = max_distance**3
    n_AGN = galaxy_density * sample_volume * AGN_fraction
    fraction_GW_emission = 1 / encounter_factor
    fraction_detectable = len(snr[snr > snr_cutoff]) / n_AGN

    n_detection = n_AGN * fraction_GW_emission * fraction_detectable

    results = {
        "m_1": m_oligarch,
        "m_2": m_immigrant,
        "dist": distance,
        "e_LISA": e_LISA,
        "f_orb_LISA": f_orb_LISA,
        "snr": snr,
        "max_snr_harmonic": max_snr_harmonic,
        "AGN_age": AGN_age,
        "t_e2m": t_encounter_to_merge,
        "t_se": t_since_encounter,
        "m_oligarch_final": final_m_oligarch,
        "a_enc": a_enc,
        "e_enc": e_enc,
        "n_detections": n_detection
    }

    return results


def usage():
    print("usage: python simulate_AGN.py [options]")
    print("options:")
    print("\t-h, --help               : print these usage instructions")
    print("\t-o, --output             : path to output h5 file")
    print("\t-n, --n-AGN              : number of AGN")
    print("\t-g, --gamma              : BH mass distribution exponent (1,2)")
    print("\t-e, --encounter-factor   : Encounters per binary (1-10)")
    print("\t-m, --mission-length     : LISA Mission length [years]")
    print("\t-d, --max-distance       : Maximum AGN distance [Gpc]")
    print("\t-G, --galaxy-density     : Local galaxy density [1/Gpc^3]")
    print("\t-a, --AGN-fraction       : Fraction of galaxies that are AGN")
    print("\t-s, --snr-cutoff         : SNR detection threshold")
    print("\t-l, --loops              : Number of simulations")


def main():
    # get command line arguments and exit if error
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "ho:n:g:e:m:d:G:a:s:l:",
                                ["help", "output=", "n-AGN=", "gamma=",
                                 "encounter-factor=", "mission-length=",
                                 "max-distance=", "galaxy-density=",
                                 "AGN-fraction=", "snr-cutoff=", "loops="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    # set default values
    i = 0
    output_filepath = '../output/AGN_LISA_results.h5'
    while os.path.isfile(output_filepath):
        output_filepath = '../output/AGN_LISA_results_{}.h5'.format(i)
        i += 1
    n_AGN = 200
    gamma = 1
    encounter_factor = 10
    t_obs = 4 * u.yr
    max_distance = 1 * u.Gpc
    galaxy_density = 4e6 * u.Gpc**(-3)
    AGN_fraction = 0.01
    snr_cutoff = 7
    loops = 10

    # change defaults based on input
    for option, value in opts:
        if option in ("-h", "--help"):
            usage()
            return
        elif option in ("-o", "--output"):
            output_filepath = value
        elif option in ("-n", "--n-AGN"):
            n_AGN = int(value)
        elif option in ("-g", "--gamma"):
            gamma = int(value)
        elif option in ("-e", "--encounter-factor"):
            encounter_factor = float(value)
        elif option in ("-m", "--mission-length"):
            t_obs = float(value) * u.yr
        elif option in ("-d", "--max-distance"):
            max_distance = float(value) * u.Gpc
        elif option in ("-G", "--galaxy-density"):
            galaxy_density = float(value) * u.Gpc**(-3)
        elif option in ("-a", "--AGN-fraction"):
            AGN_fraction = float(value)
        elif option in ("-s", "--snr-cutoff"):
            snr_cutoff = float(value)
        elif option in ("-l", "--loops"):
            loops = int(value)

    # check the file can be created without error
    with h5.File(output_filepath.replace(".h5", "_loop0.h5"), "w") as output:
        output["test"] = "TEST"

    for i in range(loops):
        results = simulate_LISA_AGN_rate(n_AGN=n_AGN, gamma=gamma,
                                         encounter_factor=encounter_factor,
                                         t_obs=t_obs,
                                         max_distance=max_distance,
                                         galaxy_density=galaxy_density,
                                         AGN_fraction=AGN_fraction,
                                         snr_cutoff=snr_cutoff)

        with h5.File(output_filepath.replace(".h5", "_loop{}.h5".format(i)),
                     "w") as output:
            output["m_1"] = results["m_1"]
            output["m_2"] = results["m_2"]
            output["a_enc"] = results["a_enc"]
            output["e_enc"] = results["e_enc"]
            output["e_LISA"] = results["e_LISA"]
            output["f_orb_LISA"] = results["f_orb_LISA"]
            output["dist"] = results["dist"]
            output["snr"] = results["snr"]
            output["max_snr_harmonic"] = results["max_snr_harmonic"]
            output["age"] = results["AGN_age"]
            output["t_e2m"] = results["t_e2m"]
            output["t_se"] = results["t_se"]
            output["m_oligarch_final"] = results["m_oligarch_final"]

            output.attrs["n_detect"] = results["n_detections"]
            output.attrs["gamma"] = gamma
            output.attrs["encounter_factor"] = encounter_factor
            output.attrs["t_obs"] = t_obs
            output.attrs["max_distance"] = max_distance
            output.attrs["galaxy_density"] = galaxy_density
            output.attrs["AGN_fraction"] = AGN_fraction
            output.attrs["snr_cutoff"] = snr_cutoff

        del results


if __name__ == "__main__":
    main()

import numpy as np
import h5py as h5
import astropy.units as u
import os.path

ids = 10
loops = 10
n_AGN = 5000
files = 0
base_filename = "BASE"

for i in range(ids):
    for j in range(loops):
        if os.path.isfile(base_filename + "_id{}_loop{}.h5".format(i, j)):
            files += 1

with h5.File(base_filename + ".h5", "w") as main:
    main["m_1"] = np.zeros(n_AGN * files) * u.Msun
    main["m_2"] = np.zeros(n_AGN * files) * u.Msun
    main["a_enc"] = np.zeros(n_AGN * files) * u.AU
    main["e_enc"] = np.zeros(n_AGN * files)
    main["e_LISA"] = np.zeros(n_AGN * files)
    main["f_orb_LISA"] = np.zeros(n_AGN * files) * u.Hz
    main["dist"] = np.zeros(n_AGN * files) * u.Gpc
    main["snr"] = np.zeros(n_AGN * files)
    main["max_snr_harmonic"] = np.zeros(n_AGN * files).astype(int)
    main["age"] = np.zeros(n_AGN * files) * u.yr
    main["t_e2m"] = np.zeros(n_AGN * files) * u.yr
    main["t_se"] = np.zeros(n_AGN * files) * u.yr
    main["m_oligarch_final"] = np.zeros(n_AGN * files) * u.Msun

    k = 0
    for i in range(ids):
        for j in range(loops):
            filename = base_filename + "_id{}_loop{}.h5".format(i, j)
            print(filename)
            if os.path.isfile(filename):
                with h5.File(filename, "r") as file:
                    main["m_1"][k * n_AGN: (k + 1) * n_AGN] = file["m_1"]
                    main["m_2"][k * n_AGN: (k + 1) * n_AGN] = file["m_2"]
                    main["a_enc"][k * n_AGN: (k + 1) * n_AGN] = file["a_enc"]
                    main["e_enc"][k * n_AGN: (k + 1) * n_AGN] = file["e_enc"]
                    main["e_LISA"][k * n_AGN: (k + 1) * n_AGN] = file["e_LISA"]
                    main["f_orb_LISA"][k * n_AGN: (k + 1) * n_AGN] = file["f_orb_LISA"]
                    main["dist"][k * n_AGN: (k + 1) * n_AGN] = file["dist"]
                    main["snr"][k * n_AGN: (k + 1) * n_AGN] = file["snr"]
                    main["max_snr_harmonic"][k * n_AGN: (k + 1) * n_AGN] = file["max_snr_harmonic"]
                    main["age"][k * n_AGN: (k + 1) * n_AGN] = file["age"]
                    main["t_e2m"][k * n_AGN: (k + 1) * n_AGN] = file["t_e2m"]
                    main["t_se"][k * n_AGN: (k + 1) * n_AGN] = file["t_se"]
                    main["m_oligarch_final"][k * n_AGN: (k + 1) * n_AGN] = file["m_oligarch_final"]
                k += 1
                print(k)

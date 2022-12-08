# Unit conversions
CMPS_2_KMPH = 3.6 / 100  # Convert [cm/s] to [km/h].
KMPH_2_CMPS = 1 / CMPS_2_KMPH  # Convert [km/h] to [cm/s].
CMPS_2_MPS = 0.01  # Convert [cm/s] to [m/s].
MPS_2_CMPS = 1 / CMPS_2_MPS  # Convert [km/h] to [cm/s].
KMPH_2_MPS = 10.0 / 36  # Convert [km/h] to [m/s].
MPS_2_KMPH = 3.6  # Convert [m/s] to [km/h].
M_2_CM = 100  # Convert [m] to [cm].
CM_2_M = 0.01  # Convert [cm] to [m].
M_2_KM = 0.001  # Convert [m] to [km].
CM_2_KM = CM_2_M * M_2_KM  # Convert [cm] to [km].
S_2_US = 1000000  # Convert [s] to [us].
US_2_S = 1e-6  # Convert [us] to [s].
NS_2_S = 1e-9  # Convert [ns] to [s].
MRAD_2_RAD = 1e-3  # Convert [mradian] to [radian].
S_2_MS = 1000  # Convert [s] to [ms].
MS_2_S = 1e-3  # Convert [ms] to [s].
MM_2_M = 0.001  # Convert [mm] to [m].
MM_2_CM = 0.1  # Convert [mm] to [cm].
RPM_2_RPS = 1 / 60  # Convert [rounds/min] to [rounds/s].
B_2_GB = 1 / (2**30)  # Convert [bytest] to [gigabytes]
HOUR_TO_S = 3600  # Convert [hour] to [s]
S_TO_HOUR = 1 / HOUR_TO_S  # Convert [s] to [hour]
MIN_2_S = 60  # Convert [minute] to [s]
B_2_MB = 1 / (2**20)  # Convert [bytes] to [megabytes]
KN_2_KPH = 1.852  # Convert [knots] to [km/h]
KPH_2_KN = 1 / KN_2_KPH  # Convert [km/h] to [knots]


def time_to_sec(time):
    """Interpret time string as float"""
    time = time.split(" ")[-1].split(":")
    return float(time[0]) * 3600 + float(time[1]) * 60 + float(time[2])

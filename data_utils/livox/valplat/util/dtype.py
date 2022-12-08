"""
Common numpy array types.
If a module needs additional fields in a dtype defined here, it should be added
on the module level. The definitions here shall be understood as common _base_
types. Different modules are free to extend these basic types according to
their need.
"""
import numpy as np

from valplat.util.custom_types import ExtendableEnum

odom_dtype = [
    ("t", "f8"),
    ("v", "f8"),
    ("yaw_rate", "f8"),
    ("transformation", "f8", (4, 4)),
]
pt_dtype = [
    ("x", "f4", (2,)),
    ("pulse_width", "f4"),
    ("layer_id", "i4"),
    ("z", "f4"),
    ("i", "f4"),
]
can_dtype = [("signal", "U60"), ("CAN_msg", "U60"), ("CAN_signal", "U60")]
gps_dtype = [
    ("latitude", "<f8"),
    ("longitude", "<f8"),
    ("velocity", "<f8"),
    ("satellites", "<i4"),
    ("streamtime", "<f8"),
    ("gpstime", "<f8"),
    ("mag_field_x", "<f8"),
    ("mag_field_y", "<f8"),
    ("mag_field_z", "<f8"),
    ("gps_pdop", "<f8"),
    ("gps_hdop", "<f8"),
    ("gps_vdop", "<f8"),
]
vlp32c_scan_dtype = [("pts", "O"), ("t", "f8"), ("streamtime", "f8")]
vlp32c_pt_dtype = [
    ("x", "f2"),
    ("y", "f2"),
    ("z", "f2"),
    ("i", "u1"),
    ("layer_id", "i2"),
    ("r", "f2"),
]
velo_pt_long_dtype = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("i", "u1"),
    ("layer_id", "i2"),
    ("r", "f4"),
    ("lidar", "i2"),
]
vlp32c_tele_dtype = [
    ("toh", "u4"),
    ("pps", "i1"),
    ("msg_type", "U6"),
    ("utc_time", "U9"),
    ("gps_valid", "U1"),
    ("latitude", "f8"),
    ("ns_indicator", "U1"),
    ("longitude", "f8"),
    ("ew_indicator", "U1"),
    ("speed", "f8"),
    ("cardinal_direction", "f2"),
    ("date", "i8"),
    ("mode", "U2"),
    ("checksum", "U2"),
]
obj_dtype = [
    ("t", "f8"),  # time in seconds
    ("frame", "i4"),  # reference to an external frame, project specific
    ("x", "f8", (4,)),  # position of the reference point with respect to ego rear axis
    ("bb", "f8", (4,)),  # offsets of the object sides from the reference point
    ("cls", "u8"),
]  # classification, project specific (to be improved)
X, Y, PHI, VEL = range(4)
rb_coord_dtype = [("x", "f8", (2,))]
kinem_dtype = [("v", "f8"), ("a", "f8"), ("o", "f8")]
line_dtype = [("pts", "O"), ("id", "u8"), ("type", "U20"), ("color", "U20")]
video_dtype = [("data", "O"), ("t", "f8")]
metadata_dtype = [
    ("gpstime_epoch", "f8"),  # UTC GPS time in epoch format
    ("temperature_c", "f4"),  # Temperature deg C
    ("weather", "b", 16),  # Boolean array of the weather condition
    ("light", "U50"),  # Default: 'day', 'night', 'dawn', 'dusk'
    ("road_type", "b", 32),  # Boolean array of the road condition
    ("num_lanes", "i4"),
    ("maxspeed_kph", "i4"),
]

metadata_global = {"test_type": "", "test_cycle_name": "", "vehicle_vin": ""}


# Enums 'weather' and 'road_type' provide description to 'b' metadata dtype.
class MetadataWeatherType(ExtendableEnum):
    rain = 0
    fog = 1
    snow = 2


class MetadataRoadType(ExtendableEnum):
    highway, urban, country, test_track, tunnel = np.arange(5)


class MetadataOSMRoadType(MetadataRoadType):
    (
        motorway,
        trunk,
        primary,
        secondary,
        tertiary,
        unclassified,
        residential,
        motorway_link,
        trunk_link,
        primary_link,
        secondary_link,
        tertiary_link,
    ) = (
        np.arange(12) + MetadataRoadType.enum_len()
    )

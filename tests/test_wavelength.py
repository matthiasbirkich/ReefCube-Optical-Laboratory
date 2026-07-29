from reefcube.wavelength import (
    AS7343_CHANNEL_ORDER,
    get_peak_wavelength_nm,
    channel_names_in_range,
)

print(AS7343_CHANNEL_ORDER)
print(get_peak_wavelength_nm("F4"))
print(channel_names_in_range(400, 700))
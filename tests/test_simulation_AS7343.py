from reefcube.sensors import SimulationSensor


for mode in ("sunny", "cloudy", "underwater"):
    sensor = SimulationSensor(
        mode=mode,
        seed=42,
    )

    measurement = sensor.acquire()

    print()
    print("=" * 50)
    print(mode.upper())
    print("=" * 50)

    print(measurement.summary())

    print()
    print("Spectral channels:")

    for channel_name, value in measurement.spectral_channels.items():
        print(f"{channel_name:>3}: {value:10.1f}")
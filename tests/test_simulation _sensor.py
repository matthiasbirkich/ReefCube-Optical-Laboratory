from reefcube.sensors import SimulationSensor

sensor = SimulationSensor()

measurement = sensor.acquire()

print(measurement.summary())
import sounddevice as sd
devices = sd.query_devices()
for i, device in enumerate(devices):
	if device['max_input_channels'] > 0:
		print(f"ID: {i} - Nome: {device['name']}")

import numpy as np

class Epoch:
	verbose = False
	def __init__(self, number, run, channels_dict = {}, verbose = False):
		self.channels_dict = channels_dict
		self.number = number
		self.run = run
		self.verbose = verbose

	def split_list(self, waveform, start_sample, end_sample, sub_epoch_width):
		sub_epochs = []
		for i in range(start_sample, end_sample, sub_epoch_width):
			sub_epoch = waveform[i:i + sub_epoch_width]
			sub_epochs.append(sub_epoch)
		if self.verbose:
			print(f"Epoch->split_list: length epoch {len(waveform)}")
			print(f"Epoch->split_list: length sub-epoch {len(sub_epochs[0])}")
		return np.array(sub_epochs)

	def split_all_waveforms(self, start_sample, end_sample, sub_epoch_width):
		for channel, data in self.channels_dict.items():
			if isinstance(data, np.ndarray):
				self.channels_dict[channel] = self.split_list(data, start_sample, end_sample, sub_epoch_width)
			elif isinstance(data, dict):
				for key, waveform in data.items():
					if isinstance(waveform, np.ndarray):
						data[key] = self.split_list(waveform, start_sample, end_sample, sub_epoch_width)
					elif isinstance(waveform, (list, tuple)):
						updated_waveforms = []
						for single_waveform in waveform:
							if isinstance(single_waveform, np.ndarray):
								split_waveform = self.split_list(single_waveform, start_sample, end_sample, sub_epoch_width)
								updated_waveforms.append(split_waveform)
						data[key] = np.array(updated_waveforms)

	def touch_raw_data(self, function_ptr, target_channels=None):
		for channel, data in self.channels_dict.items():
			if target_channels is None or channel in target_channels:
				if isinstance(data, np.ndarray):
					function_ptr(data)
				elif isinstance(data, dict):
					for key, waveform in data.items():
						if isinstance(waveform, np.ndarray):
							function_ptr(waveform)
						elif isinstance(waveform, (list, tuple)):
							for single_waveform in waveform:
								if isinstance(single_waveform, np.ndarray):
									function_ptr(single_waveform)
	def transform_raw_data(self, function_ptr, target_channels=None):
		for channel, data in self.channels_dict.items():
			if target_channels is None or channel in target_channels:
				if isinstance(data, np.ndarray):
					self.channels_dict[channel] = function_ptr(data)
				elif isinstance(data, dict):
					for key, waveform in data.items():
						if isinstance(waveform, np.ndarray):
							data[key] = function_ptr(waveform)
						elif isinstance(waveform, (list, tuple)):
							transformed_waveforms = []
							for single_waveform in waveform:
								if isinstance(single_waveform, np.ndarray):
									transformed_waveform = function_ptr(single_waveform)
									transformed_waveforms.append(transformed_waveform)
							data[key] = np.array(transformed_waveforms)

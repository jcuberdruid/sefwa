from cl.userExtension.epoch import Epoch

class Subject:
	def __init__(self, number, category):
		self.number = number
		self.category = category
		self.epochs = []
		self.waveforms = [] 
	def split_epochs(self, start_sample, end_sample, sub_epoch_width):
		for epoch in self.epochs:
			if isinstance(epoch, Epoch):
				epoch.split_all_waveforms(start_sample, end_sample, sub_epoch_width)
			else:
				print("Subject->split_epochs: epochs are not of type Epoch")	
	def touch_epoch_data(self, function_ptr, target_channels=None, return_high_level=False):
		for epoch in self.epochs:
			if isinstance(epoch, Epoch):
				epoch.touch_raw_data(function_ptr, target_channels, return_high_level)
			else:
				print("Subject->split_epochs: epochs are not of type Epoch")
	def transform_epoch_data(self, function_ptr, target_channels=None, return_high_level=False):
		for epoch in self.epochs:
			if isinstance(epoch, Epoch):
				epoch.transform_raw_data(function_ptr, target_channels, return_high_level)
			else:
				print("Subject->split_epochs: epochs are not of type Epoch")
	
	def gather_waveform(self, waveform):
		self.waveforms.append(waveform)
	def gather_all_waveforms(self, target_channels = None, return_high_level=False):
		gather_function = lambda waveform: self.gather_waveform(waveform)
		for epoch in self.epochs:
			epoch.touch_raw_data(gather_function, target_channels, return_high_level)
		

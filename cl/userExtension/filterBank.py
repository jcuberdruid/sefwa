import scipy.signal
from multiprocessing import Pool

from cl.userExtension.subject import Subject
from cl.userExtension.epoch import Epoch

class FilterBank:
	multi_processed_pool = 12
	sample_rate = 160
	def __init__(self, start_freq, end_freq, band_width, verbose=False):
		self.start_freq = start_freq
		self.end_freq = end_freq
		self.band_width = band_width
		self.verbose = verbose
	def filterBankChannel(self, waveform, sample_rate=None):
		if sample_rate is None:
			sample_rate = self.sample_rate
		filter_bank_dict = {}
		nyquist = sample_rate / 2
		for low_freq in range(self.start_freq, self.end_freq, self.band_width):
			high_freq = min(low_freq + self.band_width, self.end_freq)
			b, a = scipy.signal.butter(N=2, Wn=[low_freq / nyquist, high_freq / nyquist], btype='band')
			filtered_waveform = scipy.signal.lfilter(b, a, waveform)
			filter_bank_dict[f'{low_freq}-{high_freq}'] = filtered_waveform
		return filter_bank_dict
	def bankSingleSubject(self, subject):
		if self.verbose: 
			print(f"FilterBank->bankSingleSubject: beginning subject {subject.number}")
		for epoch in subject.epochs:
			for key, value in epoch.channels_dict.items():  # Assuming epoch is a dictionary-like object
				epoch.channels_dict[key] = self.filterBankChannel(value)
		return subject
	def bankSubjects(self, subject_list):
		with Pool(self.multi_processed_pool) as pool:
			banked_subject_list = pool.map(self.bankSingleSubject, subject_list)
		return banked_subject_list
'''
class OldFilterBank:
	multi_processed_pool = 12
	sample_rate = 160
	def __init__(start_freq, end_freq, band_width):
		self.start_freq = start_freq
		self.end_freq = end_freq
		self.band_width = band_width

	def filterBankChannel(self, waveform, sample_rate=self.sample_rate):
		filter_bank_dict = {}
		nyquist = sample_rate / 2
		for low_freq in range(self.start_freq, self.end_freq, self.band_width):
			high_freq = min(low_freq + self.band_width, self.end_freq)
			b, a = scipy.signal.butter(N=2, Wn=[low_freq / nyquist, high_freq / nyquist], btype='band')
			filtered_waveform = scipy.signal.lfilter(b, a, waveform)
			filter_bank_dict[f'{low_freq}-{high_freq}'] = filtered_waveform
		return filter_bank_dict
	
	def bankSingleSubject(subject):
		for epoch in subject.epochs:
			for value in epoch.values():
				value = filterBankChannel(values)
		return subject

	def bankSubjects(subject_list):			
		with Pool(self.multi_processed_pool) as pool:
			banked_subject_list = pool.map(bankSingleSubject, subject_list)	
		return banked_subject_list
'''

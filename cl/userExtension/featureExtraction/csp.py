from cl.userExtension import subject 
from cl.userExtension import epoch 
class CSP:
	def __init__(self, max_attempts = 10):
		self.filters = None
		self.max_attempts = max_attempts

	def make_filters(self, class_1, class_2, n__components = 8):
		if len(class_1) == len(class_2):
			print("Both classes are of the same length:", len(class_1))
		else:
			print("The classes have different lengths. Length of class_1:", len(class_1), "and length of class_2:", len(class_2))
			print(f"This may cause issues when creating filters, if it fails it will be retried {max_attempts} times")
		try:
			
			csp_model = self.train_csp_with_random_epoch_removal(class_1, class_2, n_components)
		except Exception as e:
			print(f"Error processing subject {subject}: {e}")

	def train_csp_with_random_epoch_removal(self, df_class1, df_class2, n_components):
		for attempt in range(self.max_attempts):
			try:
				if attempt > 0:
					random_epoch_class1 = random.choice(df_class1[['subject', 'run', 'epoch']].drop_duplicates().values.tolist())
					random_epoch_class2 = random.choice(df_class2[['subject', 'run', 'epoch']].drop_duplicates().values.tolist())

					df_class1 = df_class1[~((df_class1['subject'] == random_epoch_class1[0]) &
					(df_class1['run'] == random_epoch_class1[1]) &
					(df_class1['epoch'] == random_epoch_class1[2]))]

					df_class2 = df_class2[~((df_class2['subject'] == random_epoch_class2[0]) &
					(df_class2['run'] == random_epoch_class2[1]) &
					(df_class2['epoch'] == random_epoch_class2[2]))]

				csp_model = self.train_csp(df_class1, df_class2, n_components)
				return csp_model
			except np.linalg.LinAlgError:
				continue
			print(f"Failed to train CSP model after {max_attempts} attempts.")
			return None

		

# apply_filters(class_1, class_2) -> outputs CSP components 

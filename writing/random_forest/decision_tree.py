class DecisionTree(object):
	def __init__(self, min_samples_split=2, min_impurity=1e-7,
				 max_depth=float("inf")):
		self.root = None  # Root node in dec. tree
		# Minimum n of samples to justify split
		self.min_samples_split = min_samples_split
		# The minimum impurity to justify split
		self.min_impurity = min_impurity
		# The maximum depth to grow the tree to
		self.max_depth = max_depth
		# Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
		self._impurity_calculation = None
		# Function to determine prediction of y at leaf
		self._leaf_value_calculation = None
		# If y is nominal
		self.one_dim = None


	def build_tree(self, X, y, current_depth=0):
		largest_impurity = 0
		best_criteria = None    # Feature index and threshold
		best_sets = None        # Subsets of the data

		expand_needed = len(np.shape(y)) == 1
		if expand_needed:
			y = np.expand_dims(y, axis=1)

		# Add y as last column of X
		X_y = np.concatenate((X, y), axis=1)

		n_samples, n_features = np.shape(X)


def divide_on_feature(X, feature_index, threshold):
	split_func = None
	if isinstance(threshold, int) or isinstance(threshold, float):
		split_func = lambda sample: sample[feature_index] >= threshold
	else:
		split_func = lambda sample: sample[feature_index] == threshold

	X_1 = np.array([sample for sample in X if split_func(sample)])
	X_2 = np.array([sample for sample in X if not split_func(sample)])

	return np.array([X_1, X_2])


def calc_entropy(y):
	log2 = lambda x: np.log(x) / np.log(2)
	unique_labels = np.unique(y)
	entropy = 0
	for label in unique_labels:
		count = len(y[y == label])
		p = count / len(y)
		entropy += -p * log2(p)
	return entropy

def calc_information_gain(y, y_left, y_right):
	"""
	Calculates the information gained from splitting y into the subsets 
	y_let and y_right
	"""
	
	p_left = len(y_1) / len(y)
	p_right = 1 - p_left
	
	H_y = calc_entropy(y)
	H_y_left = calc_entropy(y_left)
	H_y_right = calc_entropy(y_right)
	
	information_gain = H_y - p_left * H_y_left - p_right * H_y_right 
	
	return information_gain 


def calc_split(X, y):
	if len(y.shape) == 1:
		y.shape = (len(y), 1)

	Xy = np.hstack([X, y])
	largest_info_gain = 0
	best_feature_index = None
	best_split = None
	 
	for feature_index in range(n_features):

		feature_values = X[:, feature_index]
		unique_values = np.unique(feature_values)

		for threshold in unique_values:

			#split data based on if feature_index is greater than or less than threshold 
			split_func = lambda sample: sample[feature_index] >= threshold
			Xy_left = np.array([sample for sample in Xy if split_func(sample)])
			Xy_right = np.array([sample for sample in Xy if not split_func(sample)])
			
			if len(Xy_left) > 0 and len(Xy_right) > 0:
				y_left = Xy_left[:, n_features:]
				y_right = Xy_right[:, n_features:]

				info_gain = calc_information_gain(y, y_left, y_right)


				# If this threshold resulted in a higher information gain than previously
				# recorded save the threshold value and the feature
				# index
				if info_gain > largest_info_gain:
					largest_info_gain = info_gain
					best_feature_index = feature_index
					best_split = (Xy_left, Xy_right)

	return best_feature_index, best_split, largest_info_gain
import numpy as np


class DTLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a Decision Tree Learner.
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    def __init__(self, leaf_size=1, verbose=False):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print("DTleaner - Tree shape: " + str(self.tree.shape))

    def query(self, points):
        """
        :summary: Estimate a set of test points given the model we built.
        :param points: should be a numpy array with each row corresponding to a specific query.
        :return: the estimated values according to the saved model.
        """
        
        n = points.shape[0]
        predictions = np.zeros([n])
        for i in range(n):
            predictions[i] = self.get_prediction(points[i,:], node=0)

        return predictions

    def get_prediction(self, point, node):
        """
        :summary: Estimate a set of test points given the model we built.
        :param point: a specific query
        :type point: numpy.ndarray
        :param node: the current node
        :type node: int
        :return: the estimated values according to the saved model.
        """
        feature = self.tree[node, 0]
        split_val = self.tree[node, 1]

        if feature == -1:
            return split_val

        if point[int(feature)] <= split_val:
            node += int(self.tree[node, 2])
        else:
            node += int(self.tree[node, 3])
        return self.get_prediction(point, node)


    def build_tree(self, data_x, data_y):

        """
        Build the Decision Tree.

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # -1 = N/A

        # the number of entries is less than or equal to leaf_size
        # aggregated all of the data left into a leaf
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), -1, -1]])

        # if all data.y same
        # aggregated all of the data left into a leaf
        if np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], -1, -1]])

        # determine best feature index to split on
        feature_idx = self.determine_best_feature(data_x, data_y)
        split_val = self.determine_split_val(data_x, feature_idx)

        # prevent infinite recursion when all the entries in the best feature
        # column are the same
        left_mask = data_x[:, feature_idx] <= split_val
        right_mask = data_x[:, feature_idx] > split_val
        if np.all(left_mask == left_mask[0]):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        root = np.array([[feature_idx, split_val, 1, left_tree.shape[0] + 1]])

        return np.row_stack((root, left_tree, right_tree))

    @staticmethod
    def determine_best_feature(data_x, data_y):

        """
        Determine the which column is the best factor/feature column by finding
        the best correlation between each factor column and label (i.e. data_y)

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        feature_idx = 0
        correlation = 0

        for i in range(data_x.shape[1]):
            corr = abs(np.corrcoef(data_x[:, i], data_y)[0, 1])

            if corr > correlation:
                correlation = corr
                feature_idx = i

        return feature_idx

    @staticmethod
    def determine_split_val(data_x, feature_idx):
        return np.median(data_x[:, feature_idx])


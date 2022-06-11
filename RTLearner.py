import numpy as np
import DTLearner as dtl


class RTLearner(dtl.DTLearner):
    """
    This is a  Decision Tree Learner with the choice of feature to split is made randomly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

  
    @staticmethod
    def determine_best_feature(data_x, data_y):
        # data_y is not used
        return np.random.randint(0, data_x.shape[1])

    @staticmethod
    def determine_split_val(data_x, feature_idx):

        r1 = np.random.randint(0, data_x.shape[0])
        r2 = np.random.randint(0, data_x.shape[0])
        split_val = (data_x[r1, feature_idx] + data_x[r2, feature_idx]) / 2

        return split_val

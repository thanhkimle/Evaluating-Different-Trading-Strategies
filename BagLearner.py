import numpy as np
from scipy.stats import mode


class BagLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Implement Bootstrap Aggregation as a Python class named BagLearner.
    This API is designed so that the BagLearner can accept any learner
    (e.g., RTLearner, LinRegLearner, even another BagLearner) as input
    and use it to generate a learner ensemble.

    The BagLearner constructor takes five arguments: learner, kwargs, bags, boost, and verbose.
    :param learners: The learner points to the learning class that will be used in the BagLearner.
    :param kwargs: keyword arguments that are passed on to the learner’s constructor and they can
        vary according to the learner (see example below).
    :param bags: This argument is the number of learners to train using Bootstrap Aggregation.
    :param boost: If "boost" is true, then you should implement boosting (optional implementation).
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learner = learner
        self.learners = []
        for i in range(bags):
            self.learners.append(self.learner(**kwargs))

    def add_evidence(self, data_x, data_y):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray

        """
        for learner in self.learners:
            rand_rows = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            sample_x = data_x[rand_rows]
            sample_y = data_y[rand_rows]
            learner.add_evidence(sample_x, sample_y)

    def query(self, points):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        predictions = np.zeros([self.bags, points.shape[0]])
        for i in range(self.bags):
            predictions[i] = self.learners[i].query(points)

        return mode(predictions)[0][0]


if __name__ == "__main__":
    pass
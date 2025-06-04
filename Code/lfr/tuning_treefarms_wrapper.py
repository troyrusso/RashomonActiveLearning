from lfr.treefarms_wrapper import Treefarms_LFR, _score, _predict
import pandas as pd
import numpy as np
import tqdm

class tuning_Treefarms_LFR(Treefarms_LFR): 
    '''
    Class to run online TreeFarms while actively tuning epsilon as each sample is updated
    '''

    def fit(self, X: pd.DataFrame, y: pd.Series, epsilon: float):
        '''
        treats epsilon as the max epsilon; tunes with that as upper bound
        '''
        if 'verbose' in self.static_config and self.static_config['verbose']: 
            print("Calling regular Treefarms_LFR with provided maximum value of {epsilon} for epsilon")
        super().fit(X, y, epsilon)
        self.all_accuracies = np.array([_score(tree, X, y) for tree in self.all_trees])
        self.accuracy_ordering = np.argsort(self.all_accuracies)

        predictions = np.zeros((X.shape[0], len(self.all_trees)))
        for j in range(len(self.all_trees)): 
            predictions[:, j] = _predict(self.all_trees[self.accuracy_ordering[j]], X).values

        # todo: improve efficiency of scan
        best_num_trees = 1
        best_acc = 0
        for i in range(len(self.all_trees)):
            # take the i+1 best trees
            # and compute the majority vote of those trees
            y_hat = predictions[:,:i+1].mean(axis=1)>0.5
            acc = np.mean(y_hat == y)
            if acc > best_acc: 
                best_num_trees = i+1
                best_acc = acc
        
        #compute trees in scope
        self.trees_in_scope = [self.all_trees[k] for k in self.accuracy_ordering[:best_num_trees]]
        #compute epsilon 
        if best_num_trees == len(self.all_trees): 
            self.epsilon = self.full_epsilon
        else:
            # if we don't include all trees, we set epsilon to be halfway between 
            # performance of last tree in scope and 
            # performance of first tree out of scope
            lower = self.all_accuracies[best_num_trees-1]
            upper = self.all_accuracies[best_num_trees]
            self.epsilon = (upper + lower) / 2 - self.all_accuracies[0]


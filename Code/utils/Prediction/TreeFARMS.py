# Summary: Initializes and fits a treefarms model.
# Input:
#   df_Train: The training data.
#   regularization: Penalty on the number of splits in a tree.
#   rashomon_bound_adder: A float indicating the Rashomon threshold: (1+\epsilon)*OptimalLoss
# Output:
# treeFarmsModel: A treefarms model.

### Libraries ###
from treeFarms.treefarms.model.treefarms import TREEFARMS

### Function ###
def TreeFarmsFunction(df_Train, regularization, rashomon_bound_adder):
    ### Train TreeFarms Model ###
    config = {"regularization": regularization, "rashomon_bound_adder": rashomon_bound_adder}
    TreeFarmsModel = TREEFARMS(config)
    TreeFarmsModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    
    ### Return ###
    return TreeFarmsModel
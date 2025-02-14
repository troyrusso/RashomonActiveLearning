# Summary: Initializes and fits a treefarms model.
# Input:
#   df_Train: The training data.
#   regularization: Penalty on the number of splits in a tree.
#   RashomonThreshold: A float indicating the Rashomon threshold: (1+\epsilon)*OptimalLoss
# Output:
# treeFarmsModel: A treefarms model.

### Libraries ###
from treeFarms.treefarms.model.treefarms import TREEFARMS

### Function ###
def TreeFarmsFunction(df_Train, regularization, RashomonThresholdType, RashomonThreshold):

    ## Adder ##
    if RashomonThresholdType == "Adder":
        config = {"regularization": regularization, "rashomon_bound_adder": RashomonThreshold}

    ## Multiplier ##
    if RashomonThresholdType == "Multiplier":
        config = {"regularization": regularization, "rashomon_bound_multiplier": RashomonThreshold}

    ## Train TreeFarms ##
    TreeFarmsModel = TREEFARMS(config)
    TreeFarmsModel.fit(df_Train.loc[:, df_Train.columns != "Y"], df_Train["Y"])
    
    ### Return ###
    return TreeFarmsModel
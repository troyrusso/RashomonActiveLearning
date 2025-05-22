# Summary: Initializes and fits a treefarms model.
# Input:
#   X_train_df: The training features (DataFrame).
#   y_train_series: The training target (Series).
#   regularization: Penalty on the number of splits in a tree.
#   RashomonThresholdType: Type of Rashomon threshold calculation ("Adder" or "Multiplier").
#   RashomonThreshold: A float indicating the Rashomon threshold: (1+\epsilon)*OptimalLoss or epsilon value.
# Output:
# treeFarmsModel: A treefarms model.

### Libraries ###
from treeFarms.treefarms.model.treefarms import TREEFARMS

### Function ###
def TreeFarmsFunction(X_train_df, y_train_series, regularization, RashomonThresholdType, RashomonThreshold, **kwargs):

    ## Configure TreeFarms ##
    config = {"regularization": regularization}
    if RashomonThresholdType == "Adder":
        config["rashomon_bound_adder"] = RashomonThreshold
    elif RashomonThresholdType == "Multiplier":
        config["rashomon_bound_multiplier"] = RashomonThreshold

    ## Train TreeFarms ##
    TreeFarmsModel = TREEFARMS(config)
    TreeFarmsModel.fit(X_train_df, y_train_series)
    
    ### Return ###
    return TreeFarmsModel
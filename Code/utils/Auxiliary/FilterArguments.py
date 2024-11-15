# Summary:
# Input:
# Output:

### Libraries ###
import inspect

def FilterArguments(Func, ArgumentDictionary):

    ### Set Up ###
    Signature = inspect.signature(Func)    
    FilteredArguments = {}
    
    ### Filter Arguments ###
    for ParameterName, _ in Signature.parameters.items():
        if ParameterName in ArgumentDictionary:
            FilteredArguments[ParameterName] = ArgumentDictionary[ParameterName]
    
    ### Return ###
    return FilteredArguments
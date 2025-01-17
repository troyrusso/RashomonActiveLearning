# import os
# import importlib

# # Get the current directory of the __init__.py file
# current_dir = os.path.dirname(__file__)

# # Loop through all files in the directory and import them dynamically
# for filename in os.listdir(current_dir):
#     # Only import .py files (and skip __init__.py and __utils__.py)
#     if filename.endswith(".py") and filename not in ["__init__.py", 
#                                                      "__utils__.py", 
#                                                     #  "ExtractError.py", 
#                                                      "CreateRunSimSbatch.py", 
#                                                     #  "CreateExtractSbatch.py", 
#                                                      "ProcessSimulationResults.py"
#                                                      ]:
#         module_name = filename[:-3]  # Remove the ".py" extension
#         module = importlib.import_module(f".{module_name}", package=__name__)  # Import the module
        
#         # Inject all functions and variables into the global namespace
#         globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith("__")})

# from .CreateRunSimSbatch import *
from .FilterArguments import *
from .LoadAnalyzedData import *
from .LoadDataSet import *
from .MakeSavePlots import *
from .MeanVariancePlot import *
from .PlotDecisionJsonTree import *
# from .ProcessSimulationResults import *
from .SelectionHistoryRank import *
from .WilcoxonRankSignedTest import *
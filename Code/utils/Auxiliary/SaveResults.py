# Summary:
# Input:
# Output:

### Libraries ###
import pickle

def SaveResults(data, filename):
    output_path = f"../Results/{filename}"
    with open(output_path, 'wb') as file:
        pickle.dump(data, file)
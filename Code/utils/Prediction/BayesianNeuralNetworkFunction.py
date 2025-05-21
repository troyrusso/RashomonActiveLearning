### Libraries ###
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

### Model Definition ###
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=50, dropout_rate=0.2):
        super(BayesianNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1) # For outputting probabilities

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) # Dropout applied during forward pass (inference too for MC Dropout)
        x = self.fc2(x)
        return x # Return logits before softmax for log_softmax later

### Training Function ###
def train_bnn(model, X_train_tensor, y_train_tensor, epochs=100, learning_rate=0.01, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train() # Set model to training mode (dropout is active)
    for epoch in range(epochs):
        # Shuffle and batch data
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

### Main Function for Integration ###
def BayesianNeuralNetworkFunction(X_train_df, y_train_series, Seed, **kwargs):
    
    # Extract X and Y (already separated, just convert to NumPy/Tensor)
    X_train_np = X_train_df.values
    y_train_np = y_train_series.values

    # Determine input size dynamically from the provided X_train_df
    input_size = X_train_np.shape[1] # THIS WILL NOW BE DYNAMIC TO THE DATASET

    # Determine number of classes dynamically
    num_classes = len(np.unique(y_train_np))

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)

    # Initialize model
    torch.manual_seed(Seed)
    bnn_model = BayesianNeuralNetwork(
        input_size=input_size, # Pass the dynamically determined input_size
        num_classes=num_classes,
        hidden_size=kwargs.get('hidden_size', 50),
        dropout_rate=kwargs.get('dropout_rate', 0.2)
    )

    # Train the model
    train_bnn(
        bnn_model,
        X_train_tensor,
        y_train_tensor,
        epochs=kwargs.get('epochs', 100),
        learning_rate=kwargs.get('learning_rate', 0.001),
        batch_size=kwargs.get('batch_size_train', 32) # Use batch_size_train
    )
    
    return bnn_model

# Prediction Method for BNN (to be called by TestErrorFunction and Selector)
def predict_proba_K_bnn(model, X_data_np, K_samples):
    
    # Set up 
    model.train()
    X_data_tensor = torch.tensor(X_data_np, dtype=torch.float32)

    # Create a tensor to store K sets of log probabilities
    N_data = X_data_tensor.shape[0]
    num_classes = model.fc2.out_features # Get number of output classes
    log_probs_N_K_C = torch.empty((N_data, K_samples, num_classes), dtype=torch.double)

    # No gradient calculation needed for inference
    with torch.no_grad(): 
        for k in range(K_samples):
            logits = model(X_data_tensor)
            log_softmax_output_k = torch.log_softmax(logits, dim=1)
            log_probs_N_K_C[:, k, :] = log_softmax_output_k

    return log_probs_N_K_C

BayesianNeuralNetwork.predict_proba_K = predict_proba_K_bnn
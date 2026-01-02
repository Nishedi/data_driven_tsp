"""
Neural Network for selecting Simulated Annealing parameters based on TSP instance features.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import pickle
import os


class ParameterPredictorNN(nn.Module):
    """Neural network that predicts SA parameters from TSP features."""
    
    def __init__(self, input_size: int = 7, hidden_size: int = 64):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
        """
        super(ParameterPredictorNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            
            nn.Linear(hidden_size // 2, 4)  # 4 parameters: temp, cooling, min_temp, iterations
        )
        
        # Output activation layers for each parameter
        self.temp_activation = nn.Softplus()  # Ensures positive temperature
        self.cooling_activation = nn.Sigmoid()  # Ensures 0 < cooling < 1
        self.min_temp_activation = nn.Softplus()  # Ensures positive min temp
        self.iterations_activation = nn.Softplus()  # Ensures positive iterations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Predicted parameters [initial_temp, cooling_rate, min_temp, iterations_per_temp]
        """
        output = self.network(x)
        
        # Apply specific activations and scaling to each parameter
        initial_temp = self.temp_activation(output[:, 0:1]) * 100 + 10  # Range: [10, ~110]
        cooling_rate = self.cooling_activation(output[:, 1:2]) * 0.1 + 0.9  # Range: [0.9, 1.0]
        min_temp = self.min_temp_activation(output[:, 2:3]) * 0.1  # Range: [0, ~0.1]
        iterations = self.iterations_activation(output[:, 3:4]) * 200 + 50  # Range: [50, ~250]
        
        return torch.cat([initial_temp, cooling_rate, min_temp, iterations], dim=1)


class ParameterPredictor:
    """Wrapper class for training and using the parameter prediction neural network."""
    
    def __init__(self, model_path: str = 'parameter_model.pth'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to save/load the model
        """
        self.model = ParameterPredictorNN()
        self.model_path = model_path
        self.scaler_mean = None
        self.scaler_std = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using fitted scaler."""
        if self.scaler_mean is None or self.scaler_std is None:
            return features
        return (features - self.scaler_mean) / (self.scaler_std + 1e-8)
    
    def fit_scaler(self, features: np.ndarray):
        """Fit feature scaler on training data."""
        self.scaler_mean = np.mean(features, axis=0)
        self.scaler_std = np.std(features, axis=0)
    
    def train(self, 
              features: List[np.ndarray], 
              optimal_params: List[np.ndarray],
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001):
        """
        Train the neural network.
        
        Args:
            features: List of feature vectors
            optimal_params: List of optimal parameter sets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(optimal_params)
        
        # Fit scaler
        self.fit_scaler(X)
        X = self.normalize_features(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict optimal parameters for a TSP instance.
        
        Args:
            features: Feature vector from TSP instance
            
        Returns:
            Predicted parameters [initial_temp, cooling_rate, min_temp, iterations_per_temp]
        """
        self.model.eval()
        with torch.no_grad():
            # Normalize features
            features_normalized = self.normalize_features(features.reshape(1, -1))
            X_tensor = torch.FloatTensor(features_normalized).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()[0]
    
    def save(self):
        """Save model and scaler to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load model and scaler from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Note: weights_only=False is required because we save numpy arrays (scaler_mean, scaler_std)
        # in the checkpoint along with model weights. This is safe as we control the checkpoint file.
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        print(f"Model loaded from {self.model_path}")

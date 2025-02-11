import numpy as np
from typing import List, Dict, Optional, Union, Any
from numpy.typing import NDArray

class EnsembleAverager:
    """Averages predictions from multiple models with proper normalization"""
    
    def __init__(self, method: str = 'mean'):
        """
        Initialize ensemble averager.
        
        Args:
            method: Averaging method ('mean' or 'weighted')
        """
        self.method = method
        self.models: List[Any] = []
        self.weights: List[float] = []
    
    def add_model(self, model: Any, weight: float = 1.0):
        """
        Add a trained model to the ensemble.
        
        Args:
            model: Trained classifier with predict_proba method
            weight: Weight for this model's predictions
        """
        self.models.append(model)
        self.weights.append(weight)
    
    def _normalize_probabilities(self, probs: NDArray) -> NDArray:
        """
        Normalize probabilities to sum to 1 along class axis.
        
        Args:
            probs: Probability array (n_samples, n_classes)
            
        Returns:
            NDArray: Normalized probabilities
        """
        return probs / np.sum(probs, axis=1, keepdims=True)
    
    def __call__(self, X: NDArray) -> NDArray:
        """
        Get normalized ensemble predictions for input X.
        
        Args:
            X: Input features
            
        Returns:
            NDArray: Normalized averaged probabilities (n_samples, n_classes)
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            probs = model.predict_proba(X)
            predictions.append(probs)
        
        # Stack predictions
        all_probs = np.stack(predictions, axis=0)
        
        # Average predictions
        if self.method == 'mean':
            avg_probs = np.mean(all_probs, axis=0)
        elif self.method == 'weighted':
            weights = np.array(self.weights)
            weights = weights / weights.sum()
            avg_probs = np.average(all_probs, axis=0, weights=weights)
        
        # Normalize averaged probabilities
        return self._normalize_probabilities(avg_probs)
    
    def get_statistics(self, X: NDArray) -> Dict[str, NDArray]:
        """
        Get normalized prediction statistics for input X.
        
        Args:
            X: Input features
            
        Returns:
            dict: Contains normalized mean, scaled std, min, max of predictions
        """
        predictions = []
        for model in self.models:
            probs = model.predict_proba(X)
            predictions.append(probs)
            
        all_probs = np.stack(predictions, axis=0)
        
        # Calculate mean and normalize
        mean_probs = np.mean(all_probs, axis=0)
        norm_mean = self._normalize_probabilities(mean_probs)
        
        # Calculate std and scale it according to normalization
        std_probs = np.std(all_probs, axis=0)
        scaling_factors = 1.0 / np.sum(mean_probs, axis=1, keepdims=True)
        scaled_std = std_probs * scaling_factors
        
        # Calculate min/max and normalize
        min_probs = self._normalize_probabilities(np.min(all_probs, axis=0))
        max_probs = self._normalize_probabilities(np.max(all_probs, axis=0))
        
        return {
            'mean': norm_mean,
            'std': scaled_std,
            'min': min_probs,
            'max': max_probs
        }
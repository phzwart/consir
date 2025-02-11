import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from numpy.typing import NDArray

class BalancedConformalPredictor:
    """
    Balanced conformal predictor that handles class imbalance
    by normalizing probabilities with class priors.
    """
    
    def __init__(
        self, 
        confidence_level: float = 0.95,
        class_priors: Optional[NDArray] = None
    ):
        """
        Initialize balanced conformal predictor.
        
        Args:
            confidence_level: Desired confidence level (e.g., 0.95 for 95%)
            class_priors: Prior probabilities for each class. If None,
                         will be estimated from calibration data.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
            
        self.confidence_level = confidence_level
        self.class_priors = class_priors
        self.thresholds: Dict[int, float] = {}
        self.is_calibrated = False
    
    def normalize_probabilities(
        self, 
        probabilities: NDArray,
        epsilon: float = 1e-10
    ) -> NDArray:
        """
        Normalize probabilities by dividing by class priors.
        
        Args:
            probabilities: Raw probabilities (n_samples, n_classes)
            epsilon: Small constant to avoid division by zero
            
        Returns:
            NDArray: Normalized probabilities
        """
        normalized = probabilities / (self.class_priors + epsilon)
        return normalized / np.sum(normalized, axis=1, keepdims=True)
    
    def compute_nonconformity_scores(
        self,
        probabilities: NDArray,
        true_labels: NDArray
    ) -> Dict[int, NDArray]:
        """
        Compute nonconformity scores for each class.
        
        Args:
            probabilities: Normalized probabilities (n_samples, n_classes)
            true_labels: True class labels
            
        Returns:
            Dict[int, NDArray]: Nonconformity scores per class
        """
        scores = {}
        for class_idx in range(probabilities.shape[1]):
            class_mask = (true_labels == class_idx)
            if not np.any(class_mask):
                continue
                
            # For each class, compute scores as negative log probability
            class_probs = probabilities[class_mask, class_idx]
            scores[class_idx] = -np.log(class_probs + 1e-10)
            
        return scores
    
    def calibrate(
        self, 
        cal_probabilities: NDArray, 
        cal_labels: NDArray
    ) -> 'BalancedConformalPredictor':
        """
        Calibrate predictor using normalized probabilities.
        
        Args:
            cal_probabilities: Calibration set probabilities (n_samples, n_classes)
            cal_labels: True labels for calibration set
        """
        # Estimate class priors if not provided
        if self.class_priors is None:
            class_counts = np.bincount(cal_labels)
            self.class_priors = class_counts / len(cal_labels)
        
        # Normalize probabilities
        normalized_probs = self.normalize_probabilities(cal_probabilities)
        
        # Compute nonconformity scores
        scores = self.compute_nonconformity_scores(normalized_probs, cal_labels)
        
        # Compute class-specific thresholds
        for class_idx, class_scores in scores.items():
            n_samples = len(class_scores)
            threshold_idx = int(np.ceil(n_samples * self.confidence_level)) - 1
            threshold_idx = min(threshold_idx, n_samples - 1)
            
            sorted_scores = np.sort(class_scores)
            self.thresholds[class_idx] = sorted_scores[threshold_idx]
        
        self.is_calibrated = True
        return self
    
    def __call__(self, probabilities: NDArray) -> Tuple[List[Set[int]], NDArray]:
        """
        Get prediction sets using normalized probabilities.
        
        Args:
            probabilities: Raw probabilities (n_samples, n_classes)
            
        Returns:
            Tuple[List[Set[int]], NDArray]: Prediction sets and normalized probabilities
        """
        if not self.is_calibrated:
            raise RuntimeError("Predictor must be calibrated before prediction")
        
        # Normalize probabilities
        normalized_probs = self.normalize_probabilities(probabilities)
        
        n_samples = len(normalized_probs)
        prediction_sets = []
        
        for i in range(n_samples):
            pred_set = set()
            
            # Check each class
            for class_idx, threshold in self.thresholds.items():
                score = -np.log(normalized_probs[i, class_idx] + 1e-10)
                if score <= threshold:
                    pred_set.add(class_idx)
            
            # If empty prediction set, add highest probability class
            if not pred_set:
                pred_set.add(np.argmax(normalized_probs[i]))
            
            prediction_sets.append(pred_set)
        
        return prediction_sets, normalized_probs
    
    def evaluate_coverage(
        self,
        probabilities: NDArray,
        true_labels: NDArray
    ) -> Dict[int, float]:
        """
        Evaluate per-class coverage.
        
        Args:
            probabilities: Test set probabilities
            true_labels: True labels
            
        Returns:
            Dict[int, float]: Coverage per class
        """
        prediction_sets, _ = self(probabilities)
        coverage = {}
        
        for class_idx in range(probabilities.shape[1]):
            class_mask = (true_labels == class_idx)
            if not np.any(class_mask):
                continue
                
            class_indices = np.where(class_mask)[0]
            correct = sum(
                class_idx in prediction_sets[i] 
                for i in class_indices
            )
            coverage[class_idx] = correct / len(class_indices)
            
        return coverage 
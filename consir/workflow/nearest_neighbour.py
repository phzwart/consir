import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Union
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


class NearestNeighbourClassifier:
    """K-Nearest Neighbors classifier with probability estimates"""
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'distance',
        metric: str = 'minkowski',
        p: int = 2,
        random_state: Optional[int] = None
    ):
        """
        Initialize KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric to use
            p: Power parameter for Minkowski metric
            random_state: Random state for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=-1  # Use all CPU cores
        )
        
    def fit(self, X: NDArray, y: NDArray) -> 'NearestNeighbourClassifier':
        """
        Fit the classifier.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self: The fitted classifier
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the classifier
        self.classifier.fit(X_scaled, y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        return self
    
    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities for X.
        
        Args:
            X: Input samples
            
        Returns:
            NDArray: Class probabilities (n_samples, n_classes)
        """
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Get probability estimates
        probabilities = self.classifier.predict_proba(X_scaled)
        
        return probabilities
    
    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class labels for X.
        
        Args:
            X: Input samples
            
        Returns:
            NDArray: Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
    
    def get_neighbors(
        self, 
        X: NDArray, 
        n_neighbors: Optional[int] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Find nearest neighbors of input samples.
        
        Args:
            X: Input samples
            n_neighbors: Number of neighbors (defaults to self.n_neighbors)
            
        Returns:
            Tuple[NDArray, NDArray]: (distances, indices)
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.kneighbors(
            X_scaled, 
            n_neighbors=n_neighbors,
            return_distance=True
        )

class LocalClassProbability:
    """Local class probability estimator using KDTree for efficient neighbor search"""
    
    def __init__(
        self,
        radius: float = 1.0,
        min_points: int = 5,
        leaf_size: int = 40
    ):
        """
        Initialize local probability estimator.
        
        Args:
            radius: Initial radius for neighbor search
            min_points: Minimum number of neighbors required
            leaf_size: Number of points at which to switch to brute-force
        """
        self.radius = radius
        self.min_points = min_points
        self.leaf_size = leaf_size
        self.scaler = StandardScaler()
        
        # Initialize as None
        self.tree: Optional[cKDTree] = None
        self.X: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.n_classes: Optional[int] = None
        self.class_labels: Optional[NDArray] = None
    
    def fit(self, X: NDArray, y: NDArray) -> 'LocalClassProbability':
        """
        Fit the local probability estimator.
        
        Args:
            X: Training data
            y: Target values
        """
        # Scale features
        self.X = self.scaler.fit_transform(X)
        self.y = np.asarray(y)
        
        # Build KD-tree
        self.tree = cKDTree(
            self.X,
            leafsize=self.leaf_size
        )
        
        # Get class information
        self.class_labels = np.unique(self.y)
        self.n_classes = len(self.class_labels)
        
        return self
    
    def _get_neighbors(self, x: NDArray, radius: float) -> Tuple[NDArray, NDArray]:
        """Get neighbors within radius using KDTree."""
        if self.tree is None or self.X is None:
            raise RuntimeError("Estimator must be fitted before prediction")
            
        # Find points within radius
        indices = self.tree.query_ball_point(x, radius, workers=-1)
        indices = np.array(indices[0])  # First point only
        
        if len(indices) == 0:
            return np.array([]), np.array([])
            
        # Calculate distances
        distances = np.sqrt(np.sum((self.X[indices] - x) ** 2, axis=1))
        return distances, indices
    
    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities for X."""
        if self.tree is None or self.y is None:
            raise RuntimeError("Estimator must be fitted before prediction")
        
        # Scale input data
        X_scaled = self.scaler.transform(X)
        
        # Initialize probabilities
        n_samples = len(X)
        probabilities = np.zeros((n_samples, self.n_classes))
        
        # Calculate probabilities for each point
        for i in range(n_samples):
            current_radius = self.radius
            max_radius = self.radius * 10
            
            # Find enough neighbors
            while current_radius <= max_radius:
                distances, indices = self._get_neighbors(
                    X_scaled[i:i+1],
                    current_radius
                )
                
                if len(indices) >= self.min_points:
                    break
                    
                current_radius *= 2
            
            if len(indices) > 0:
                # Get labels and calculate weights
                neighbor_labels = self.y[indices]
                weights = 1 / (distances + 1e-10)
                weights /= weights.sum()
                
                # Calculate weighted probabilities
                for j, class_label in enumerate(self.class_labels):
                    class_mask = (neighbor_labels == class_label)
                    if np.any(class_mask):
                        probabilities[i, j] = np.sum(weights[class_mask])
            
            # Handle empty neighborhoods
            if np.sum(probabilities[i]) == 0:
                probabilities[i] = 1 / self.n_classes
            else:
                probabilities[i] /= np.sum(probabilities[i])
        
        return probabilities
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for X."""
        probabilities = self.predict_proba(X)
        return self.class_labels[np.argmax(probabilities, axis=1)]



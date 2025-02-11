import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.stats import uniform, loguniform

def train_svm_classifier(points, classes, n_iter=100, cv=5, random_state=None, gamma_range=(1e-2, 1e1)):
    """
    Train an SVM classifier with hyperparameter tuning using random search cross-validation.
    
    Args:
        points (ndarray): Array of points, shape (n_samples, n_features)
        classes (ndarray): Array of class labels
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 20.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to None.
    
    Returns:
        tuple: (trained_svm, scaler) - The trained SVM classifier and the fitted scaler
    """
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(points)
    
    # Define the parameter space
    param_distributions = {
        'C': loguniform(1e-6, 1e3),
        'gamma': loguniform(gamma_range[0], gamma_range[1]),
        'kernel': ['rbf'],  # Can add more kernels if needed
        'probability': [True],  # We need probability estimates
        'class_weight': ['balanced', None],
    }
    
    # Initialize the base classifier
    base_svm = SVC(random_state=random_state)
    
    # Set up the random search
    random_search = RandomizedSearchCV(
        base_svm,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,  # Use all available cores
        random_state=random_state,
        verbose=1
    )
    
    # Fit the random search
    random_search.fit(X_scaled, classes)
    
    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score:", random_search.best_score_)
    
    return random_search.best_estimator_, scaler

def calculate_class_probabilities(svm, scaler, points):
    """
    Calculate class probabilities for new points using the trained SVM.
    
    Args:
        svm: Trained SVM classifier
        scaler: Fitted StandardScaler
        points (ndarray): Points to evaluate, shape (n_samples, n_features)
    
    Returns:
        ndarray: Class probabilities for each point
    """
    X_scaled = scaler.transform(points)
    return svm.predict_proba(X_scaled)

def cross_validate_performance(points, classes, n_splits=5, test_size=0.2, random_state=None):
    """
    Evaluate SVM performance using random subselections for cross-validation.
    
    Args:
        points (ndarray): Array of points
        classes (ndarray): Array of class labels
        n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to None.
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    from sklearn.model_selection import ShuffleSplit
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    
    cv = ShuffleSplit(
        n_splits=n_splits, 
        test_size=test_size, 
        random_state=random_state
    )
    
    scores = {
        'balanced_accuracy': [],
        'roc_auc': []
    }
    
    for train_idx, test_idx in cv.split(points):
        # Split data
        X_train, X_test = points[train_idx], points[test_idx]
        y_train, y_test = classes[train_idx], classes[test_idx]
        
        # Train model
        svm, scaler = train_svm_classifier(
            X_train, 
            y_train, 
            n_iter=10,  # Reduced iterations for CV
            random_state=random_state
        )
        
        # Calculate probabilities
        y_pred = svm.predict(scaler.transform(X_test))
        y_prob = svm.predict_proba(scaler.transform(X_test))
        
        # Calculate metrics
        scores['balanced_accuracy'].append(
            balanced_accuracy_score(y_test, y_pred)
        )
        scores['roc_auc'].append(
            roc_auc_score(y_test, y_prob[:, 1])
        )
    
    # Calculate mean and std for each metric
    return {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric, values in scores.items()
    }



class BaseClassifier:
    """Base class for all classifiers"""
    def __init__(self, random_state=None):
        self.random_state = random_state
        
    def fit(self, X, y):
        raise NotImplementedError
        
    def predict_proba(self, X):
        raise NotImplementedError

class SVMEnsembleClassifier(BaseClassifier):
    """Ensemble of SVM classifiers"""
    def __init__(self, n_splits=5, n_iter=10, gamma_range=(0.1, 2.0), random_state=None):
        super().__init__(random_state)
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.gamma_range = gamma_range
        self.models = []
        self.scalers = []
        
    def fit(self, X, y):
        """Fit ensemble of SVM models"""
        for split in range(self.n_splits):
            # Train model
            svm, scaler = train_svm_classifier(
                X, y, 
                n_iter=self.n_iter,
                random_state=self.random_state,
                gamma_range=self.gamma_range
            )
            self.models.append(svm)
            self.scalers.append(scaler)
        return self
        
    def predict_proba(self, X):
        """Average predictions from ensemble"""
        probas = []
        for svm, scaler in zip(self.models, self.scalers):
            proba = calculate_class_probabilities(svm, scaler, X)
            probas.append(proba)
        return np.mean(probas, axis=0)

class ConformalPredictor:
    """Class for conformal prediction"""
    def __init__(self, confidence_levels=[0.95], random_state=None):
        self.confidence_levels = confidence_levels
        self.random_state = random_state
        self.ensemble_scores = []
        self.unique_classes = None
        
    def calibrate(self, cal_probabilities, cal_classes):
        """Calibrate using provided probabilities"""
        self.unique_classes = np.unique(cal_classes)
        
        nonconf_scores = {}
        for class_label in self.unique_classes:
            class_mask = (cal_classes == class_label)
            if np.any(class_mask):
                class_idx = np.where(self.unique_classes == class_label)[0][0]
                scores = 1 - cal_probabilities[class_mask, class_idx]
                nonconf_scores[class_label] = scores
        
        self.ensemble_scores.append({'cal_scores': nonconf_scores})
        return self
        
    def predict(self, probabilities):
        """Generate prediction sets for new probabilities"""
        if not self.ensemble_scores:
            raise ValueError("Predictor not calibrated")
            
        results = {}
        for confidence in self.confidence_levels:
            prediction_sets = []
            
            for i in range(len(probabilities)):
                class_thresholds = self._calculate_thresholds(confidence)
                pred_set = self._get_prediction_set(probabilities[i], class_thresholds)
                prediction_sets.append(sorted(pred_set))
                
            results[confidence] = prediction_sets
            
        return results, probabilities
    
    def _calculate_thresholds(self, confidence):
        """Calculate class-specific thresholds"""
        class_thresholds = {}
        for class_label in self.unique_classes:
            thresholds = []
            for member in self.ensemble_scores:
                if class_label in member['cal_scores']:
                    scores = member['cal_scores'][class_label]
                    threshold_idx = int(np.ceil(len(scores) * confidence))
                    if threshold_idx >= len(scores):
                        threshold_idx = len(scores) - 1
                    threshold = np.sort(scores)[threshold_idx]
                    thresholds.append(threshold)
            if thresholds:
                class_thresholds[class_label] = np.mean(thresholds)
        return class_thresholds
    
    def _get_prediction_set(self, probs, class_thresholds):
        """Get prediction set for single point"""
        pred_set = []
        for j, class_label in enumerate(self.unique_classes):
            if class_label in class_thresholds:
                nonconf_score = 1 - probs[j]
                if nonconf_score <= class_thresholds[class_label]:
                    pred_set.append(class_label)
        
        if not pred_set:
            pred_set = [self.unique_classes[np.argmax(probs)]]
        
        return pred_set

class ConformalClassifier:
    """Combines classifier and conformal predictor"""
    def __init__(self, base_classifier, confidence_levels=[0.95], random_state=None):
        self.base_classifier = base_classifier
        self.conformal_predictor = ConformalPredictor(confidence_levels, random_state)
        self.random_state = random_state
        
    def fit(self, X, y, cal_X=None, cal_y=None):
        """Fit classifier and calibrate conformal predictor"""
        if cal_X is None or cal_y is None:
            # Split data if calibration set not provided
            train_idx, cal_idx = stratified_split(X, y, random_state=self.random_state)
            cal_X, cal_y = X[cal_idx], y[cal_idx]
            X, y = X[train_idx], y[train_idx]
            
        # Fit classifier
        self.base_classifier.fit(X, y)
        
        # Calibrate predictor
        cal_probs = self.base_classifier.predict_proba(cal_X)
        self.conformal_predictor.calibrate(cal_probs, cal_y)
        return self
        
    def predict(self, X):
        """Make predictions with uncertainty"""
        probabilities = self.base_classifier.predict_proba(X)
        return self.conformal_predictor.predict(probabilities)

def get_points_with_labels(prediction_sets, confidence_level, target_labels):
    """
    Extract points that have specific labels in their prediction sets.
    
    Args:
        prediction_sets (dict): Dictionary of prediction sets per confidence level
        confidence_level (float): Confidence level to use
        target_labels: Labels to search for
    
    Returns:
        list: Indices of points containing all target_labels
    """
    # Get prediction sets for specified confidence level
    sets = prediction_sets[confidence_level]
    
    # Find indices where all target_labels appear in prediction set
    indices = [i for i, pred_set in enumerate(sets) if all(label in pred_set for label in target_labels)]
    
    return indices
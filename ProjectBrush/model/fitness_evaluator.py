import numpy as np

class FitnessEvaluator:
    """
    Evaluates pattern fitness for NEAT evolution with robust error handling.
    Supports hybrid objective/subjective scoring.
    """

    def __init__(self, criteria_weights=None, user_weight=0.3):
        """
        Initializes fitness evaluator with configurable weights.
        
        Args:
            criteria_weights (dict): Objective criteria weights
            user_weight (float): Weight for subjective user scores (0-1)
        """
        self.criteria_weights = criteria_weights or {
            "symmetry": 0.4,
            "complexity": 0.3,
            "contrast": 0.3
        }
        self.user_weight = max(0, min(1, user_weight))

    def evaluate(self, pattern, user_feedback=None):
        """
        Main evaluation method combining objective and subjective scores.
        
        Args:
            pattern (Pattern): Pattern to evaluate
            user_feedback (float): Optional user rating (0-1)
            
        Returns:
            float: Combined fitness score
        """
        objective = self.evaluate_objective(pattern)
        
        if user_feedback is not None:
            subjective = self.evaluate_subjective(user_feedback)
            return self.combine_scores(objective, subjective)
        return objective

    def evaluate_objective(self, pattern):
        """Calculate objective fitness score with error handling"""
        if pattern.canvas is None:
            return 0.0
            
        try:
            symmetry = self._evaluate_symmetry(pattern.canvas)
            complexity = self._evaluate_complexity(pattern.canvas)
            contrast = self._evaluate_contrast(pattern.canvas)
            
            return (self.criteria_weights["symmetry"] * symmetry +
                    self.criteria_weights["complexity"] * complexity +
                    self.criteria_weights["contrast"] * contrast)
                    
        except Exception as e:
            print(f"Fitness evaluation error: {e}")
            return 0.0

    def evaluate_subjective(self, user_feedback):
        """Process user rating with sanity checks"""
        return max(0.0, min(1.0, float(user_feedback)))

    def combine_scores(self, objective_score, subjective_score):
        """Combine scores using configured weights"""
        objective = max(0.0, min(1.0, objective_score))
        subjective = max(0.0, min(1.0, subjective_score))
        return (1 - self.user_weight) * objective + self.user_weight * subjective

    def _evaluate_symmetry(self, canvas):
        """Calculate symmetry score with input validation"""
        try:
            assert canvas.ndim == 2, "Canvas must be 2D array"
            h, w = canvas.shape
            half = w // 2
            left = canvas[:, :half]
            right = np.flip(canvas[:, half + w%2:], axis=1)
            return 1 - np.mean(np.abs(left - right))
        except Exception as e:
            print(f"Symmetry evaluation failed: {e}")
            return 0.0

    def _evaluate_complexity(self, canvas):
        """Calculate complexity using normalized entropy"""
        try:
            flattened = canvas.flatten()
            hist = np.histogram(flattened, bins=256, range=(0, 1))[0] + 1e-7
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            return entropy / np.log2(256)  # Normalize to [0,1]
        except Exception as e:
            print(f"Complexity evaluation failed: {e}")
            return 0.0

    def _evaluate_contrast(self, canvas):
        """Calculate contrast with minimum threshold"""
        try:
            contrast = np.std(canvas)
            return max(contrast, 0.1)  # Ensure minimum contrast
        except Exception as e:
            print(f"Contrast evaluation failed: {e}")
            return 0.1  # Return minimum value
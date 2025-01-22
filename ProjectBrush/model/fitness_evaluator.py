import numpy as np

class FitnessEvaluator:
    """
    Evaluates the fitness of a pattern based on objective and subjective criteria.
    """

    def __init__(self, criteria_weights=None):
        """
        Initializes the fitness evaluator with optional weights for different criteria.

        Args:
            criteria_weights (dict, optional): Weights for objective criteria.
                Example: {"symmetry": 0.4, "complexity": 0.3, "contrast": 0.3}
        """
        # Default weights for different objective criteria
        self.criteria_weights = criteria_weights or {
            "symmetry": 0.4,
            "complexity": 0.3,
            "contrast": 0.3
        }

    def evaluate_objective(self, pattern):
        symmetry_score = max(self._evaluate_symmetry(pattern.canvas), 0)
        complexity_score = max(self._evaluate_complexity(pattern.canvas), 0)
        contrast_score = max(self._evaluate_contrast(pattern.canvas), 0)
        
        objective_score = (
            self.criteria_weights["symmetry"] * symmetry_score +
            self.criteria_weights["complexity"] * complexity_score +
            self.criteria_weights["contrast"] * contrast_score
        )
        return objective_score

    def evaluate_subjective(self, user_feedback):
        """
        Adjusts the fitness score based on user feedback.

        Args:
            user_feedback (float): User rating (e.g., 0-1 scale).

        Returns:
            float: Subjective fitness score.
        """
        return user_feedback  # Assume direct mapping for now

    def combine_scores(self, objective_score, subjective_score, user_weight=0.5):
        """
        Combines objective and subjective scores into a composite score.

        Args:
            objective_score (float): Fitness score from objective evaluation.
            subjective_score (float): Fitness score from user feedback.
            user_weight (float): Weight assigned to subjective scores (default: 0.5).

        Returns:
            float: Composite fitness score.
        """
        return (1 - user_weight) * objective_score + user_weight * subjective_score

    def _evaluate_symmetry(self, canvas):
        """
        Evaluates the symmetry of a pattern.

        Args:
            canvas (numpy array): The artwork's canvas.

        Returns:
            float: Symmetry score (0 to 1).
        """
        # Compare left and right halves
        half_width = canvas.shape[1] // 2
        left_half = canvas[:, :half_width]
        right_half = np.flip(canvas[:, half_width:], axis=1)
        return 1 - np.mean(np.abs(left_half - right_half))

    def _evaluate_complexity(self, canvas):
        """
        Evaluates the complexity of a pattern.

        Args:
            canvas (numpy array): The artwork's canvas.

        Returns:
            float: Complexity score (0 to 1).
        """
        # Complexity can be measured using entropy
        flattened = canvas.flatten()
        histogram, _ = np.histogram(flattened, bins=256, range=(0, 1), density=True)
        entropy = -np.sum(histogram * np.log2(histogram + 1e-7))  # Add small value to avoid log(0)
        return entropy / np.log2(len(histogram))  # Normalize to range [0, 1]

    def _evaluate_contrast(self, canvas):
        # Add minimum contrast threshold
        contrast = np.std(canvas)
        return max(contrast, 0.1)

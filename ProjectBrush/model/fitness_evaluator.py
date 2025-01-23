import numpy as np
from model.fitness_utils import FitnessUtils

class FitnessEvaluator:
    def __init__(self, user_weight=0.3):
        self.generation = 0
        self.archive = []
        self.user_weight = max(0.0, min(1.0, user_weight))

    def evaluate(self, pattern, user_feedback=None):
        objective = self.evaluate_objective(pattern)
        if user_feedback is not None:
            return self.combine_scores(objective, self.evaluate_subjective(user_feedback))
        return objective

    def evaluate_objective(self, pattern):
        """Main entry point for phased evaluation"""
        if not isinstance(pattern.canvas, np.ndarray):
            return 0.0
            
        if self.generation <= 50:
            return self._phase1_symmetry_contrast(pattern)
        elif 51 <= self.generation <= 150:
            return self._phase2_composition(pattern)
        else:
            return self._phase3_novelty(pattern)
    
    def _phase1_symmetry_contrast(self, pattern):
        symmetry = self._evaluate_symmetry(pattern.canvas)
        contrast = self._evaluate_contrast(pattern.canvas)
        active_area = np.mean(pattern.canvas > 0.1)  # Use element-wise comparison
        return 0.4*symmetry + 0.4*contrast + 0.2*active_area

    def _phase2_composition(self, pattern):
        phase1_score = self._phase1_symmetry_contrast(pattern)
        edge_density = FitnessUtils.detect_edge_density(pattern.canvas)
        color_coherence = FitnessUtils.calculate_color_coherence(pattern.canvas)
        return 0.6*phase1_score + 0.25*edge_density + 0.15*color_coherence

    def _phase3_novelty(self, pattern):
        phase2_score = self._phase2_composition(pattern)
        novelty = self._compare_to_archive(pattern.canvas)
        return 0.7*phase2_score + 0.3*novelty

    def _compare_to_archive(self, canvas):
        """Novelty detection with archive management"""
        hash_size = 32
        flat = (canvas * 255).astype(np.uint8).flatten()
        features = np.mean(flat.reshape(-1, hash_size), axis=1)
        
        if not self.archive:
            self.archive.append(features)
            return 1.0  # First pattern is maximally novel
            
        distances = [np.linalg.norm(features - a) for a in self.archive]
        novelty = np.mean(distances) / (255 * np.sqrt(hash_size))
        
        if novelty > 0.7:
            self.archive.append(features)
            
        return min(novelty, 1.0)

    def _evaluate_symmetry(self, canvas):
        """Calculate horizontal symmetry score (0-1)"""
        try:
            if canvas is None or canvas.ndim != 2:
                return 0.0
                
            h, w = canvas.shape
            half = w // 2
            left = canvas[:, :half]
            right = np.flip(canvas[:, half + w%2:], axis=1)
            return 1 - np.mean(np.abs(left - right))
        except Exception as e:
            print(f"Symmetry evaluation error: {e}")
            return 0.0

    def _evaluate_contrast(self, canvas):
        """Calculate contrast with minimum threshold"""
        try:
            return max(np.std(canvas), 0.1)  # Ensure minimum contrast
        except Exception as e:
            print(f"Contrast evaluation error: {e}")
            return 0.1

    def evaluate_subjective(self, user_feedback):
        return np.clip(float(user_feedback), 0.0, 1.0)

    def combine_scores(self, objective, subjective):
        return (1 - self.user_weight)*objective + self.user_weight*subjective
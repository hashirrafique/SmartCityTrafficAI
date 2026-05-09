"""
ann_priority.py
---------------
Module 3: ANN Priority Prediction.

A small Multi-Layer Perceptron (MLPClassifier from scikit-learn) is
trained on a manually curated dataset to classify each request into one
of four priority levels: Low / Normal / High / Critical.

The ANN does not decide whether a request is legally allowed - that is
the job of the Logic / Knowledge Base module. It only estimates urgency.

Multiclass output (rather than the binary baseline) is preferred because
it provides richer downstream signal for the CSP and Final Response.
"""

import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from data.ann_training_data import get_training_data

# Some MLP configurations on small data can throw convergence warnings
# during training. They are harmless for an academic demo, so we silence
# them so the CLI output stays clean.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ANNPriorityPredictor:
    """
    Wrapper around two scikit-learn MLPClassifier models providing a
    clean API that the rest of the project can use.

    * Multiclass model (preferred, used by the rest of the project):
        4 classes - Low / Normal / High / Critical
    * Binary baseline model (kept for completeness with the spec, which
      describes Option A: Binary classifier "urgent / non-urgent"):
        2 classes - urgent / non-urgent
        urgent = predicted multiclass label in {High, Critical}

    Both models are trained at construction time on the same dataset.
    """

    PRIORITY_LEVELS = ("Low", "Normal", "High", "Critical")
    URGENT_LEVELS = {"High", "Critical"}      # used for binary labelling

    # -----------------------------------------------------------------
    def __init__(self, random_state=42):
        """
        Build the scaler + MLP pipeline (multiclass) AND a separate
        binary baseline classifier, then train both immediately so the
        predictor is ready for use after construction.
        """
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=4000,
            random_state=random_state,
        )
        # Binary baseline - smaller net for the simpler task
        self.binary_model = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="adam",
            max_iter=4000,
            random_state=random_state,
        )
        self.training_accuracy = None           # multiclass
        self.binary_training_accuracy = None    # binary
        self._train()

    # -----------------------------------------------------------------
    def _train(self):
        """
        Fit the scaler and BOTH MLPs on the bundled training dataset,
        then record the training-set accuracy of each for reporting.
        """
        X_raw, y = get_training_data()
        X = np.array(X_raw, dtype=float)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Multiclass head
        self.model.fit(X_scaled, y)
        preds = self.model.predict(X_scaled)
        self.training_accuracy = float(np.mean(preds == np.array(y)))

        # Binary head: re-label urgent vs non-urgent
        y_binary = ["urgent" if lbl in self.URGENT_LEVELS else "non-urgent"
                    for lbl in y]
        self.binary_model.fit(X_scaled, y_binary)
        bin_preds = self.binary_model.predict(X_scaled)
        self.binary_training_accuracy = float(
            np.mean(bin_preds == np.array(y_binary))
        )

    # -----------------------------------------------------------------
    def predict(self, feature_vector):
        """
        Predict the priority level for a single feature vector.

        Parameters
        ----------
        feature_vector : list[int]  6 numeric features

        Returns
        -------
        dict with keys
            predicted_priority : str   one of PRIORITY_LEVELS
            confidence         : float (0.0 - 1.0)
            class_probabilities: dict  level -> probability
            explanation        : str   short natural-language rationale
        """
        # Defensive validation - never let bad input crash the system
        if not isinstance(feature_vector, (list, tuple)):
            raise TypeError("feature_vector must be a list or tuple.")
        if len(feature_vector) != 6:
            raise ValueError(
                f"feature_vector must have exactly 6 elements, got {len(feature_vector)}."
            )
        try:
            arr = np.array(feature_vector, dtype=float).reshape(1, -1)
        except Exception:
            raise ValueError("feature_vector contains non-numeric values.")

        x_scaled = self.scaler.transform(arr)
        prediction = str(self.model.predict(x_scaled)[0])
        probs = self.model.predict_proba(x_scaled)[0]

        # Build a probability dict in canonical order
        class_probs = {}
        for cls, p in zip(self.model.classes_, probs):
            class_probs[str(cls)] = round(float(p), 4)
        confidence = round(float(max(probs)), 4)

        explanation = self._build_explanation(feature_vector, prediction, confidence)

        return {
            "predicted_priority": prediction,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "explanation": explanation,
        }

    # -----------------------------------------------------------------
    def predict_binary(self, feature_vector):
        """
        Predict using the binary urgent/non-urgent baseline classifier.

        Returns
        -------
        dict with keys
            predicted_label : 'urgent' or 'non-urgent'
            confidence      : float
            explanation     : str
        """
        if not isinstance(feature_vector, (list, tuple)):
            raise TypeError("feature_vector must be a list or tuple.")
        if len(feature_vector) != 6:
            raise ValueError(
                f"feature_vector must have exactly 6 elements, got {len(feature_vector)}."
            )
        try:
            arr = np.array(feature_vector, dtype=float).reshape(1, -1)
        except Exception:
            raise ValueError("feature_vector contains non-numeric values.")

        x_scaled = self.scaler.transform(arr)
        label = str(self.binary_model.predict(x_scaled)[0])
        probs = self.binary_model.predict_proba(x_scaled)[0]
        confidence = round(float(max(probs)), 4)
        return {
            "predicted_label": label,
            "confidence": confidence,
            "explanation": (
                f"Binary baseline classified the request as '{label}' "
                f"with confidence {confidence:.2f}."
            ),
        }

    # -----------------------------------------------------------------
    @staticmethod
    def _build_explanation(features, prediction, confidence):
        """
        Generate a short, human-readable sentence describing why the
        ANN reached its prediction. Purely heuristic - meant for the
        Final Response layer to surface to operators.
        """
        v_type, severity, time_s, density, distance, claim = features
        vehicle_names = {0: "civilian", 1: "police", 2: "fire", 3: "ambulance"}
        vname = vehicle_names.get(int(v_type), "unknown vehicle")
        bits = [
            f"vehicle={vname}",
            f"severity_lvl={int(severity)}",
            f"time_sens_lvl={int(time_s)}",
            f"density_lvl={int(density)}",
            f"distance={int(distance)}",
            f"claim={'yes' if int(claim) else 'no'}",
        ]
        return (
            f"ANN predicted '{prediction}' priority with confidence {confidence:.2f} "
            f"based on features [{', '.join(bits)}]."
        )


# ---------------------------------------------------------------------------
# Module-level singleton helpers (so the model is only trained once per run)
# ---------------------------------------------------------------------------
_PREDICTOR_SINGLETON = None


def get_predictor():
    """
    Lazily build the singleton ANNPriorityPredictor. Subsequent calls
    return the same trained instance.
    """
    global _PREDICTOR_SINGLETON
    if _PREDICTOR_SINGLETON is None:
        _PREDICTOR_SINGLETON = ANNPriorityPredictor()
    return _PREDICTOR_SINGLETON


def predict_priority(feature_vector):
    """
    Convenience function - run a single prediction using the singleton
    predictor and return the result dictionary.
    """
    return get_predictor().predict(feature_vector)


def predict_priority_binary(feature_vector):
    """
    Convenience function - run a single binary urgent/non-urgent
    prediction using the singleton predictor.
    """
    return get_predictor().predict_binary(feature_vector)

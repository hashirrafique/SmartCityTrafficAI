"""
ann_training_data.py
--------------------
Manually curated training dataset used to train the ANN priority predictor.

Each row represents one historical traffic request and contains the six
input features expected by the ANN, plus its priority label.

Feature order (must match preprocessing.build_feature_vector):
    [vehicle_type_code, incident_severity_code, time_sensitivity_code,
     traffic_density_code, estimated_distance, priority_claim_code]

Categorical encoding (also defined in preprocessing.py):
    vehicle_type:         Civilian=0, Police=1, Fire=2, Ambulance=3
    incident_severity:    None=0, Low=1, Medium=2, High=3
    time_sensitivity:     Low=0, Normal=1, High=2
    traffic_density:      Low=0, Medium=1, High=2
    priority_claim:       No=0, Yes=1
    estimated_distance:   integer in km (0-50)

Priority labels (multiclass):
    "Low", "Normal", "High", "Critical"
"""


# ---------------------------------------------------------------------------
# Training samples: (features, label)
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    # ---- Civilian, normal cases ----------------------------------------
    ([0, 0, 0, 0,  3, 0], "Low"),
    ([0, 0, 0, 1,  5, 0], "Low"),
    ([0, 0, 1, 0,  4, 0], "Normal"),
    ([0, 0, 1, 1,  6, 0], "Normal"),
    ([0, 0, 1, 2,  8, 0], "Normal"),
    ([0, 1, 1, 1,  7, 0], "Normal"),
    ([0, 1, 0, 2, 10, 0], "Normal"),
    ([0, 0, 0, 2, 12, 0], "Low"),
    ([0, 0, 1, 0,  2, 0], "Normal"),
    ([0, 1, 1, 2, 11, 0], "Normal"),

    # Civilian falsely claiming priority -> still Normal at most
    ([0, 1, 1, 2,  9, 1], "Normal"),
    ([0, 0, 2, 2, 14, 1], "Normal"),

    # ---- Police, varying urgency ---------------------------------------
    ([1, 1, 1, 1,  6, 0], "Normal"),
    ([1, 1, 2, 1,  5, 1], "High"),
    ([1, 2, 2, 2,  8, 1], "High"),
    ([1, 3, 2, 2, 10, 1], "Critical"),
    ([1, 2, 1, 2,  9, 0], "High"),
    ([1, 1, 2, 0,  3, 1], "High"),

    # ---- Fire ----------------------------------------------------------
    ([2, 2, 2, 1,  7, 1], "High"),
    ([2, 3, 2, 2,  9, 1], "Critical"),
    ([2, 3, 2, 1,  6, 1], "Critical"),
    ([2, 2, 1, 2,  8, 1], "High"),
    ([2, 1, 1, 0,  4, 0], "Normal"),
    ([2, 3, 2, 0,  3, 1], "Critical"),

    # ---- Ambulance ------------------------------------------------------
    ([3, 1, 1, 1,  5, 1], "High"),
    ([3, 2, 2, 1,  6, 1], "High"),
    ([3, 2, 2, 2,  9, 1], "Critical"),
    ([3, 3, 2, 2,  8, 1], "Critical"),
    ([3, 3, 2, 1,  7, 1], "Critical"),
    ([3, 3, 2, 0,  4, 1], "Critical"),
    ([3, 1, 0, 0,  3, 0], "Normal"),
    ([3, 2, 1, 1,  6, 0], "High"),
    ([3, 3, 1, 2, 12, 1], "Critical"),
    ([3, 2, 2, 2, 11, 1], "Critical"),

    # ---- Edge / boundary cases -----------------------------------------
    ([0, 0, 0, 0,  1, 0], "Low"),
    ([3, 3, 2, 2, 15, 1], "Critical"),
    ([2, 3, 2, 2, 13, 1], "Critical"),
    ([1, 3, 2, 2, 14, 1], "Critical"),
    ([0, 1, 1, 2,  4, 0], "Normal"),
    ([1, 0, 0, 0,  2, 0], "Low"),
    ([2, 0, 0, 0,  3, 0], "Low"),
    ([3, 0, 0, 0,  2, 0], "Low"),
]


def get_training_data():
    """
    Return the training dataset as two parallel lists (X, y).

    Returns
    -------
    X : list[list[int]]   feature vectors
    y : list[str]         priority labels
    """
    X = [sample[0] for sample in TRAINING_DATA]
    y = [sample[1] for sample in TRAINING_DATA]
    return X, y

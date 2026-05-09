"""
preprocessing.py
----------------
Module 1: Input & Preprocessing.

Responsibilities
----------------
1. Receive a raw structured traffic request (dictionary).
2. Validate that all required fields are present.
3. Normalize values (consistent capitalization, allowed enums).
4. Reject invalid request categories, vehicle types, locations,
   severities, densities, etc.
5. Return a clean internal request object.
6. Build a numerical feature vector for the ANN module when needed.

NO Natural Language Processing is used: all inputs are structured.
"""

# ---------------------------------------------------------------------------
# Allowed values (enum-like sets)
# ---------------------------------------------------------------------------
VALID_VEHICLE_TYPES = {"Civilian", "Police", "Fire", "Ambulance"}

VALID_REQUEST_CATEGORIES = {
    "Route_Request",
    "Policy_Check",
    "Control_Allocation_Request",
    "Emergency_Response_Request",
    "Integrated_City_Service_Request",
}

VALID_LOCATIONS = {
    "Police_HQ", "Traffic_Control_Center", "River_Bridge", "North_Station",
    "Stadium", "East_Market", "Airport_Road", "City_Hospital",
    "South_Residential", "Central_Junction", "West_Terminal",
    "Fire_Station", "Industrial_Zone",
}

VALID_SEVERITY = {"None", "Low", "Medium", "High"}
VALID_TIME_SENSITIVITY = {"Low", "Normal", "High"}
VALID_TRAFFIC_DENSITY = {"Low", "Medium", "High"}
VALID_PRIORITY_CLAIM = {"Yes", "No"}

# Required keys in every incoming request
REQUIRED_FIELDS = [
    "request_id", "vehicle_type", "request_category",
    "current_location", "destination",
    "incident_severity", "time_sensitivity", "traffic_density",
    "priority_claim",
]

# ---------------------------------------------------------------------------
# Categorical encoders (used by ANN feature builder)
# ---------------------------------------------------------------------------
VEHICLE_CODE = {"Civilian": 0, "Police": 1, "Fire": 2, "Ambulance": 3}
SEVERITY_CODE = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
TIME_SENS_CODE = {"Low": 0, "Normal": 1, "High": 2}
DENSITY_CODE = {"Low": 0, "Medium": 1, "High": 2}
CLAIM_CODE = {"No": 0, "Yes": 1}

# Categorical mapping for request_category (declared even though the ANN
# does not currently consume it, because the spec lists it among the
# fields that must have a categorical mapping).
REQUEST_CATEGORY_CODE = {
    "Route_Request": 0,
    "Policy_Check": 1,
    "Control_Allocation_Request": 2,
    "Emergency_Response_Request": 3,
    "Integrated_City_Service_Request": 4,
}


# ---------------------------------------------------------------------------
def _normalize_string(value):
    """
    Trim whitespace from a string field. Returns None if value is None
    and returns the value unchanged if it is not a string. Case-folding
    is handled separately by `_canonicalize_against` so the user can
    enter "ambulance" and have it accepted as "Ambulance".
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    return value.strip()


# ---------------------------------------------------------------------------
def _canonicalize_against(value, allowed_set):
    """
    Try to map `value` to a member of `allowed_set` ignoring case and
    surrounding whitespace. Returns the canonical (correctly-cased)
    value if a match is found, else returns the original value
    (so downstream validation still rejects it cleanly).
    """
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    if cleaned in allowed_set:
        return cleaned
    lower_map = {a.lower(): a for a in allowed_set}
    return lower_map.get(cleaned.lower(), cleaned)


# ---------------------------------------------------------------------------
def _check_field(name, value, allowed_set):
    """
    Verify that a single field's value is contained in the allowed set.
    Raises ValueError with a clear message if it is not.
    """
    if value not in allowed_set:
        allowed = ", ".join(sorted(allowed_set))
        raise ValueError(
            f"Invalid value for '{name}': '{value}'. "
            f"Allowed values are: {allowed}."
        )


# ---------------------------------------------------------------------------
def estimate_distance(current_location, destination):
    """
    Approximate the road distance (in km) between two locations.
    Uses the weighted graph from search_navigation as the reference.
    A small lookup is sufficient for ANN feature input.
    """
    # Local import avoids circular-import problems at module load time
    from modules.search_navigation import compute_estimated_distance
    return compute_estimated_distance(current_location, destination)


# ---------------------------------------------------------------------------
def preprocess_request(raw_request):
    """
    Validate and normalize a raw request dictionary.

    Parameters
    ----------
    raw_request : dict   incoming structured request

    Returns
    -------
    dict   a cleaned internal request object

    Raises
    ------
    ValueError   if any field is missing or invalid
    TypeError    if the input is not a dictionary
    """
    if not isinstance(raw_request, dict):
        raise TypeError("Request must be a dictionary of fields.")

    # 1. Check for missing required fields
    missing = [f for f in REQUIRED_FIELDS if f not in raw_request or raw_request[f] in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    # 2. Build a normalized copy (whitespace stripped)
    cleaned = {key: _normalize_string(val) for key, val in raw_request.items()}

    # 2b. Case-insensitive canonicalization for enum fields:
    # accept "ambulance" / "AMBULANCE" / "Ambulance" all the same.
    cleaned["vehicle_type"]      = _canonicalize_against(cleaned["vehicle_type"],      VALID_VEHICLE_TYPES)
    cleaned["request_category"]  = _canonicalize_against(cleaned["request_category"],  VALID_REQUEST_CATEGORIES)
    cleaned["current_location"]  = _canonicalize_against(cleaned["current_location"],  VALID_LOCATIONS)
    cleaned["destination"]       = _canonicalize_against(cleaned["destination"],       VALID_LOCATIONS)
    cleaned["incident_severity"] = _canonicalize_against(cleaned["incident_severity"], VALID_SEVERITY)
    cleaned["time_sensitivity"]  = _canonicalize_against(cleaned["time_sensitivity"],  VALID_TIME_SENSITIVITY)
    cleaned["traffic_density"]   = _canonicalize_against(cleaned["traffic_density"],   VALID_TRAFFIC_DENSITY)
    cleaned["priority_claim"]    = _canonicalize_against(cleaned["priority_claim"],    VALID_PRIORITY_CLAIM)

    # 3. Field-level validation (after canonicalization)
    _check_field("vehicle_type", cleaned["vehicle_type"], VALID_VEHICLE_TYPES)
    _check_field("request_category", cleaned["request_category"], VALID_REQUEST_CATEGORIES)
    _check_field("current_location", cleaned["current_location"], VALID_LOCATIONS)
    _check_field("destination", cleaned["destination"], VALID_LOCATIONS)
    _check_field("incident_severity", cleaned["incident_severity"], VALID_SEVERITY)
    _check_field("time_sensitivity", cleaned["time_sensitivity"], VALID_TIME_SENSITIVITY)
    _check_field("traffic_density", cleaned["traffic_density"], VALID_TRAFFIC_DENSITY)
    _check_field("priority_claim", cleaned["priority_claim"], VALID_PRIORITY_CLAIM)

    if cleaned["current_location"] == cleaned["destination"]:
        raise ValueError("current_location and destination must be different.")

    # 4. Optional / derived fields
    cleaned.setdefault("control_zone", "None")
    cleaned.setdefault("description_note", "")

    cleaned["estimated_distance"] = estimate_distance(
        cleaned["current_location"], cleaned["destination"]
    )

    # 5. Helpful boolean flags used downstream
    cleaned["is_emergency_vehicle"] = cleaned["vehicle_type"] in {"Police", "Fire", "Ambulance"}
    cleaned["is_civilian_vehicle"] = cleaned["vehicle_type"] == "Civilian"

    # Numeric request-category code (kept for completeness; spec lists
    # request_category as a categorical field with a mapping).
    cleaned["request_category_code"] = REQUEST_CATEGORY_CODE[cleaned["request_category"]]

    return cleaned


# ---------------------------------------------------------------------------
def build_feature_vector(cleaned_request):
    """
    Convert a cleaned request dictionary into a numeric feature vector
    suitable for the ANN priority predictor.

    Feature order:
        [vehicle_type_code, severity_code, time_sens_code,
         density_code, estimated_distance, priority_claim_code]
    """
    try:
        vec = [
            VEHICLE_CODE[cleaned_request["vehicle_type"]],
            SEVERITY_CODE[cleaned_request["incident_severity"]],
            TIME_SENS_CODE[cleaned_request["time_sensitivity"]],
            DENSITY_CODE[cleaned_request["traffic_density"]],
            int(cleaned_request["estimated_distance"]),
            CLAIM_CODE[cleaned_request["priority_claim"]],
        ]
    except KeyError as exc:
        raise ValueError(f"Cannot build feature vector - field {exc} missing.")
    return vec

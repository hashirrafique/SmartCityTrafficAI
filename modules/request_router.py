"""
request_router.py
-----------------
Module 2: Request Router (control-flow manager).

The router looks at the standardized request_category and decides which
AI modules to activate. It does NOT run them itself - it only returns a
pipeline plan. This keeps the system disciplined: the wrong modules are
never called for the wrong category.
"""


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------
# Each entry maps a request_category to the ordered list of module names
# that must be executed for that category.
PIPELINES = {
    "Route_Request":                  ["Search"],
    "Policy_Check":                   ["Logic"],
    "Control_Allocation_Request":     ["Logic", "CSP"],
    "Emergency_Response_Request":     ["ANN", "Logic", "CSP", "Search"],
    "Integrated_City_Service_Request":["ANN", "Logic", "CSP", "Search", "FinalResponse"],
}


# ---------------------------------------------------------------------------
def route_request(cleaned_request):
    """
    Determine the processing pipeline for a cleaned request.

    Parameters
    ----------
    cleaned_request : dict   the output of preprocessing.preprocess_request

    Returns
    -------
    list[str]  ordered list of module names to execute

    Raises
    ------
    ValueError if the request_category is not recognised
    """
    category = cleaned_request.get("request_category")
    if category not in PIPELINES:
        raise ValueError(
            f"Unsupported request_category '{category}'. "
            f"Allowed: {list(PIPELINES.keys())}."
        )
    # Return a fresh copy so callers cannot mutate the master list
    return list(PIPELINES[category])


# ---------------------------------------------------------------------------
def explain_pipeline(category):
    """
    Return a short human-readable explanation of why a particular
    pipeline was chosen for the given request category.
    """
    explanations = {
        "Route_Request":
            "Standard navigation only - search the city graph for the best path.",
        "Policy_Check":
            "Rule-based authorization check - logic module only.",
        "Control_Allocation_Request":
            "Authorize first (Logic), then assign signal/lane controls (CSP).",
        "Emergency_Response_Request":
            "Predict urgency (ANN), validate authority (Logic), allocate corridor "
            "(CSP), then route (Search).",
        "Integrated_City_Service_Request":
            "Full integrated pipeline: ANN -> Logic -> CSP -> Search -> Final Response.",
    }
    return explanations.get(category, "Unknown category.")

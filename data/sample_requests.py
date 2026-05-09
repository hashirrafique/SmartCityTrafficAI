"""
sample_requests.py
------------------
Predefined demo scenarios used by the menu-driven main program and the
test harness. Each scenario is a structured dictionary that mirrors the
input fields described in the project specification.
"""


SAMPLE_REQUESTS = {

    # ---------------------------------------------------------------
    # Scenario 1: Plain civilian route request (Search only)
    # ---------------------------------------------------------------
    "scenario_1_route": {
        "request_id": "REQ-001",
        "vehicle_type": "Civilian",
        "request_category": "Route_Request",
        "current_location": "Stadium",
        "destination": "City_Hospital",
        "incident_severity": "None",
        "time_sensitivity": "Normal",
        "traffic_density": "Medium",
        "priority_claim": "No",
        "control_zone": "None",
        "description_note": "Civilian wants quickest path to hospital."
    },

    # ---------------------------------------------------------------
    # Scenario 2: Civilian asking for signal override (Logic only -> Reject)
    # ---------------------------------------------------------------
    "scenario_2_policy": {
        "request_id": "REQ-002",
        "vehicle_type": "Civilian",
        "request_category": "Policy_Check",
        "current_location": "East_Market",
        "destination": "North_Station",
        "incident_severity": "Low",
        "time_sensitivity": "Low",
        "traffic_density": "High",
        "priority_claim": "Yes",
        "control_zone": "East_Market",
        "description_note": "Civilian requesting signal override (should be rejected)."
    },

    # ---------------------------------------------------------------
    # Scenario 3: Emergency vehicle wants control allocation (Logic -> CSP)
    # ---------------------------------------------------------------
    "scenario_3_control": {
        "request_id": "REQ-003",
        "vehicle_type": "Fire",
        "request_category": "Control_Allocation_Request",
        "current_location": "Fire_Station",
        "destination": "Industrial_Zone",
        "incident_severity": "High",
        "time_sensitivity": "High",
        "traffic_density": "High",
        "priority_claim": "Yes",
        "control_zone": "Central_Junction",
        "description_note": "Fire truck needs intersection signal coordination."
    },

    # ---------------------------------------------------------------
    # Scenario 4: Ambulance emergency (ANN -> Logic -> CSP -> Search)
    # ---------------------------------------------------------------
    "scenario_4_emergency": {
        "request_id": "REQ-004",
        "vehicle_type": "Ambulance",
        "request_category": "Emergency_Response_Request",
        "current_location": "Central_Junction",
        "destination": "City_Hospital",
        "incident_severity": "High",
        "time_sensitivity": "High",
        "traffic_density": "High",
        "priority_claim": "Yes",
        "control_zone": "Central_Junction",
        "description_note": "Ambulance carrying critical patient to hospital."
    },

    # ---------------------------------------------------------------
    # Scenario 5: Integrated full pipeline (ANN -> Logic -> CSP -> Search -> Final)
    # ---------------------------------------------------------------
    "scenario_5_integrated": {
        "request_id": "REQ-005",
        "vehicle_type": "Ambulance",
        "request_category": "Integrated_City_Service_Request",
        "current_location": "North_Station",
        "destination": "City_Hospital",
        "incident_severity": "High",
        "time_sensitivity": "High",
        "traffic_density": "High",
        "priority_claim": "Yes",
        "control_zone": "Central_Junction",
        "description_note": "Full integrated emergency response with corridor."
    },
}


def get_scenario(name):
    """
    Return a copy of one named demo scenario dictionary.

    Parameters
    ----------
    name : str   key of the desired scenario

    Returns
    -------
    dict (copy) so the original template is not mutated by callers.
    """
    if name not in SAMPLE_REQUESTS:
        raise KeyError(f"Unknown scenario: {name}")
    return dict(SAMPLE_REQUESTS[name])


def list_scenarios():
    """Return the list of scenario keys for menu display."""
    return list(SAMPLE_REQUESTS.keys())

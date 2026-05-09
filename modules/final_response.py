"""
final_response.py
-----------------
Module 7: Final Response & Integration.

Aggregates the outputs of every AI module that was actually executed
for the current request and produces a single clean, explainable
response object. Modules that were not used are NOT included in the
output - this preserves the project's "selective response" rule.
"""


# ---------------------------------------------------------------------------
# Pretty separators used by the CLI display
# ---------------------------------------------------------------------------
LINE = "-" * 70
DLINE = "=" * 70


# ---------------------------------------------------------------------------
def build_final_response(request, pipeline, ann_result=None,
                         logic_result=None, csp_result=None,
                         search_result=None):
    """
    Combine the outputs of the modules that were used. Any argument
    left as None is interpreted as "this module was not used".

    Parameters
    ----------
    request        : dict   the cleaned request object
    pipeline       : list[str]   ordered list of module names used
    ann_result     : dict or None
    logic_result   : dict or None
    csp_result     : dict or None
    search_result  : dict or None

    Returns
    -------
    dict   aggregated final response (only relevant fields included)
    """
    response = {
        "request_id": request["request_id"],
        "request_category": request["request_category"],
        "vehicle_type": request["vehicle_type"],
        "current_location": request["current_location"],
        "destination": request["destination"],
        "modules_used": list(pipeline),
    }

    # Decide a single high-level decision message
    decision = "Approved"
    decision_source = []

    if ann_result:
        response["predicted_priority_level"] = ann_result["predicted_priority"]
        response["ann_confidence"] = ann_result["confidence"]
        response["ann_explanation"] = ann_result["explanation"]
        # Binary baseline result (passed through from main.process_request)
        if "binary_label" in ann_result:
            response["binary_priority_label"] = ann_result["binary_label"]
            response["binary_priority_confidence"] = ann_result["binary_confidence"]

    if logic_result:
        response["rule_priority"] = logic_result["rule_priority"]
        response["authorized_actions"] = logic_result["authorized_actions"]
        response["allowed_actions"] = logic_result["allowed_actions"]
        response["policy_status"] = logic_result["status"]
        response["rules_fired"] = logic_result["rules_fired"]
        if logic_result["status"] == "Rejected":
            decision = "Rejected"
            decision_source.append("Logic")

    if csp_result:
        response["control_assignment"] = csp_result.get("assignment")
        response["constraints_satisfied"] = csp_result.get("constraints_satisfied")
        response["control_explanation"] = csp_result.get("explanation")
        if not csp_result.get("success"):
            decision = "Rejected"
            decision_source.append("CSP")

    if search_result:
        response["recommended_route"] = search_result.get("path")
        response["route_cost"] = search_result.get("cost")
        response["search_algorithm"] = search_result.get("algorithm")
        response["search_explanation"] = search_result.get("explanation")
        if not search_result.get("path"):
            decision = "Rejected"
            decision_source.append("Search")

    # If nothing rejected and we have at least one module run, mark approved
    response["final_decision"] = decision
    if decision == "Approved":
        response["decision_message"] = _build_approved_message(request, response)
    else:
        response["decision_message"] = _build_rejected_message(decision_source, response)

    return response


# ---------------------------------------------------------------------------
def _build_approved_message(request, response):
    """Produce a one-line summary message for an approved request."""
    cat = request["request_category"]
    if cat == "Route_Request":
        return (f"Route generated from {response['current_location']} to "
                f"{response['destination']} (cost {response.get('route_cost')}).")
    if cat == "Policy_Check":
        return "Policy check passed: vehicle has at least one authorized action."
    if cat == "Control_Allocation_Request":
        return "Control allocation succeeded under all safety constraints."
    if cat == "Emergency_Response_Request":
        return (f"Emergency response approved: priority "
                f"{response.get('predicted_priority_level')}, route via "
                f"{response.get('search_algorithm')} (cost "
                f"{response.get('route_cost')}).")
    if cat == "Integrated_City_Service_Request":
        return ("Integrated emergency service approved: corridor allocated, "
                "signal support assigned, route computed.")
    return "Request approved."


# ---------------------------------------------------------------------------
def _build_rejected_message(sources, response):
    """Produce a one-line summary message for a rejected request."""
    src = ", ".join(sources) if sources else "system"
    return f"Request rejected by {src}. See module-level details above."


# ---------------------------------------------------------------------------
def render_response(response):
    """
    Pretty-print the aggregated response to stdout in a readable
    block format. Returns the same dict so it can be chained.
    """
    print(DLINE)
    print(f" FINAL RESPONSE  -  {response['request_id']}")
    print(DLINE)
    print(f" Category        : {response['request_category']}")
    print(f" Vehicle         : {response['vehicle_type']}")
    print(f" From -> To      : {response['current_location']} -> "
          f"{response['destination']}")
    print(f" Modules used    : {' -> '.join(response['modules_used'])}")
    print(LINE)

    if "predicted_priority_level" in response:
        print(f" ANN priority    : {response['predicted_priority_level']} "
              f"(confidence {response['ann_confidence']})")
        if "binary_priority_label" in response:
            print(f" ANN binary      : {response['binary_priority_label']} "
                  f"(confidence {response['binary_priority_confidence']})")

    if "rule_priority" in response:
        print(f" Rule priority   : {response['rule_priority']}")
        print(f" Authorized      : {response['authorized_actions']}")
        print(f" Allowed actions : {response['allowed_actions']}")
        print(f" Policy status   : {response['policy_status']}")
        if response.get("rules_fired"):
            print(f" Rules fired ({len(response['rules_fired'])}):")
            for rule in response["rules_fired"]:
                # Show only the rule label + short description, not the full
                # derivation tuple, so the line stays readable.
                head = rule.split("  ==>")[0]
                print(f"     {head}")

    if "control_assignment" in response and response["control_assignment"]:
        print(" Signal plan     :")
        for var, val in response["control_assignment"].items():
            print(f"     {var:<20} -> {val}")

    if response.get("recommended_route"):
        path = " -> ".join(response["recommended_route"])
        print(f" Route ({response['search_algorithm']:<3}) cost={response['route_cost']}")
        print(f"     Path: {path}")

    print(LINE)
    print(f" Final decision  : {response['final_decision']}")
    print(f" Message         : {response['decision_message']}")
    print(DLINE)
    return response

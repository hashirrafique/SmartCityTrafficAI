"""
main.py
-------
Entry point for the Smart City Traffic & Emergency Response AI System.

Provides a menu-driven CLI that:
    1. Lets the user run any of the 5 demo scenarios
    2. Lets the user enter a custom request
    3. Trains the ANN once at startup and reports its accuracy
    4. Routes each request through only the appropriate AI modules
    5. Pretty-prints the integrated final response

Module sequence (per the project specification)
    Input & Preprocessing -> Request Router -> ANN -> Logic ->
    CSP -> Search -> Final Response
"""

import sys
import traceback

from data.sample_requests import SAMPLE_REQUESTS, get_scenario, list_scenarios

from modules.preprocessing import (
    preprocess_request, build_feature_vector,
    VALID_VEHICLE_TYPES, VALID_REQUEST_CATEGORIES, VALID_LOCATIONS,
    VALID_SEVERITY, VALID_TIME_SENSITIVITY, VALID_TRAFFIC_DENSITY,
    VALID_PRIORITY_CLAIM,
)
from modules.request_router import route_request, explain_pipeline
from modules.ann_priority import get_predictor
from modules.knowledge_base import evaluate_request
from modules.csp_scheduler import allocate_controls
from modules.search_navigation import navigate
from modules.final_response import build_final_response, render_response


BANNER = r"""
======================================================================
   SMART CITY TRAFFIC & EMERGENCY RESPONSE AI SYSTEM
   AI Lab Final Project  -  FAST-NUCES (CFD Campus)
======================================================================
"""


# ---------------------------------------------------------------------------
def process_request(raw_request, ann_predictor, verbose=True):
    """
    Run a raw request through the entire AI pipeline (only the modules
    appropriate for its category). Returns the aggregated final-response
    dictionary.
    """
    # 1. Preprocess
    cleaned = preprocess_request(raw_request)
    if verbose:
        print(f"\n[Preprocessing] OK - cleaned request id={cleaned['request_id']}")
        print(f"                estimated_distance = {cleaned['estimated_distance']} km")

    # 2. Route
    pipeline = route_request(cleaned)
    if verbose:
        print(f"[Router]        category={cleaned['request_category']}")
        print(f"                pipeline = {' -> '.join(pipeline)}")
        print(f"                why = {explain_pipeline(cleaned['request_category'])}")

    ann_result = logic_result = csp_result = search_result = None

    # 3. ANN
    if "ANN" in pipeline:
        feats = build_feature_vector(cleaned)
        ann_result = ann_predictor.predict(feats)
        binary_result = ann_predictor.predict_binary(feats)
        ann_result["binary_label"] = binary_result["predicted_label"]
        ann_result["binary_confidence"] = binary_result["confidence"]
        if verbose:
            print(f"[ANN]           multiclass = "
                  f"{ann_result['predicted_priority']} "
                  f"(confidence {ann_result['confidence']})")
            print(f"                binary     = "
                  f"{binary_result['predicted_label']} "
                  f"(confidence {binary_result['confidence']})")

    # 4. Logic
    if "Logic" in pipeline:
        logic_result = evaluate_request(cleaned)
        if verbose:
            print(f"[Logic]         status={logic_result['status']}, "
                  f"priority={logic_result['rule_priority']}, "
                  f"authorized={logic_result['authorized_actions']}")

    # 5. CSP
    if "CSP" in pipeline:
        csp_result = allocate_controls(cleaned, logic_result or {})
        if verbose:
            ok = "OK" if csp_result["success"] else "FAILED"
            print(f"[CSP]           {ok} - {csp_result['explanation']}")

    # 6. Search
    if "Search" in pipeline:
        algo = "auto"
        # For pure Route_Request we use UCS (weighted, no heuristic needed)
        if cleaned["request_category"] == "Route_Request":
            algo = "UCS"
        search_result = navigate(cleaned, algorithm=algo)
        if verbose:
            print(f"[Search]        {search_result['algorithm']} cost="
                  f"{search_result['cost']} path="
                  f"{' -> '.join(search_result['path']) if search_result['path'] else 'NONE'}")

    # 7. Final Response
    response = build_final_response(
        cleaned, pipeline,
        ann_result=ann_result,
        logic_result=logic_result,
        csp_result=csp_result,
        search_result=search_result,
    )
    return response


# ---------------------------------------------------------------------------
def show_main_menu():
    """Print the top-level menu and read the user's choice."""
    print()
    print("MAIN MENU")
    print("---------")
    print("  1. Run Scenario 1 - Civilian Route Request (Search only)")
    print("  2. Run Scenario 2 - Civilian Policy Check (Logic only - rejected)")
    print("  3. Run Scenario 3 - Fire Truck Control Allocation (Logic + CSP)")
    print("  4. Run Scenario 4 - Ambulance Emergency (ANN + Logic + CSP + Search)")
    print("  5. Run Scenario 5 - Integrated City Service (full pipeline)")
    print("  6. Run ALL scenarios (1 to 5)")
    print("  7. Enter a CUSTOM request")
    print("  8. Show ANN training accuracy")
    print("  9. Show city graph (locations & weights)")
    print("  0. Exit")
    return input("Select an option: ").strip()


# ---------------------------------------------------------------------------
def _prompt(field, allowed):
    """Repeatedly prompt the user until they enter a value from `allowed`."""
    allowed_list = sorted(allowed)
    while True:
        value = input(f"   {field} {allowed_list}: ").strip()
        if value in allowed:
            return value
        print(f"   Invalid value '{value}'. Please choose from the listed options.")


def get_custom_request():
    """Interactively collect a custom request from the user."""
    print("\nEnter custom request fields (Ctrl+C to abort):")
    req_id = input("   request_id (e.g. CUST-001): ").strip() or "CUST-001"
    req = {
        "request_id": req_id,
        "vehicle_type":      _prompt("vehicle_type",      VALID_VEHICLE_TYPES),
        "request_category":  _prompt("request_category",  VALID_REQUEST_CATEGORIES),
        "current_location":  _prompt("current_location",  VALID_LOCATIONS),
        "destination":       _prompt("destination",       VALID_LOCATIONS),
        "incident_severity": _prompt("incident_severity", VALID_SEVERITY),
        "time_sensitivity":  _prompt("time_sensitivity",  VALID_TIME_SENSITIVITY),
        "traffic_density":   _prompt("traffic_density",   VALID_TRAFFIC_DENSITY),
        "priority_claim":    _prompt("priority_claim",    VALID_PRIORITY_CLAIM),
        "control_zone":      input("   control_zone (or 'None'): ").strip() or "None",
        "description_note":  input("   description_note (optional): ").strip(),
    }
    return req


# ---------------------------------------------------------------------------
def show_graph_summary():
    """Print the locations and edge weights of the city graph."""
    from modules.search_navigation import (
        UNWEIGHTED_GRAPH, WEIGHTED_EDGES,
    )
    print("\nCity locations:")
    for loc in sorted(UNWEIGHTED_GRAPH):
        print(f"  - {loc}")
    print("\nWeighted edges (km):")
    for u, v, w in WEIGHTED_EDGES:
        print(f"  {u:<25} <-> {v:<25} {w}")


# ---------------------------------------------------------------------------
def run_scenario_by_key(key, ann_predictor):
    """Helper: run a named demo scenario and pretty-print the response."""
    raw = get_scenario(key)
    print(f"\n>>> Running {key}")
    response = process_request(raw, ann_predictor)
    render_response(response)
    return response


# ---------------------------------------------------------------------------
def main():
    """Top-level CLI loop."""
    print(BANNER)
    print("Initializing AI subsystems...")
    ann_predictor = get_predictor()
    print(f"ANN ready - multiclass accuracy = {ann_predictor.training_accuracy:.2%}, "
          f"binary accuracy = {ann_predictor.binary_training_accuracy:.2%}")

    scenario_keys = list_scenarios()

    while True:
        try:
            choice = show_main_menu()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if choice == "0":
            print("Goodbye.")
            return

        try:
            if choice in {"1", "2", "3", "4", "5"}:
                run_scenario_by_key(scenario_keys[int(choice) - 1], ann_predictor)
            elif choice == "6":
                for key in scenario_keys:
                    run_scenario_by_key(key, ann_predictor)
            elif choice == "7":
                raw = get_custom_request()
                response = process_request(raw, ann_predictor)
                render_response(response)
            elif choice == "8":
                print(f"\nANN multiclass training accuracy: "
                      f"{ann_predictor.training_accuracy:.2%}")
                print(f"ANN binary    training accuracy: "
                      f"{ann_predictor.binary_training_accuracy:.2%}")
            elif choice == "9":
                show_graph_summary()
            else:
                print("Please select a valid option (0-9).")
        except ValueError as ve:
            print(f"\n[Input error] {ve}")
        except KeyError as ke:
            print(f"\n[Missing field] {ke}")
        except Exception as exc:                        # pragma: no cover
            print(f"\n[Unexpected error] {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

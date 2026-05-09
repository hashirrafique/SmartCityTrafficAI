"""
test_scenarios.py
-----------------
Light-weight test harness covering all five demo scenarios plus a few
input-validation sanity checks. Run from the project root:

    python -m tests.test_scenarios

Each test prints PASS or FAIL with a short reason. Exits with a non-zero
status code if any test fails.
"""

import sys
import os

# Allow this file to be run directly from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.sample_requests import get_scenario
from modules.preprocessing import preprocess_request
from modules.request_router import route_request
from modules.ann_priority import get_predictor
from modules.knowledge_base import evaluate_request
from modules.csp_scheduler import allocate_controls
from modules.search_navigation import navigate
from modules.final_response import build_final_response


# Track pass/fail tally
RESULTS = []


def _check(name, condition, detail=""):
    """Record a single PASS/FAIL line."""
    status = "PASS" if condition else "FAIL"
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
def _run_pipeline(raw):
    """
    Helper to run a request through whichever modules its pipeline calls
    for and return all intermediate results plus the final response.
    """
    cleaned = preprocess_request(raw)
    pipeline = route_request(cleaned)
    ann_res = logic_res = csp_res = search_res = None
    if "ANN" in pipeline:
        ann_res = get_predictor().predict(__import__(
            "modules.preprocessing", fromlist=["build_feature_vector"]
        ).build_feature_vector(cleaned))
    if "Logic" in pipeline:
        logic_res = evaluate_request(cleaned)
    if "CSP" in pipeline:
        csp_res = allocate_controls(cleaned, logic_res or {})
    if "Search" in pipeline:
        algo = "UCS" if cleaned["request_category"] == "Route_Request" else "auto"
        search_res = navigate(cleaned, algorithm=algo)
    response = build_final_response(
        cleaned, pipeline,
        ann_result=ann_res, logic_result=logic_res,
        csp_result=csp_res, search_result=search_res,
    )
    return cleaned, pipeline, response


# ---------------------------------------------------------------------------
def test_scenario_1():
    """Scenario 1: civilian route - Search only, must produce a path."""
    print("\n--- Scenario 1: Civilian Route Request ---")
    raw = get_scenario("scenario_1_route")
    _, pipeline, resp = _run_pipeline(raw)
    _check("uses Search only", pipeline == ["Search"], f"got {pipeline}")
    _check("path generated", bool(resp.get("recommended_route")),
           f"path={resp.get('recommended_route')}")
    _check("decision Approved", resp["final_decision"] == "Approved")


def test_scenario_2():
    """Scenario 2: civilian policy check, override request - must Reject."""
    print("\n--- Scenario 2: Civilian Policy Check (expect Reject) ---")
    raw = get_scenario("scenario_2_policy")
    _, pipeline, resp = _run_pipeline(raw)
    _check("uses Logic only", pipeline == ["Logic"], f"got {pipeline}")
    _check("policy rejected", resp["policy_status"] == "Rejected",
           f"policy_status={resp['policy_status']}")
    _check("final decision Rejected", resp["final_decision"] == "Rejected")


def test_scenario_3():
    """Scenario 3: fire-truck control allocation - Logic then CSP."""
    print("\n--- Scenario 3: Fire-Truck Control Allocation ---")
    raw = get_scenario("scenario_3_control")
    _, pipeline, resp = _run_pipeline(raw)
    _check("uses Logic + CSP", pipeline == ["Logic", "CSP"], f"got {pipeline}")
    _check("CSP produced assignment",
           bool(resp.get("control_assignment")),
           f"assignment={resp.get('control_assignment')}")


def test_scenario_4():
    """Scenario 4: ambulance emergency - ANN + Logic + CSP + Search."""
    print("\n--- Scenario 4: Ambulance Emergency Response ---")
    raw = get_scenario("scenario_4_emergency")
    _, pipeline, resp = _run_pipeline(raw)
    _check("uses ANN+Logic+CSP+Search",
           pipeline == ["ANN", "Logic", "CSP", "Search"], f"got {pipeline}")
    _check("ANN priority is High or Critical",
           resp["predicted_priority_level"] in ("High", "Critical"),
           f"priority={resp['predicted_priority_level']}")
    _check("rule_priority is Critical",
           resp["rule_priority"] == "Critical",
           f"rule_priority={resp['rule_priority']}")
    _check("EmergencyRoute authorized",
           "EmergencyRoute" in resp["authorized_actions"])
    _check("path computed", bool(resp.get("recommended_route")))
    _check("decision Approved", resp["final_decision"] == "Approved")


def test_scenario_5():
    """Scenario 5: integrated city service - full pipeline."""
    print("\n--- Scenario 5: Integrated City Service Request ---")
    raw = get_scenario("scenario_5_integrated")
    _, pipeline, resp = _run_pipeline(raw)
    _check("uses ANN+Logic+CSP+Search+FinalResponse",
           pipeline == ["ANN", "Logic", "CSP", "Search", "FinalResponse"],
           f"got {pipeline}")
    _check("ANN priority is High or Critical",
           resp["predicted_priority_level"] in ("High", "Critical"))
    _check("control_assignment exists",
           bool(resp.get("control_assignment")))
    _check("path exists", bool(resp.get("recommended_route")))


# ---------------------------------------------------------------------------
def test_input_validation():
    """Sanity checks on preprocessing rejecting bad input."""
    print("\n--- Input Validation Tests ---")

    # Missing field
    bad = {"request_id": "X", "vehicle_type": "Civilian"}
    try:
        preprocess_request(bad)
        _check("missing fields rejected", False, "no exception raised")
    except ValueError:
        _check("missing fields rejected", True)

    # Invalid vehicle_type
    raw = get_scenario("scenario_1_route")
    raw["vehicle_type"] = "Tank"
    try:
        preprocess_request(raw)
        _check("invalid vehicle_type rejected", False)
    except ValueError:
        _check("invalid vehicle_type rejected", True)

    # Same start and destination
    raw = get_scenario("scenario_1_route")
    raw["destination"] = raw["current_location"]
    try:
        preprocess_request(raw)
        _check("same source/destination rejected", False)
    except ValueError:
        _check("same source/destination rejected", True)

    # Case-insensitive normalization: "ambulance" should be accepted
    raw = get_scenario("scenario_4_emergency")
    raw["vehicle_type"] = "ambulance"
    raw["incident_severity"] = "high"
    try:
        cleaned = preprocess_request(raw)
        ok = (cleaned["vehicle_type"] == "Ambulance" and
              cleaned["incident_severity"] == "High")
        _check("case-insensitive normalization", ok,
               f"got vehicle_type={cleaned['vehicle_type']}, "
               f"severity={cleaned['incident_severity']}")
    except Exception as exc:
        _check("case-insensitive normalization", False, str(exc))


# ---------------------------------------------------------------------------
def test_search_algorithms():
    """Direct tests of BFS, UCS and A* on the city graph."""
    print("\n--- Search Algorithm Tests ---")
    from modules.search_navigation import bfs, ucs, astar

    res = bfs("Stadium", "City_Hospital")
    _check("BFS returns a path", res["path"] is not None,
           f"path={res['path']}")
    _check("BFS path starts at Stadium",
           res["path"] and res["path"][0] == "Stadium")
    _check("BFS path ends at City_Hospital",
           res["path"] and res["path"][-1] == "City_Hospital")

    ucs_res = ucs("Stadium", "City_Hospital")
    _check("UCS returns a path", ucs_res["path"] is not None)
    _check("UCS cost is finite and >= 1",
           isinstance(ucs_res["cost"], (int, float))
           and ucs_res["cost"] >= 1)

    a_res = astar("Central_Junction", "City_Hospital")
    _check("A* returns a path", a_res["path"] is not None)
    # A* should never beat UCS on optimal cost; verify equal here
    a_ucs = ucs("Central_Junction", "City_Hospital")
    _check("A* matches UCS optimal cost",
           a_res["cost"] == a_ucs["cost"],
           f"A*={a_res['cost']} UCS={a_ucs['cost']}")

    # Invalid endpoint must raise
    try:
        bfs("Mars", "City_Hospital")
        _check("invalid start raises ValueError", False)
    except ValueError:
        _check("invalid start raises ValueError", True)


# ---------------------------------------------------------------------------
def test_ann_accuracy():
    """Ensure the ANN trains successfully and meets a reasonable bar."""
    print("\n--- ANN Training & Binary Baseline Tests ---")
    pred = get_predictor()
    _check("multiclass training accuracy >= 80%",
           pred.training_accuracy >= 0.8,
           f"got {pred.training_accuracy:.2%}")
    _check("binary training accuracy >= 80%",
           pred.binary_training_accuracy >= 0.8,
           f"got {pred.binary_training_accuracy:.2%}")
    # Sanity: ambulance + high severity should be urgent
    res = pred.predict_binary([3, 3, 2, 2, 6, 1])
    _check("binary classifier flags ambulance+High as urgent",
           res["predicted_label"] == "urgent",
           f"got {res['predicted_label']}")


# ---------------------------------------------------------------------------
def test_kb_rule_count():
    """The KB must reference all 19 spec-listed rules in its trace."""
    print("\n--- Knowledge Base Rule Coverage Tests ---")
    raw = get_scenario("scenario_5_integrated")
    cleaned = preprocess_request(raw)
    res = evaluate_request(cleaned)
    fired = res["rules_fired"]
    # Collect distinct rule labels (R1, R2, ...) from the trace
    labels = set()
    for line in fired:
        token = line.split(":", 1)[0].strip()
        if token.startswith("R"):
            labels.add(token)
    _check("integrated scenario fires several rules",
           len(labels) >= 6, f"distinct rule labels = {sorted(labels)}")
    # The KB module *defines* 19 rules (R1..R19); verify by reading source.
    import modules.knowledge_base as kb_mod
    src = open(kb_mod.__file__).read()
    referenced = {f"R{i}" for i in range(1, 20) if f'"R{i}:' in src or f"'R{i}:" in src}
    _check("KB source references all 19 spec rule labels",
           len(referenced) == 19,
           f"found {sorted(referenced)}")


# ---------------------------------------------------------------------------
def test_csp_failure_case():
    """The CSP must report failure if constraints cannot be satisfied."""
    print("\n--- CSP Failure-Case Test ---")
    from modules.csp_scheduler import CSPScheduler
    # Force an impossible target: emergency_target not in variables AND
    # has_emergency=True is fine; but with target=City_Hospital and no
    # emergency, City_Hospital is restricted to {Red, Emergency_Green}
    # AND Emergency_Green is removed, leaving only Red - solvable.
    # Instead, demonstrate failure by manually breaking domains.
    solver = CSPScheduler(emergency_target=None, has_emergency=False)
    # Empty every domain to force failure
    for var in solver.variables:
        solver.domains[var] = []
    res = solver.solve()
    _check("CSP reports failure when domains empty",
           res["success"] is False,
           f"got success={res['success']}")


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Smart City Traffic & Emergency Response - Test Harness")
    print("=" * 70)
    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
    test_scenario_4()
    test_scenario_5()
    test_input_validation()
    test_search_algorithms()
    test_ann_accuracy()
    test_kb_rule_count()
    test_csp_failure_case()

    passed = sum(1 for r in RESULTS if r[0] == "PASS")
    failed = sum(1 for r in RESULTS if r[0] == "FAIL")
    print()
    print("=" * 70)
    print(f" Summary: {passed} passed, {failed} failed, "
          f"{len(RESULTS)} total")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

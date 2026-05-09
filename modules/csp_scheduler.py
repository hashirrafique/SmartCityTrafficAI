"""
csp_scheduler.py
----------------
Module 5: CSP Scheduler / Control Allocation.

Models traffic-signal allocation across a small set of intersections as
a Constraint Satisfaction Problem and solves it with a custom backtracking
search (chosen over python-constraint for transparency: each step can be
explained in the report).

Variables (intersections / control points):
    Central_Junction, East_Market, West_Terminal,
    South_Residential, City_Hospital

Domain values (signal states):
    Green_NS         - Green for North-South traffic
    Green_EW         - Green for East-West traffic
    Red              - All directions stopped
    Emergency_Green  - Reserved priority green for an emergency corridor
    Hold             - Maintain current state, no transition this cycle

Constraints:
    C1: No two adjacent intersections may both be Green_EW (avoids
        unsafe waves of east-west traffic colliding at shared edges).
    C2: At most one intersection may be assigned Emergency_Green per
        plan (only one emergency corridor at a time).
    C3: An emergency corridor request must place Emergency_Green on
        the corridor's destination intersection (when supplied).
    C4: City_Hospital, when present, may only be Emergency_Green or
        Red (a hospital intersection never serves as a generic green).
    C5: If an intersection is assigned Emergency_Green, none of its
        adjacent intersections may be Green_NS or Green_EW (corridor
        must not be crossed by general traffic).
    C6: Throughput / signal-timing cap: no more than two intersections
        in the plan may be in a Green state simultaneously (Green_NS,
        Green_EW or Emergency_Green). This models the "signal timing
        must stay within allowed range" requirement of the spec.
"""


# ---------------------------------------------------------------------------
# CSP definition
# ---------------------------------------------------------------------------
DEFAULT_VARIABLES = [
    "Central_Junction",
    "East_Market",
    "West_Terminal",
    "South_Residential",
    "City_Hospital",
]

DEFAULT_DOMAIN = ["Green_NS", "Green_EW", "Red", "Emergency_Green", "Hold"]

# Adjacency between intersections (undirected)
ADJACENCY = {
    "Central_Junction":   {"East_Market", "West_Terminal", "South_Residential"},
    "East_Market":        {"Central_Junction", "South_Residential", "City_Hospital"},
    "West_Terminal":      {"Central_Junction"},
    "South_Residential":  {"Central_Junction", "East_Market", "City_Hospital"},
    "City_Hospital":      {"East_Market", "South_Residential"},
}


# ---------------------------------------------------------------------------
class CSPScheduler:
    """
    Backtracking CSP solver for the traffic-control allocation problem.
    """

    def __init__(self, emergency_target=None, has_emergency=False):
        """
        Build a fresh CSP instance.

        Parameters
        ----------
        emergency_target : str or None
            The intersection (variable) that should host the emergency
            corridor green light. If None, the solver picks one
            automatically when has_emergency is True.
        has_emergency : bool
            True when the request has an authorized emergency corridor
            (i.e. the Logic module produced Authorized(EmergencyRoute)).
        """
        self.variables = list(DEFAULT_VARIABLES)
        self.domains = {v: list(DEFAULT_DOMAIN) for v in self.variables}
        self.has_emergency = has_emergency
        self.emergency_target = emergency_target if emergency_target in self.variables else None

        # Tighten domains based on the situation
        if not self.has_emergency:
            # Emergency_Green only allowed when an emergency is authorized
            for v in self.variables:
                if "Emergency_Green" in self.domains[v]:
                    self.domains[v].remove("Emergency_Green")

        # Hospital intersection can only be Red or Emergency_Green
        self.domains["City_Hospital"] = [
            d for d in self.domains["City_Hospital"]
            if d in ("Red", "Emergency_Green")
        ]

        self.steps = []   # explanation log

    # -----------------------------------------------------------------
    def _consistent(self, var, value, assignment):
        """
        Return True iff assigning `value` to `var` is consistent with
        all currently assigned variables and all hard constraints.
        Records a step description on success and a violation note on
        failure (when verbose backtracking would be useful).
        """
        # C1: No two adjacent variables both Green_EW
        if value == "Green_EW":
            for n in ADJACENCY.get(var, set()):
                if assignment.get(n) == "Green_EW":
                    return False

        # C2: at most one Emergency_Green
        if value == "Emergency_Green":
            for v, val in assignment.items():
                if v != var and val == "Emergency_Green":
                    return False

        # C3: enforce target if specified
        if self.emergency_target == var:
            if self.has_emergency and value != "Emergency_Green":
                return False
        else:
            # If a target is specified, no other variable may take Emergency_Green
            if self.emergency_target is not None and value == "Emergency_Green":
                return False

        # C5: neighbours of an Emergency_Green node cannot be a generic green
        if value in ("Green_NS", "Green_EW"):
            for n in ADJACENCY.get(var, set()):
                if assignment.get(n) == "Emergency_Green":
                    return False
        if value == "Emergency_Green":
            for n in ADJACENCY.get(var, set()):
                if assignment.get(n) in ("Green_NS", "Green_EW"):
                    return False

        # C6: throughput cap - at most 2 simultaneous greens in the plan
        # (any flavour: Green_NS, Green_EW, Emergency_Green).
        green_states = {"Green_NS", "Green_EW", "Emergency_Green"}
        if value in green_states:
            already_green = sum(1 for v in assignment.values()
                                if v in green_states)
            if already_green >= 2:
                return False

        return True

    # -----------------------------------------------------------------
    def _select_unassigned(self, assignment):
        """Pick the next unassigned variable (preserves listed order)."""
        for v in self.variables:
            if v not in assignment:
                return v
        return None

    # -----------------------------------------------------------------
    def _ordered_domain_values(self, var):
        """
        Order domain values so emergency comes first when relevant,
        otherwise prefer Hold/Red over generic greens (defensive default).
        """
        if self.has_emergency and "Emergency_Green" in self.domains[var]:
            return ["Emergency_Green"] + [d for d in self.domains[var]
                                          if d != "Emergency_Green"]
        # Default order: prefer safer choices first
        order = ["Hold", "Red", "Green_NS", "Green_EW", "Emergency_Green"]
        return [d for d in order if d in self.domains[var]]

    # -----------------------------------------------------------------
    def _backtrack(self, assignment):
        """
        Recursive backtracking core. Returns a complete consistent
        assignment dict or None if no solution exists.
        """
        if len(assignment) == len(self.variables):
            return dict(assignment)

        var = self._select_unassigned(assignment)
        for value in self._ordered_domain_values(var):
            if self._consistent(var, value, assignment):
                assignment[var] = value
                self.steps.append(f"Assign {var} = {value}")
                result = self._backtrack(assignment)
                if result is not None:
                    return result
                # Undo
                self.steps.append(f"Backtrack on {var} (was {value})")
                del assignment[var]
        return None

    # -----------------------------------------------------------------
    def solve(self):
        """
        Run the solver and produce a structured result dictionary.
        """
        # If an emergency target was named but is not in the variables,
        # treat as "no specific target" rather than failing outright.
        assignment = self._backtrack({})

        if assignment is None:
            return {
                "success": False,
                "assignment": None,
                "constraints_satisfied": [],
                "explanation": "No valid signal assignment could be found.",
                "steps": list(self.steps),
            }

        constraints_passed = [
            "C1: No adjacent Green_EW conflicts",
            "C2: At most one Emergency_Green",
            "C5: Emergency_Green isolated from generic greens",
            "C6: Throughput cap (<= 2 simultaneous greens)",
        ]
        if self.has_emergency:
            constraints_passed.append(
                "C3: Emergency corridor placed at "
                f"{self.emergency_target or 'auto-selected node'}"
            )
        constraints_passed.append("C4: City_Hospital restricted to Red/Emergency_Green")

        if self.has_emergency and self.emergency_target:
            explanation = (
                f"Emergency_Green allocated at {self.emergency_target}; "
                "all other intersections coordinated under safety constraints."
            )
        elif self.has_emergency:
            explanation = (
                "Emergency corridor honored at automatically selected "
                "intersection while preserving safety constraints."
            )
        else:
            explanation = (
                "Standard control plan generated with no emergency overrides; "
                "all signal-conflict and hospital constraints satisfied."
            )

        return {
            "success": True,
            "assignment": assignment,
            "constraints_satisfied": constraints_passed,
            "explanation": explanation,
            "steps": list(self.steps),
        }


# ---------------------------------------------------------------------------
def allocate_controls(cleaned_request, logic_result):
    """
    Public entry-point used by the rest of the project.

    Combines information from the cleaned request and the Logic module
    output to configure and run the CSP solver. The function decides:

      * has_emergency  -> True if Authorized(EmergencyRoute) OR any
                          Authorized(SignalOverride(<zone>)) is present
      * target         -> the specific intersection that should receive
                          Emergency_Green, derived (in priority order)
                          from a SignalOverride zone, the destination
                          (when it is a hospital), or the request's
                          control_zone field.
    """
    authorized = list(logic_result.get("authorized_actions") or [])
    has_emergency_route = "EmergencyRoute" in authorized

    # Detect SignalOverride(Zone) actions and pull out the zone name
    override_zone = None
    for action in authorized:
        if action.startswith("SignalOverride(") and action.endswith(")"):
            override_zone = action[len("SignalOverride("):-1]
            break

    has_emergency = has_emergency_route or (override_zone is not None)

    # Priority for picking the Emergency_Green target
    target = override_zone
    if target is None and has_emergency_route \
            and cleaned_request.get("destination") == "City_Hospital":
        target = "City_Hospital"
    if target is None:
        zone = cleaned_request.get("control_zone")
        if zone and zone != "None":
            target = zone

    solver = CSPScheduler(emergency_target=target, has_emergency=has_emergency)
    return solver.solve()

"""
knowledge_base.py
-----------------
Module 4: Logic / Knowledge Base.

A symbolic, rule-based reasoning engine. It implements the predicates
and inference rules defined in the project specification using a
forward-chaining style: facts are derived from the request, then rules
fire repeatedly until no new facts can be added.

The output of this module decides:
    * the rule-based priority level
    * which actions are authorized
    * which actions are allowed (effectively granted)
    * whether the request as a whole is Approved or Rejected
    * a human-readable trace of which rules fired
"""


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
EMERGENCY_VEHICLES = {"Police", "Fire", "Ambulance"}
HOSPITALS = {"City_Hospital"}     # locations classified as hospitals
SIGNAL_ZONES = {                   # locations regarded as signal-controlled
    "Central_Junction", "East_Market", "North_Station",
    "West_Terminal", "South_Residential", "Traffic_Control_Center",
}


# ===========================================================================
class KnowledgeBase:
    """
    A small forward-chaining knowledge base scoped to one request.
    Each call to evaluate(request) creates a fresh instance so there is
    no cross-request leakage of facts.
    """

    # -----------------------------------------------------------------
    def __init__(self, request):
        """
        Seed the KB with the base facts derivable directly from the
        cleaned request dictionary.
        """
        self.request = request
        # facts is a set of tuples representing predicates, e.g.
        #   ("EmergencyVehicle", "Ambulance")
        #   ("Authorized", "Ambulance", "EmergencyRoute")
        self.facts = set()
        self.fired_rules = []   # ordered explanation log
        self._seed_initial_facts()

    # -----------------------------------------------------------------
    def _seed_initial_facts(self):
        """Convert the request fields into atomic logical facts."""
        v = self.request["vehicle_type"]
        loc = self.request["current_location"]
        dest = self.request["destination"]

        self.facts.add(("Vehicle", v))
        self.facts.add(("Location", loc))
        self.facts.add(("Location", dest))
        self.facts.add(("CurrentLocation", v, loc))
        self.facts.add(("Destination", v, dest))
        self.facts.add(("RequestType", self.request["request_id"],
                        self.request["request_category"]))

        if v in EMERGENCY_VEHICLES:
            self.facts.add(("EmergencyVehicle", v))
        else:
            self.facts.add(("CivilianVehicle", v))

        if dest in HOSPITALS:
            self.facts.add(("Hospital", dest))

        zone = self.request.get("control_zone", "None")
        if zone in SIGNAL_ZONES:
            self.facts.add(("SignalZone", zone))

        self.facts.add(("IncidentSeverity", v, self.request["incident_severity"]))
        if self.request["time_sensitivity"] == "High":
            self.facts.add(("TimeSensitive", v))

    # -----------------------------------------------------------------
    def _has(self, *pattern):
        """
        Helper: return True iff at least one fact matches the given
        pattern (None acts as a wildcard).
        """
        for fact in self.facts:
            if len(fact) != len(pattern):
                continue
            if all(p is None or p == f for p, f in zip(pattern, fact)):
                return True
        return False

    # -----------------------------------------------------------------
    def _add(self, fact, rule_description):
        """
        Add a derived fact and record which rule produced it.
        Returns True if the fact is genuinely new.
        """
        if fact in self.facts:
            return False
        self.facts.add(fact)
        self.fired_rules.append(f"{rule_description}  ==>  {fact}")
        return True

    # -----------------------------------------------------------------
    def evaluate(self):
        """
        Apply rules repeatedly until the fact set stabilizes (a simple
        forward-chaining fixed-point loop). Returns a structured result
        dictionary summarizing the inferences.

        Order matters here: derivations come first (R1-R8), then the
        Critical-priority adjustments (R9, R10), then the specific
        request-type approval/rejection rules (R13-R19), and only then
        the general fallback rules (R11, R12) which would otherwise
        short-circuit the more informative specific rules in the trace.
        """
        changed = True
        while changed:
            changed = False
            # Layer 1: priority + authorization derivations
            changed |= self._rule_priority_critical()         # R1
            changed |= self._rule_priority_high()             # R2
            changed |= self._rule_priority_normal()           # R3
            changed |= self._rule_authorize_signal_override() # R4
            changed |= self._rule_civilian_no_signal_override() # R5
            changed |= self._rule_emergency_corridor()        # R6
            changed |= self._rule_corridor_authorizes_route() # R7
            changed |= self._rule_authorized_implies_allowed() # R8
            changed |= self._rule_critical_blocks_unauthorized() # R9
            changed |= self._rule_critical_signal_override()  # R10
            # Layer 2: request-type-specific decisions (fire FIRST so the
            # trace highlights the strongest applicable rule per category)
            changed |= self._rule_route_request_approval()         # R13
            changed |= self._rule_policy_check_approval()          # R14, R15
            changed |= self._rule_control_allocation_approval()    # R16, R17
            changed |= self._rule_emergency_response_approval()    # R18
            changed |= self._rule_integrated_service_approval()    # R19
            # Layer 3: general fallback rules (only fire if a specific
            # rule above has not already produced a verdict)
            changed |= self._rule_allowed_implies_approved()  # R11
            changed |= self._rule_not_allowed_implies_rejected() # R12

        return self._summarize()

    # ===========================================================
    # Individual rules - each returns True if it added new facts
    # ===========================================================

    # EmergencyVehicle(v) AND IncidentSeverity(v,High) -> Priority(v,Critical)
    def _rule_priority_critical(self):
        v = self.request["vehicle_type"]
        if self._has("EmergencyVehicle", v) and \
           self._has("IncidentSeverity", v, "High"):
            return self._add(("Priority", v, "Critical"),
                             "R1: EmergencyVehicle ∧ Severity=High")
        return False

    # EmergencyVehicle(v) AND TimeSensitive(v) -> Priority(v, High)
    def _rule_priority_high(self):
        v = self.request["vehicle_type"]
        if self._has("EmergencyVehicle", v) and self._has("TimeSensitive", v) \
           and not self._has("Priority", v, "Critical"):
            return self._add(("Priority", v, "High"),
                             "R2: EmergencyVehicle ∧ TimeSensitive")
        return False

    # CivilianVehicle(v) -> Priority(v, Normal)
    def _rule_priority_normal(self):
        v = self.request["vehicle_type"]
        if self._has("CivilianVehicle", v) and not self._has("Priority", v, None):
            return self._add(("Priority", v, "Normal"),
                             "R3: CivilianVehicle ⇒ Priority=Normal")
        return False

    # EmergencyVehicle(v) AND SignalZone(z) -> Authorized(v, SignalOverride(z))
    def _rule_authorize_signal_override(self):
        v = self.request["vehicle_type"]
        if not self._has("EmergencyVehicle", v):
            return False
        added = False
        for fact in list(self.facts):
            if fact[0] == "SignalZone":
                z = fact[1]
                added |= self._add(("Authorized", v, f"SignalOverride({z})"),
                                   f"R4: EmergencyVehicle ∧ SignalZone({z})")
        return added

    # CivilianVehicle(v) AND SignalZone(z) -> NOT Authorized(v, SignalOverride(z))
    def _rule_civilian_no_signal_override(self):
        v = self.request["vehicle_type"]
        if not self._has("CivilianVehicle", v):
            return False
        added = False
        for fact in list(self.facts):
            if fact[0] == "SignalZone":
                z = fact[1]
                added |= self._add(("NotAuthorized", v, f"SignalOverride({z})"),
                                   f"R5: CivilianVehicle ∧ SignalZone({z})")
        return added

    # EmergencyVehicle(v) AND Destination(v,h) AND Hospital(h)
    #                                     -> EmergencyCorridor(v)
    def _rule_emergency_corridor(self):
        v = self.request["vehicle_type"]
        dest = self.request["destination"]
        if self._has("EmergencyVehicle", v) and self._has("Hospital", dest):
            return self._add(("EmergencyCorridor", v),
                             "R6: EmergencyVehicle ∧ Destination is Hospital")
        return False

    # EmergencyCorridor(v) -> Authorized(v, EmergencyRoute)
    def _rule_corridor_authorizes_route(self):
        v = self.request["vehicle_type"]
        if self._has("EmergencyCorridor", v):
            return self._add(("Authorized", v, "EmergencyRoute"),
                             "R7: EmergencyCorridor ⇒ Authorized(EmergencyRoute)")
        return False

    # Authorized(v,action) -> AllowedAction(v,action)
    def _rule_authorized_implies_allowed(self):
        added = False
        for fact in list(self.facts):
            if fact[0] == "Authorized":
                v, action = fact[1], fact[2]
                added |= self._add(("AllowedAction", v, action),
                                   f"R8: Authorized({v},{action}) ⇒ AllowedAction")
        return added

    # Priority(v,Critical) AND NOT Authorized(v,action) ->
    #                                  NOT AllowedAction(v,action)
    # General "critical-without-authorization is blocked" rule. We mark
    # any *not-authorized* action as also not-allowed when the vehicle
    # has Critical priority. This makes the rule visible in the trace
    # for civilian-Critical edge cases.
    def _rule_critical_blocks_unauthorized(self):
        v = self.request["vehicle_type"]
        if not self._has("Priority", v, "Critical"):
            return False
        added = False
        for fact in list(self.facts):
            if fact[0] == "NotAuthorized" and fact[1] == v:
                action = fact[2]
                added |= self._add(
                    ("NotAllowedAction", v, action),
                    f"R9: Priority(Critical) ∧ ¬Authorized({action}) ⇒ "
                    f"¬AllowedAction({action})"
                )
        return added

    # Priority(v,Critical) AND Authorized(v,EmergencyRoute) ->
    #                                  AllowedAction(v, SignalOverride)
    def _rule_critical_signal_override(self):
        v = self.request["vehicle_type"]
        if self._has("Priority", v, "Critical") and \
           self._has("Authorized", v, "EmergencyRoute"):
            return self._add(("AllowedAction", v, "SignalOverride"),
                             "R10: Priority(Critical) ∧ Authorized(EmergencyRoute) "
                             "⇒ AllowedAction(SignalOverride)")
        return False

    # AllowedAction(v,action) -> Approved(v,req)  [GENERAL approval rule]
    # The request-type-specific approval rules are tightened versions of
    # this. Firing it explicitly makes the trace match the spec exactly.
    def _rule_allowed_implies_approved(self):
        v = self.request["vehicle_type"]
        rid = self.request["request_id"]
        if ("Approved", rid) in self.facts:
            return False
        # Only fire for request types where any AllowedAction is sufficient
        # to imply approval - this is the case for Control_Allocation and
        # Integrated requests in the spec rule list.
        cat = self.request["request_category"]
        if cat not in ("Control_Allocation_Request",
                       "Integrated_City_Service_Request",
                       "Emergency_Response_Request"):
            return False
        for fact in self.facts:
            if fact[0] == "AllowedAction" and fact[1] == v:
                return self._add(
                    ("Approved", rid),
                    f"R11: AllowedAction({fact[2]}) ⇒ Approved(req)"
                )
        return False

    # NOT AllowedAction(v,action) -> Rejected(v,req)
    # General rejection rule: if the vehicle has any explicitly-blocked
    # action AND no approving allowed action, mark the request rejected.
    def _rule_not_allowed_implies_rejected(self):
        v = self.request["vehicle_type"]
        rid = self.request["request_id"]
        if ("Rejected", rid) in self.facts or ("Approved", rid) in self.facts:
            return False
        # Need at least one NotAuthorized OR NotAllowedAction fact to fire
        has_block = any(
            (f[0] in ("NotAuthorized", "NotAllowedAction") and f[1] == v)
            for f in self.facts
        )
        # And no AllowedAction that would otherwise approve
        has_allow = any(f[0] == "AllowedAction" and f[1] == v
                        for f in self.facts)
        if has_block and not has_allow:
            return self._add(
                ("Rejected", rid),
                "R12: ¬AllowedAction ⇒ Rejected(req)"
            )
        return False

    # ---------------- request-type approvals --------------------------

    # RequestType(req,Route_Request) -> Approved(v,req)
    def _rule_route_request_approval(self):
        if self.request["request_category"] == "Route_Request":
            return self._add(("Approved", self.request["request_id"]),
                             "R13: Route_Request ⇒ Approved")
        return False

    # Policy_Check: approved iff vehicle has at least one Authorized action,
    # otherwise rejected.
    def _rule_policy_check_approval(self):
        if self.request["request_category"] != "Policy_Check":
            return False
        v = self.request["vehicle_type"]
        has_auth = any(f[0] == "Authorized" and f[1] == v for f in self.facts)
        rid = self.request["request_id"]
        if has_auth:
            return self._add(("Approved", rid),
                             "R14: Policy_Check ∧ Authorized(action) ⇒ Approved")
        return self._add(("Rejected", rid),
                         "R15: Policy_Check ∧ ¬Authorized ⇒ Rejected")

    # Control_Allocation_Request: approved iff vehicle has any AllowedAction
    def _rule_control_allocation_approval(self):
        if self.request["request_category"] != "Control_Allocation_Request":
            return False
        v = self.request["vehicle_type"]
        has_allow = any(f[0] == "AllowedAction" and f[1] == v for f in self.facts)
        rid = self.request["request_id"]
        if has_allow:
            return self._add(("Approved", rid),
                             "R16: Control_Allocation ∧ AllowedAction ⇒ Approved")
        return self._add(("Rejected", rid),
                         "R17: Control_Allocation ∧ ¬AllowedAction ⇒ Rejected")

    # Emergency_Response_Request: needs Priority + Authorized(EmergencyRoute)
    def _rule_emergency_response_approval(self):
        if self.request["request_category"] != "Emergency_Response_Request":
            return False
        v = self.request["vehicle_type"]
        rid = self.request["request_id"]
        has_priority = any(f[0] == "Priority" and f[1] == v for f in self.facts)
        if has_priority and self._has("Authorized", v, "EmergencyRoute"):
            return self._add(("Approved", rid),
                             "R18: Emergency_Response ∧ Priority ∧ "
                             "Authorized(EmergencyRoute) ⇒ Approved")
        return False

    # Integrated_City_Service_Request: needs Critical priority +
    #                Authorized(EmergencyRoute) + an AllowedAction
    def _rule_integrated_service_approval(self):
        if self.request["request_category"] != "Integrated_City_Service_Request":
            return False
        v = self.request["vehicle_type"]
        rid = self.request["request_id"]
        if self._has("Priority", v, "Critical") and \
           self._has("Authorized", v, "EmergencyRoute") and \
           any(f[0] == "AllowedAction" and f[1] == v for f in self.facts):
            return self._add(("Approved", rid),
                             "R19: Integrated_City_Service ∧ Critical ∧ "
                             "Authorized(EmergencyRoute) ∧ AllowedAction ⇒ Approved")
        return False

    # -----------------------------------------------------------------
    def _summarize(self):
        """Build the structured result dict returned to callers."""
        v = self.request["vehicle_type"]
        rid = self.request["request_id"]

        # Collect priority (highest if multiple)
        priority_order = {"Low": 0, "Normal": 1, "High": 2, "Critical": 3}
        prio = None
        for fact in self.facts:
            if fact[0] == "Priority" and fact[1] == v:
                if prio is None or priority_order.get(fact[2], 0) > priority_order.get(prio, 0):
                    prio = fact[2]

        authorized_actions = sorted({f[2] for f in self.facts
                                     if f[0] == "Authorized" and f[1] == v})
        allowed_actions = sorted({f[2] for f in self.facts
                                  if f[0] == "AllowedAction" and f[1] == v})
        not_authorized = sorted({f[2] for f in self.facts
                                 if f[0] == "NotAuthorized" and f[1] == v})
        not_allowed = sorted({f[2] for f in self.facts
                              if f[0] == "NotAllowedAction" and f[1] == v})

        approved = ("Approved", rid) in self.facts
        rejected = ("Rejected", rid) in self.facts
        if approved and rejected:
            # Approval wins if both somehow fired
            rejected = False
        status = "Approved" if approved else ("Rejected" if rejected else "Pending")

        return {
            "rule_priority": prio,
            "authorized_actions": authorized_actions,
            "allowed_actions": allowed_actions,
            "not_authorized_actions": not_authorized,
            "not_allowed_actions": not_allowed,
            "status": status,
            "rules_fired": list(self.fired_rules),
        }


# ---------------------------------------------------------------------------
def evaluate_request(cleaned_request):
    """
    Convenience entry point used by the rest of the project.
    Builds a fresh KnowledgeBase, runs forward-chaining, and returns
    the summary dictionary.
    """
    return KnowledgeBase(cleaned_request).evaluate()

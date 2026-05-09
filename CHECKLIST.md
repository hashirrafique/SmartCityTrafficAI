# Final-Marks Checklist

This file maps every requirement in the project specification and the
AL-2002 rubric to the file(s) that satisfy it. Use it as a quick
self-grading sheet.

---

## 1. Marking rubric (100 marks)

| # | Component | Marks | Where it lives |
|---|---|---|---|
| 1 | Input & Preprocessing | 10 | `modules/preprocessing.py` |
| 2 | ANN Priority Prediction | 20 | `modules/ann_priority.py` (multiclass + binary baseline) + `data/ann_training_data.py` |
| 3 | Logic / Knowledge Base | 20 | `modules/knowledge_base.py` (all 19 rules R1–R19) |
| 4 | CSP Scheduler / Control Allocation | 15 | `modules/csp_scheduler.py` (custom backtracking, 6 constraints C1–C6) |
| 5 | Search & Navigation | 15 | `modules/search_navigation.py` (BFS + UCS + A*) |
| 6 | Final Response & Integration | 10 | `modules/final_response.py` + `main.py` |
| 7 | Documentation & Code Quality | 10 | docstrings on every function, `README.md`, `report/Project_Report.docx`, this file |

---

## 2. Per-module checklist

### Module 1 — Input & Preprocessing
- [x] Receives structured request (dictionary)
- [x] Validates every required field is present
- [x] Rejects invalid vehicle types, request categories, locations, severities, densities
- [x] Case-insensitive normalization (`"ambulance"` → `"Ambulance"`)
- [x] Builds clean internal request object
- [x] Builds 6-feature numeric vector for ANN
- [x] Categorical mapping for `vehicle_type`, `request_category`, `incident_severity`, `time_sensitivity`, `traffic_density`, `priority_claim`
- [x] No NLP — structured input only

### Module 2 — Request Router
- [x] Selects correct pipeline per `request_category`
- [x] Route_Request → Search only
- [x] Policy_Check → Logic only
- [x] Control_Allocation_Request → Logic → CSP
- [x] Emergency_Response_Request → ANN → Logic → CSP → Search
- [x] Integrated_City_Service_Request → ANN → Logic → CSP → Search → Final Response

### Module 3 — ANN Priority
- [x] Multiclass classifier (Low / Normal / High / Critical) — preferred path
- [x] Binary baseline classifier (urgent / non-urgent) — Option A from spec
- [x] scikit-learn `MLPClassifier`
- [x] Manually curated training set (42 rows)
- [x] All 6 features used: vehicle_type, severity, time_sensitivity, density, distance, priority_claim
- [x] Reports training accuracy at startup
- [x] Returns predicted level + confidence + explanation
- [x] Robust to invalid feature vectors (raises `ValueError`)
- [x] **Does not** decide authorization (only urgency)

### Module 4 — Logic / Knowledge Base
- [x] Implements every predicate in the spec: `Vehicle`, `EmergencyVehicle`, `CivilianVehicle`, `Location`, `SignalZone`, `Hospital`, `Request`, `RequestType`, `CurrentLocation`, `Destination`, `IncidentSeverity`, `TimeSensitive`, `Priority`, `Authorized`, `AllowedAction`, `EmergencyCorridor`, `EmergencyRoute`, `SignalOverride`, `Approved`, `Rejected`
- [x] All **19 spec rules** present, labelled R1–R19 in the firing trace
- [x] Forward-chaining loop until fixpoint
- [x] Returns priority, authorized actions, allowed actions, status, full rule-firing trace

### Module 5 — CSP Scheduler
- [x] 5 intersections: Central_Junction, East_Market, West_Terminal, South_Residential, City_Hospital
- [x] 5 domain values: Green_NS, Green_EW, Red, Emergency_Green, Hold
- [x] Custom backtracking solver
- [x] 6 constraints (C1 conflict greens, C2 single emergency, C3 corridor placement, C4 hospital restricted, C5 corridor isolation, C6 throughput cap / signal-timing range)
- [x] Returns assignment + constraints satisfied + step trace + explanation
- [x] Reports failure cleanly when no solution exists

### Module 6 — Search & Navigation
- [x] All 13 specified locations
- [x] BFS for unweighted shortest path
- [x] UCS for weighted shortest path
- [x] A* with admissible heuristic (Floyd–Warshall precomputed)
- [x] Predefined graph with realistic edge weights
- [x] Validates source/destination, raises on bad input
- [x] Returns path + cost + algorithm name + explanation

### Module 7 — Final Response
- [x] Aggregates only modules that actually ran
- [x] Does **not** show ANN result for plain `Route_Request`
- [x] Does **not** show CSP result for `Policy_Check`
- [x] Includes route, predicted priority, policy status, control plan, decision, explanation
- [x] Pretty-prints to terminal via `render_response`

---

## 3. General coding requirements (PDF instructions)

- [x] 3-student group submission (insert real roll numbers in title page + ZIP filename)
- [x] Proper indentation throughout
- [x] Block comment / docstring at the start of every function
- [x] Exception handling everywhere user input enters the system
- [x] Understandable variable names
- [x] Sample test scenarios (`tests/test_scenarios.py` — 36 assertions)
- [x] All possible screenshot placeholders in the report (14 of them)
- [x] Submitted as a ZIP

---

## 4. Demo scenarios

- [x] Scenario 1 — Civilian Stadium → City_Hospital (Search only)
- [x] Scenario 2 — Civilian SignalOverride request (Logic only, expected Reject)
- [x] Scenario 3 — Fire truck control allocation (Logic → CSP)
- [x] Scenario 4 — Ambulance Central_Junction → City_Hospital (ANN → Logic → CSP → Search)
- [x] Scenario 5 — Integrated emergency corridor (ANN → Logic → CSP → Search → Final)

All five run via the CLI menu (option 6 = run all).

---

## 5. Submission

1. Replace `23F-XXXX` placeholders on the title page of `report/Project_Report.docx` with the three actual group roll numbers.
2. Take screenshots and paste them into the dashed-border placeholders inside the Word report.
3. Re-zip with the correct filename:
   ```
   23F-1234_23F-5678_23F-9012_Project.zip
   ```
4. Upload to Google Classroom from one group member's account.

---

## 6. Self-check before submitting

```bash
python -m tests.test_scenarios     # expect: 36 passed, 0 failed
python main.py                      # walk through option 6 and verify outputs
```

If both succeed, every rubric component is covered.

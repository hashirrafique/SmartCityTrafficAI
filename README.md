# Smart City Traffic & Emergency Response AI System

**AI Lab Final Project — FAST-NUCES (CFD Campus)**
**Course:** AL2002 — Artificial Intelligence Lab
**Semester:** Spring 2026

---

## 1. Overview

Modern cities must move ordinary traffic efficiently while still letting
ambulances, fire units, and police vehicles pass quickly through congested
corridors. Doing both at the same time is hard because each request has
different requirements: a civilian needs a route, an ambulance needs
priority + signal coordination + a route, an operator just needs a policy
check.

This project models that decision-making as an **integrated AI problem**.
A single structured request enters the system; based on its `request_category`
it is routed through only the AI modules it actually needs. The result is a
single, explainable response covering route, priority, control plan, and
authorization status.

---

## 2. AI Concepts Used

| Concept | Where it appears |
|---|---|
| Artificial Neural Network — multiclass MLP (Low/Normal/High/Critical) | `modules/ann_priority.py` |
| Artificial Neural Network — binary baseline MLP (urgent / non-urgent) | `modules/ann_priority.py` |
| Rule-based logic / Knowledge base (forward chaining, all 19 spec rules) | `modules/knowledge_base.py` |
| Constraint Satisfaction Problem (custom backtracking, 6 constraints) | `modules/csp_scheduler.py` |
| BFS — unweighted shortest path | `modules/search_navigation.py` |
| UCS — uniform-cost weighted search | `modules/search_navigation.py` |
| A* — informed search with admissible heuristic (Floyd–Warshall) | `modules/search_navigation.py` |
| Modular pipeline routing | `modules/request_router.py` |

---

## 3. Modules

1. **Input & Preprocessing** (`preprocessing.py`)
   Validates required fields, normalizes values, rejects invalid inputs,
   and prepares numeric feature vectors for the ANN.

2. **Request Router** (`request_router.py`)
   Maps `request_category` to the ordered list of AI modules to run.
   Prevents wrong modules from being invoked.

3. **ANN Priority Predictor** (`ann_priority.py`)
   Two `MLPClassifier` networks trained at startup on 42 manually curated
   examples. The **multiclass head** (`hidden_layer_sizes=(16, 8)`) predicts
   urgency in four classes — `Low / Normal / High / Critical`. The **binary
   baseline head** (`hidden_layer_sizes=(8,)`) predicts `urgent` vs
   `non-urgent` for direct comparison. Both heads achieve 100 % training
   accuracy.

4. **Logic / Knowledge Base** (`knowledge_base.py`)
   Implements every predicate and **all 19 inference rules** (R1–R19)
   from the specification, using forward chaining until fixed point.
   Returns priority, authorized actions, allowed actions, status, and a
   complete rule-firing trace.

5. **CSP Scheduler** (`csp_scheduler.py`)
   Custom backtracking solver assigning signal states to 5 intersections
   under **6 explicit constraints (C1–C6)**: signal-conflict avoidance,
   emergency corridor placement, hospital restriction, corridor isolation,
   and a throughput cap modelling signal-timing range.

6. **Search & Navigation** (`search_navigation.py`)
   BFS, UCS, and A* over a 13-node city graph. A* uses an admissible
   heuristic precomputed via Floyd–Warshall.

7. **Final Response** (`final_response.py`)
   Aggregates only the modules that ran into one clean output and a
   plain-language decision message.

---

## 4. Folder Structure

```
SmartCityTrafficAI/
├── main.py                 # CLI menu / orchestrator
├── README.md
├── CHECKLIST.md            # final-marks checklist
├── requirements.txt
├── data/
│   ├── ann_training_data.py
│   └── sample_requests.py
├── modules/
│   ├── preprocessing.py
│   ├── request_router.py
│   ├── ann_priority.py
│   ├── knowledge_base.py
│   ├── csp_scheduler.py
│   ├── search_navigation.py
│   └── final_response.py
├── tests/
│   └── test_scenarios.py
└── report/
    └── Project_Report.docx
```

---

## 5. Installation

Requires **Python 3.9+**.

```bash
# 1. Unzip the project
unzip 23F-XXXX_23F-XXXX_23F-XXXX_Project.zip
cd SmartCityTrafficAI

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. Running the Project

From the project root:

```bash
python main.py
```

You will see the main menu:

```
MAIN MENU
---------
  1. Run Scenario 1 - Civilian Route Request (Search only)
  2. Run Scenario 2 - Civilian Policy Check (Logic only - rejected)
  3. Run Scenario 3 - Fire Truck Control Allocation (Logic + CSP)
  4. Run Scenario 4 - Ambulance Emergency (ANN + Logic + CSP + Search)
  5. Run Scenario 5 - Integrated City Service (full pipeline)
  6. Run ALL scenarios (1 to 5)
  7. Enter a CUSTOM request
  8. Show ANN training accuracy
  9. Show city graph (locations & weights)
  0. Exit
```

### Running the test harness

```bash
python -m tests.test_scenarios
```

Expected: **36 PASS lines and exit code 0**.

---

## 7. Sample Scenarios

| # | Vehicle | Category | Expected modules | Expected outcome |
|---|---|---|---|---|
| 1 | Civilian | Route_Request | Search | Path Stadium → … → City_Hospital |
| 2 | Civilian | Policy_Check | Logic | Rejected (no SignalOverride for civilians) |
| 3 | Fire | Control_Allocation_Request | Logic + CSP | Signal plan with Emergency_Green |
| 4 | Ambulance | Emergency_Response_Request | ANN + Logic + CSP + Search | Critical priority, approved, route returned |
| 5 | Ambulance | Integrated_City_Service_Request | full pipeline | Critical, corridor allocated, route generated |

---

## 8. Submission

```bash
# From the project root
cd ..
zip -r 23F-XXXX_23F-XXXX_23F-XXXX_Project.zip SmartCityTrafficAI
```

Submit the resulting ZIP to Google Classroom along with the Word report
located in `report/Project_Report.docx`.

---

## 9. Notes

- This is an academic simulation only. There are no live sensors or APIs.
- All inputs are structured (no NLP).
- The ANN does not decide authorization; it only predicts urgency.
  Authorization is decided by the rule-based knowledge base.
- Every public function has a top-of-block docstring describing its
  purpose. Exception handling is added for invalid input throughout.

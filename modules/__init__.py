"""
modules/
--------
AI pipeline modules for the Smart City Traffic & Emergency Response system.

Submodules
----------
preprocessing       Input validation, normalization, and ANN feature building
request_router      Maps request categories to the correct processing pipeline
ann_priority        Multiclass & binary MLP priority prediction (scikit-learn)
knowledge_base      Forward-chaining rule engine (19 spec rules, R1–R19)
csp_scheduler       Backtracking CSP solver for signal-control allocation
search_navigation   BFS, UCS, and A* over the 13-node city graph
final_response      Aggregates module outputs into a single explainable response
"""

"""
Placeholder for monitoring tests.

This file is intentionally empty as part of the selective TDD approach.

Design Intent:
--------------
Drift monitoring thresholds are business decisions, not correctness criteria.
The monitoring logic is validated through demonstration and manual inspection
rather than unit tests.

Validation Approach:
--------------------
- Demonstration script: src/monitoring/drift_detector.py (run with -m flag)
- Shows normal vs. drifted scenarios with clear output
- Threshold effectiveness validated through production monitoring

Current Validation:
-------------------
- Statistical drift calculation: Verified through demonstration
- Alert triggering: Verified through demonstration (30% threshold)
- Report formatting: Verified through manual inspection

Rationale:
----------
Drift detection is inherently subjective. A 30% threshold may be appropriate
for one feature but too sensitive for another. Testing drift detection would
require arbitrary assertions about what constitutes "correct" drift, which
provides false confidence. Instead, we demonstrate the monitoring system
works and let production data inform threshold tuning.

If monitoring logic becomes more complex (e.g., multiple drift metrics,
adaptive thresholds), targeted tests for the calculation logic should be
added here.
"""

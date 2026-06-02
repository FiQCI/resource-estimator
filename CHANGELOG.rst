=========
Changelog
=========

Version 0.5.0
==========================

* Add Aalto Q20 (``aalto-q20``) as a supported quantum computer.
  * Add device card and image to the device selector UI.
  * Disable depth input for Aalto Q20 (depth has no measurable effect on QPU time).
* Introduce qubit-scaled analytical model (``analytical-qubit``) for fitting devices where per-shot cost depends on qubit count.
  * Formula: ``T = T_init + η(B) × B × shots × (α_base + α_q × qubits)``
  * Cross-validated R² = 0.98 for Aalto Q20, versus 0.83 for the polynomial alternative.
* Fix incorrect ``use_log_transform`` property name in JavaScript model (was ``logTransform``).
* Add Aalto Q20 section to documentation including model formula, parameters, and actual-vs-predicted plot.

Version 0.4.0
==========================

* Overhauled VTT Q50 model to use an analytical model without a depth parameters
  * Not using depth avoids numerical instability and unphysical behavior while maintaining high accuracy
* Updated UI to remove depth input and reflect new model structure
* Modify the CLI build process to allow both Helmi (polynomial model) and Q50 (analytical model) approaches
* Update the documentation.

Version 0.3.0
============

* Add python source files for generating data, building the model, and validating the model.
  * Implement status polling for async job submission to prevent timeout issues.
* Add VTT Q50 log-transform model with improved accuracy
* Update documentation to reflect log-transform approach for VTT Q50.
* Update VTT Q50 model coefficients.
  `#5 <https://github.com/FiQCI/resource-estimator/pull/5>`_

Version 0.2.2
============

* Add link for data.
  `#4 <https://github.com/FiQCI/resource-estimator/pull/4>`_



Version 0.2.1
============

* Add FAQ section.
  `#3 <https://github.com/FiQCI/resource-estimator/pull/3>`_


Version 0.2.0
============

* Add documentation.
  `#2 <https://github.com/FiQCI/resource-estimator/pull/2>`_


Version 0.1.0
============

* Initial version.
  `#1 <https://github.com/FiQCI/resource-estimator/pull/1>`_

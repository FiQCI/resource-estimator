"""Logging configuration for resource_estimator."""

import logging


def setup_logging() -> logging.Logger:
	"""Configure logging for the application.

	Sets root logger to WARNING to suppress verbose third-party logs,
	while keeping resource_estimator logs at INFO level.

	Returns:
		Logger for the calling module
	"""
	# Set root logger to WARNING to suppress verbose third-party logs (qiskit, iqm-client, etc.)
	logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

	# Set our module loggers to INFO
	logging.getLogger("resource_estimator").setLevel(logging.INFO)

	# Return logger for the calling module
	import inspect

	frame = inspect.currentframe()
	if frame and frame.f_back:
		caller_module = frame.f_back.f_globals.get("__name__", "resource_estimator")
		return logging.getLogger(caller_module)
	return logging.getLogger("resource_estimator")

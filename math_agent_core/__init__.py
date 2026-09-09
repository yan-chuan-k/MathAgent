__all__ = ["MathAgentOrchestrator"]


def __getattr__(name):
    # Keep the public import stable without loading the legacy orchestrator in
    # the dedicated deterministic subprocess worker.
    if name == "MathAgentOrchestrator":
        from .orchestrator import MathAgentOrchestrator

        return MathAgentOrchestrator
    raise AttributeError(name)

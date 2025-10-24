def _ensure_list_signal(signal, fallback_time=[0.0, 1.0]):
    """Ensure signal is a list; return default time vector if needed."""
    if not isinstance(signal, list):
        return [signal] * 2, fallback_time
    return signal, fallback_time if len(signal) == 1 else None
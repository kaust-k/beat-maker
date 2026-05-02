class ProfileManager:
    """
    Tracks the active profile index and a pending switch request.
    Pending switches are applied at cycle end to avoid mid-beat sound changes.
    Thread-safe for CPython: active_index and _pending_index are single int assignments.
    """

    def __init__(self, profiles: list[dict]):
        self._profiles = profiles
        self.active_index: int = 0
        self._pending_index: int | None = None

    @property
    def active(self) -> dict:
        return self._profiles[self.active_index]

    @property
    def count(self) -> int:
        return len(self._profiles)

    @property
    def pending_index(self) -> int | None:
        return self._pending_index

    def request_switch(self, idx: int) -> None:
        if 0 <= idx < len(self._profiles) and idx != self.active_index:
            self._pending_index = idx

    def apply_pending(self) -> bool:
        """Apply pending profile switch. Returns True if the active profile changed."""
        if self._pending_index is not None and self._pending_index != self.active_index:
            self.active_index = self._pending_index
            self._pending_index = None
            return True
        self._pending_index = None
        return False

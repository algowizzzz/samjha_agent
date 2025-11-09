import json
import os
import threading
from typing import Any, Dict, Optional


class QueryAgentConfig:
    """
    Centralized configuration loader for the Parquet Agent.
    Reads config/agent/queryagent_planner.json with simple mtime-based caching.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.config_path = config_path or os.path.join("config", "agent", "queryagent_planner.json")
        self._cache: Dict[str, Any] = {}
        self._mtime: Optional[float] = None
        self._config_lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        with self._config_lock:
            try:
                if not os.path.exists(self.config_path):
                    self._cache = {}
                    self._mtime = None
                    return
                mtime = os.path.getmtime(self.config_path)
                if self._mtime is not None and mtime == self._mtime:
                    return
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                self._mtime = mtime
            except Exception:
                # Keep previous cache on error
                pass

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve top-level value by key. Auto-reloads on change.
        """
        self._load()
        return self._cache.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve nested value using a sequence of keys.
        """
        self._load()
        node: Any = self._cache
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def all(self) -> Dict[str, Any]:
        """
        Return full config dictionary (read-only).
        """
        self._load()
        return dict(self._cache)
    
    def is_streaming_enabled(self) -> bool:
        """
        Check if streaming is enabled.
        """
        return self.get_nested("streaming", "enabled", default=False)
    
    def get_streaming_nodes(self) -> Dict[str, bool]:
        """
        Get streaming configuration for each node.
        Returns dict with node names as keys and enabled status as values.
        """
        nodes = self.get_nested("streaming", "nodes", default={})
        return {
            "planner": nodes.get("planner", True),
            "evaluator": nodes.get("evaluator", True),
            "end_response": nodes.get("end_response", True),
            "end_prompt_monitor": nodes.get("end_prompt_monitor", True)
        }



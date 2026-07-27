"""Four compact brownfield cases; hidden tests are never copied into agent workdirs."""

from __future__ import annotations

from dataclasses import dataclass

# The long literals below are exact fixture files, not executable source layout.
# ruff: noqa: E501


@dataclass(frozen=True)
class MiniCase:
    id: str
    prompt: str
    starter_files: dict[str, str]
    hidden_test: str


CASES = [
    MiniCase(
        id="nested_config",
        prompt=(
            "Improve the existing config_store package. ConfigStore.get must support "
            "dot-separated traversal through nested dictionaries while preserving direct "
            "key lookup (an exact key containing dots wins). Missing paths return the supplied "
            "default, and reads must not mutate the input. Inspect the existing files and make "
            "the smallest compatible change."
        ),
        starter_files={
            "config_store/__init__.py": "from .store import ConfigStore\n",
            "config_store/store.py": '''"""Small configuration lookup object."""\n\n\nclass ConfigStore:\n    def __init__(self, data: dict):\n        self._data = data\n\n    def get(self, key: str, default=None):\n        return self._data.get(key, default)\n''',
            "app.py": '''from config_store import ConfigStore\n\n\ndef database_host(data: dict) -> str:\n    return ConfigStore(data).get("database.host", "localhost")\n''',
        },
        hidden_test='''import copy\nimport unittest\n\nfrom config_store import ConfigStore\n\n\nclass ConfigStoreTests(unittest.TestCase):\n    def test_nested_lookup(self):\n        cfg = ConfigStore({"database": {"host": "db", "port": 5432}})\n        self.assertEqual(cfg.get("database.host"), "db")\n        self.assertEqual(cfg.get("database.port"), 5432)\n\n    def test_direct_dotted_key_wins(self):\n        cfg = ConfigStore({"database.host": "override", "database": {"host": "nested"}})\n        self.assertEqual(cfg.get("database.host"), "override")\n\n    def test_missing_and_non_mapping_segment_return_default(self):\n        cfg = ConfigStore({"database": {"host": "db"}, "flat": 3})\n        self.assertEqual(cfg.get("database.user", "guest"), "guest")\n        self.assertEqual(cfg.get("flat.value", "missing"), "missing")\n\n    def test_lookup_does_not_mutate_input(self):\n        data = {"database": {"host": "db"}}\n        before = copy.deepcopy(data)\n        ConfigStore(data).get("database.missing")\n        self.assertEqual(data, before)\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
    ),
    MiniCase(
        id="event_unsubscribe",
        prompt=(
            "Fix the existing events package. Registry.unsubscribe(topic, handler) must remove "
            "only the exact handler instance requested, preserve other handlers and their order, "
            "return True only when removal happened, and remove the topic entry when it becomes "
            "empty. Preserve EventBus behavior and public signatures. Inspect the repository "
            "before editing and verify the fix."
        ),
        starter_files={
            "events/__init__.py": "from .bus import EventBus\nfrom .registry import Registry\n",
            "events/registry.py": '''class Registry:\n    def __init__(self):\n        self._handlers = {}\n\n    def subscribe(self, topic, handler):\n        self._handlers.setdefault(topic, []).append(handler)\n\n    def unsubscribe(self, topic, handler):\n        if topic not in self._handlers:\n            return False\n        self._handlers.pop(topic, None)\n        return True\n\n    def handlers(self, topic):\n        return list(self._handlers.get(topic, ()))\n''',
            "events/bus.py": '''from .registry import Registry\n\n\nclass EventBus:\n    def __init__(self):\n        self.registry = Registry()\n\n    def publish(self, topic, payload):\n        return [handler(payload) for handler in self.registry.handlers(topic)]\n''',
        },
        hidden_test='''import unittest\n\nfrom events import EventBus, Registry\n\n\nclass RegistryTests(unittest.TestCase):\n    def test_removes_only_exact_instance_and_preserves_order(self):\n        registry = Registry()\n        first = lambda value: ("first", value)\n        second = lambda value: ("second", value)\n        third = lambda value: ("third", value)\n        for handler in (first, second, third):\n            registry.subscribe("order", handler)\n        self.assertTrue(registry.unsubscribe("order", second))\n        self.assertEqual(registry.handlers("order"), [first, third])\n\n    def test_equal_but_distinct_callable_is_not_removed(self):\n        class Handler:\n            def __call__(self, value):\n                return value\n            def __eq__(self, other):\n                return isinstance(other, Handler)\n        registered = Handler()\n        other = Handler()\n        registry = Registry()\n        registry.subscribe("topic", registered)\n        self.assertFalse(registry.unsubscribe("topic", other))\n        self.assertEqual(registry.handlers("topic"), [registered])\n\n    def test_empty_topic_is_cleaned_and_missing_returns_false(self):\n        registry = Registry()\n        handler = lambda value: value\n        registry.subscribe("topic", handler)\n        self.assertTrue(registry.unsubscribe("topic", handler))\n        self.assertNotIn("topic", registry._handlers)\n        self.assertFalse(registry.unsubscribe("topic", handler))\n\n    def test_bus_keeps_remaining_handler(self):\n        bus = EventBus()\n        first = lambda value: value + 1\n        second = lambda value: value * 2\n        bus.registry.subscribe("n", first)\n        bus.registry.subscribe("n", second)\n        bus.registry.unsubscribe("n", first)\n        self.assertEqual(bus.publish("n", 3), [6])\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
    ),
    MiniCase(
        id="retry_policy",
        prompt=(
            "Repair the existing client.retry RetryPolicy. Attempts are numbered from 1 and "
            "max_attempts counts the original request, so the final attempt is not retried. "
            "Retry only HTTP 429 and 5xx responses. backoff(attempt) should return base_delay "
            "for the first retry, double thereafter, and never exceed max_delay. Keep the "
            "public class and constructor compatible."
        ),
        starter_files={
            "client/__init__.py": "from .retry import RetryPolicy\n",
            "client/retry.py": '''class RetryPolicy:\n    def __init__(self, max_attempts=3, base_delay=0.1, max_delay=2.0):\n        self.max_attempts = max_attempts\n        self.base_delay = base_delay\n        self.max_delay = max_delay\n\n    def should_retry(self, status, attempt):\n        return status >= 400 and attempt <= self.max_attempts\n\n    def backoff(self, attempt):\n        return self.base_delay * (2 ** attempt)\n''',
            "client/session.py": '''from .retry import RetryPolicy\n\n\ndef retry_delays(statuses, policy=None):\n    policy = policy or RetryPolicy()\n    return [policy.backoff(i) for i, status in enumerate(statuses, 1)\n            if policy.should_retry(status, i)]\n''',
        },
        hidden_test='''import unittest\n\nfrom client import RetryPolicy\n\n\nclass RetryPolicyTests(unittest.TestCase):\n    def test_only_transient_statuses_retry(self):\n        policy = RetryPolicy(max_attempts=3)\n        for status in (429, 500, 503):\n            self.assertTrue(policy.should_retry(status, 1))\n        for status in (200, 301, 400, 401, 404):\n            self.assertFalse(policy.should_retry(status, 1))\n\n    def test_final_attempt_never_retries(self):\n        policy = RetryPolicy(max_attempts=3)\n        self.assertTrue(policy.should_retry(500, 1))\n        self.assertTrue(policy.should_retry(500, 2))\n        self.assertFalse(policy.should_retry(500, 3))\n        self.assertFalse(policy.should_retry(500, 4))\n\n    def test_backoff_numbering_and_cap(self):\n        policy = RetryPolicy(base_delay=0.25, max_delay=0.75)\n        self.assertEqual(policy.backoff(1), 0.25)\n        self.assertEqual(policy.backoff(2), 0.5)\n        self.assertEqual(policy.backoff(3), 0.75)\n        self.assertEqual(policy.backoff(8), 0.75)\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
    ),
    MiniCase(
        id="unique_slugs",
        prompt=(
            "Add unique_slugs(values) to the existing text_utils package and export it. Reuse "
            "the existing slugify behavior. Preserve input order; duplicate base slugs receive "
            "-2, -3, etc.; empty slugs use 'item'; and suffix allocation must avoid collisions "
            "with values that already naturally contain a suffix. Do not regress slugify."
        ),
        starter_files={
            "text_utils/__init__.py": "from .slug import slugify\n",
            "text_utils/slug.py": '''import re\n\n\ndef slugify(value: str) -> str:\n    value = value.strip().lower()\n    value = re.sub(r"[^a-z0-9]+", "-", value)\n    return value.strip("-")\n''',
        },
        hidden_test='''import unittest\n\nfrom text_utils import slugify, unique_slugs\n\n\nclass SlugTests(unittest.TestCase):\n    def test_slugify_is_preserved(self):\n        self.assertEqual(slugify(" Hello, World! "), "hello-world")\n\n    def test_duplicates_and_empty_values(self):\n        self.assertEqual(\n            unique_slugs(["Hello World", "hello-world", "!!!", ""]),\n            ["hello-world", "hello-world-2", "item", "item-2"],\n        )\n\n    def test_natural_suffix_does_not_collide(self):\n        self.assertEqual(\n            unique_slugs(["post", "post-2", "post", "post-2"]),\n            ["post", "post-2", "post-3", "post-2-2"],\n        )\n\n    def test_returns_a_new_list(self):\n        values = ["A", "A"]\n        result = unique_slugs(values)\n        self.assertEqual(values, ["A", "A"])\n        self.assertEqual(result, ["a", "a-2"])\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
    ),
]


def cases() -> list[MiniCase]:
    return list(CASES)

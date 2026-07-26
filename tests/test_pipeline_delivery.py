from __future__ import annotations

import asyncio

from conftest import make_proposal

from metaimprove.improve.deliverer import Delivery
from metaimprove.improve.pipeline import ImprovementPipeline
from metaimprove.tools.registry import ToolRegistry


class FakeClient:
    model_name = "fake"
    provider_name = "fake"


class FakeDeliverer:
    def __init__(self):
        self.received_diff = ""

    async def deliver(self, proposal, *, cwd, loop_diff):
        self.received_diff = loop_diff
        return Delivery(passed=True, reasons=["accepted"])


class FakeWorktree:
    def __init__(self, trees):
        self.trees = iter(trees)
        self.diff_calls = 0

    async def snapshot(self):
        return next(self.trees)

    async def diff_since(self, ref):
        self.diff_calls += 1
        if ref == "candidate":
            return "delivery mutation"
        return f"loop diff {self.diff_calls}"


def _pipeline(deliverer):
    return ImprovementPipeline(
        client=FakeClient(),
        cwd=".",
        registry=ToolRegistry(),
        deliverer=deliverer,
    )


def test_deliverer_receives_loop_base_diff_and_unchanged_tree_passes():
    deliverer = FakeDeliverer()
    worktree = FakeWorktree(["candidate", "candidate"])

    delivery, final_diff = asyncio.run(
        _pipeline(deliverer)._deliver_immutable(
            make_proposal(), worktree, "loop-base", "/candidate"
        )
    )

    assert deliverer.received_diff == "loop diff 1"
    assert final_diff == "loop diff 2"
    assert delivery.passed
    assert delivery.integrity_ok
    assert delivery.verified_tree == "candidate"


def test_delivery_mutation_is_rejected_and_reported():
    deliverer = FakeDeliverer()
    worktree = FakeWorktree(["candidate", "mutated"])

    delivery, _ = asyncio.run(
        _pipeline(deliverer)._deliver_immutable(
            make_proposal(), worktree, "loop-base", "/candidate"
        )
    )

    assert not delivery.passed
    assert not delivery.integrity_ok
    assert delivery.verified_tree == ""
    assert delivery.mutation_diff == "delivery mutation"
    assert "mutated the candidate" in delivery.reasons[0]

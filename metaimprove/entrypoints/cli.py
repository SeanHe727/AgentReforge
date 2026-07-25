"""
Interface between CLI and client
CLI -> Create Client -> Use Client
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from ..agent.query import query
from ..config import load_config
from ..improve.trajectory import load_recent_trajectory, log_trajectory
from ..llm.openai_compatible import OpenAICompatibleClient
from ..mcp.client import McpClientManager
from ..memory.manager import MemoryManager
from ..prompt.assembler import PromptAssembler
from ..rag.code_index import CodeIndex
from ..tools.builtins import get_builtin_tools
from ..tools.registry import ToolRegistry

# Long-term memory and the code index live in SQLite files under the user's home.
_MEMORY_DB = Path.home() / ".meta-improve" / "memory.db"
_CODE_INDEX_DB = Path.home() / ".meta-improve" / "code_index.db"

app = typer.Typer(
    name="meta-improve",
    help="meta-improve - Terminal AI Agent in Python",
    invoke_without_command=True,
    no_args_is_help=False,
)


# register the main entrypoints
@app.callback()
def main(
    ctx: typer.Context,
    prompt: Annotated[
        str | None, typer.Option("-p", "--prompt", help="use -p/--prompt to start")
    ] = None,
    model: Annotated[
        str | None, typer.Option("-m", "--model", help="use -m/--model to start")
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Override provider")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Auto-approve dangerous tools (skip HITL prompts)")
    ] = False,
) -> None:
    # if a subcommand (e.g. `improve`, `serve`) was invoked, let it handle everything.
    if ctx.invoked_subcommand is not None:
        return
    # the bare `-p` path is the plain ReAct worker (chat). Self-improvement work
    # goes through the single orchestrator entry: the `improve` subcommand.
    root = str((cwd or Path.cwd()).resolve())
    config = load_config(cli_provider=provider, cli_model=model)

    # generate response, calling run_prompt() in one-shot mode. No REPL yet.
    if prompt is not None:
        asyncio.run(run_prompt(prompt, config, root, auto_yes=yes))
    else:
        typer.echo('Interactive REPL is not implemented yet. Use -p "your prompt".')


def _make_approval_callback(auto_yes: bool):
    # HITL: ask the user to approve a dangerous tool before it runs.
    def approve(info: dict) -> bool:
        if auto_yes:
            return True
        typer.echo(f"\n Agent wants to run '{info['name']}' with: {info['args']}", err=True)
        answer = input("Approve? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    return approve


async def run_prompt(
    prompt: str,
    config,
    cwd: str,
    auto_yes: bool = False,
) -> None:
    # fail fast if there is no API key.
    if not config.api_key:
        typer.echo("Fatal error: API key is not configured.", err=True)
        raise typer.Exit(1)

    # build client from config.
    client = OpenAICompatibleClient(
        provider_name=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )

    # register built-in tools so the agent can act, not just chat.
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())

    # discover + register external MCP tools from .meta-improve/mcp.json (if any).
    mcp_manager = McpClientManager(cwd)
    registry.register_all(await mcp_manager.load_tools())
    for name, err in mcp_manager.last_errors.items():
        typer.echo(f"[mcp] server '{name}' failed to load: {err}", err=True)

    # long-term memory, scoped to this project (cwd).
    memory = MemoryManager(_MEMORY_DB, scope=cwd)

    # code index for this project; rebuild at startup so search_code is fresh.
    # (real tools rebuild incrementally / on demand; full rebuild is fine here.)
    code_index = CodeIndex(root=cwd, db_path=_CODE_INDEX_DB)
    code_index.rebuild()

    # build structured prompt; tool names + recalled memories go in for grounding.
    system_prompt = PromptAssembler(
        cwd=cwd,
        model=config.model,
        provider=config.provider,
        tool_names=registry.list_names(),
        memories=[m.content for m in memory.list_memory()],  # inject memory into system_prompt
    ).build()

    # the bare prompt runs the plain ReAct worker (chat). Plan/Team are no longer
    # top-level modes — they are executor configs the orchestrator uses internally.
    events = query(
        client=client,
        registry=registry,
        system_prompt=system_prompt,
        user_message=prompt,
        cwd=cwd,
        memory=memory,
        code_index=code_index,
        approval_callback=_make_approval_callback(auto_yes),
    )
    # record this run as a trajectory so a later `improve` can use it as evidence.
    events = log_trajectory(events, cwd=cwd)

    # consume the event stream: stream text, note tool/task use, report errors.
    async for event in events:
        etype = event["type"]
        if etype == "text_delta":
            print(event["text"], end="", flush=True)
        elif etype == "tool_result":
            status = "error" if event["is_error"] else "ok"
            typer.echo(f"\n[{event['name']} -> {status}]", err=True)
        elif etype == "error":
            typer.echo(f"\nFatal error: {event['error']}", err=True)
            raise typer.Exit(1)
    print()  # final newline after the streamed reply


def _make_improve_approver(auto_yes: bool):
    # HITL at the SPEC stage: show the proposal + gate decision, approve the
    # intent BEFORE any code is written or the proposal is frozen.
    from ..improve.pipeline import Approval

    async def approve(proposal, decision):
        if auto_yes:
            return Approval(approved=True, approved_by="cli --yes")
        typer.echo("\n=== Improvement proposal ===", err=True)
        typer.echo(f"Summary : {proposal.summary}", err=True)
        typer.echo(f"Problem : {proposal.problem_statement}", err=True)
        typer.echo(
            f"Scores  : benefit={proposal.benefit} risk={proposal.risk} "
            f"effort={proposal.effort} confidence={proposal.confidence}",
            err=True,
        )
        typer.echo(f"Gate    : {decision.decision} — {'; '.join(decision.reasons)}", err=True)
        typer.echo("Tasks   :", err=True)
        for t in proposal.tasks:
            typer.echo(f"  - [{t.id}] {t.description}", err=True)
        answer = input("Approve this improvement intent? [y/N] ").strip().lower()
        return Approval(approved=answer in ("y", "yes"), approved_by="cli")

    return approve


async def run_improve(
    intent: str,
    config,
    cwd: str,
    level: str = "standard",
    governance: str = "supervised",
    cycles: int = 1,
    auto_merge: bool = False,
    keep_worktree: bool = True,
) -> None:
    from ..improve.pipeline import ImprovementPipeline
    from ..improve.run_config import RecursionPolicy

    if not config.api_key:
        typer.echo("Fatal error: API key is not configured.", err=True)
        raise typer.Exit(1)

    client = OpenAICompatibleClient(
        provider_name=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )

    # detail level = effort + test ceiling; governance = HITL (autonomous asks no
    # human; supervised uses the interactive approver at the proposal checkpoint).
    pipeline = ImprovementPipeline(
        client=client,
        cwd=cwd,
        level=level,
        governance=governance,
        recursion=RecursionPolicy(max_loops=cycles),
        approver=_make_improve_approver(auto_yes=False),
        auto_merge=auto_merge,
        keep_worktree=keep_worktree,
    )

    # load recent agent runs as evidence for the Analyzer (empty on a fresh repo).
    trajectory = load_recent_trajectory(cwd)
    typer.echo(
        f"[improve] level={level} mode={governance} cycles={cycles} "
        f"trajectory={len(trajectory)} records analyzing: {intent}",
        err=True,
    )
    result = await pipeline.run(intent=intent, trajectory=trajectory)

    typer.echo(f"\n[improve] stage: {result.stage}", err=True)
    if result.proposal and result.stage in ("abstained", "rejected"):
        p = result.proposal
        typer.echo(f"[improve] proposal: {p.summary}", err=True)
        typer.echo(
            f"[improve] scores: benefit={p.benefit} risk={p.risk} "
            f"effort={p.effort} confidence={p.confidence}",
            err=True,
        )
    if result.stage == "failed" and result.error:
        typer.echo(f"[improve] failed: {result.error[:300]}", err=True)
    if result.gate and result.stage == "abstained":
        typer.echo(f"[improve] gate abstained: {'; '.join(result.gate.reasons)}", err=True)
    if result.gate and result.stage == "needs_human":
        typer.echo(
            f"[improve] needs human (autonomous stopped): {'; '.join(result.gate.reasons)}",
            err=True,
        )
    if result.frozen:
        typer.echo(f"[improve] frozen proposal: {result.frozen.proposal_id}", err=True)
    if result.blocker:
        typer.echo(
            f"[improve] BLOCKED on task {result.blocker.task_id} -> "
            f"{result.blocker.recommendation}",
            err=True,
        )
    if result.delivery:
        verdict = "passed" if result.delivery.passed else "rejected"
        reasons = "; ".join(result.delivery.reasons)
        typer.echo(f"[improve] delivery: {verdict} — {reasons}", err=True)
    if result.merged_commit:
        typer.echo(f"[improve] merged: {result.merged_commit[:12]}", err=True)
    elif result.stage == "delivered":
        typer.echo(
            "[improve] delivered but not merged — review the worktree/report, then merge.",
            err=True,
        )
    if result.manifest:
        m = result.manifest
        typer.echo(
            f"[improve] run {m.get('run_id')} on {m.get('candidate_branch')} "
            f"(base {m.get('base_commit')})",
            err=True,
        )
    if result.report_path:
        typer.echo(f"[improve] loop report: {result.report_path}", err=True)
    if result.run_report_path:
        typer.echo(f"[improve] run report: {result.run_report_path}", err=True)


@app.command("improve")
def improve(
    intent: Annotated[str, typer.Option("-i", "--intent", help="What to improve about the agent")],
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Target repository")] = None,
    model: Annotated[str | None, typer.Option("-m", "--model")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    level: Annotated[
        str, typer.Option("--level", help="Effort + test rigor: quick | standard | deep")
    ] = "standard",
    mode: Annotated[
        str, typer.Option("--mode", help="Governance (HITL): autonomous | supervised")
    ] = "supervised",
    cycles: Annotated[
        int, typer.Option("--loops", "--cycles", help="Max improvement loops in the run")
    ] = 1,
    merge: Annotated[
        bool, typer.Option("--merge", help="Auto-merge when all required generated tests pass")
    ] = False,
    keep: Annotated[
        bool, typer.Option("--keep/--no-keep", help="Keep the worktree for inspection")
    ] = True,
) -> None:
    """Run the self-improvement pipeline: propose -> gate -> approve -> isolate ->
    test -> implement -> evaluate -> report."""
    root = str((cwd or Path.cwd()).resolve())
    config = load_config(cli_provider=provider, cli_model=model)
    asyncio.run(
        run_improve(
            intent, config, root,
            level=level, governance=mode, cycles=cycles, auto_merge=merge, keep_worktree=keep,
        )
    )


@app.command("serve")
def serve(
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 8080,
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "127.0.0.1",
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """Start the Runtime API HTTP server."""
    import uvicorn

    from ..runtime.api import create_app
    from ..runtime.state import RuntimeState

    root = str((cwd or Path.cwd()).resolve())  # cwd
    config = load_config()
    if not config.api_key:
        typer.echo("Fatal error: model API key is not configured.", err=True)
        raise typer.Exit(1)
    # the Runtime API's own key (clients must present this), separate from the model key.
    api_key = os.getenv("METAIMPROVE_RUNTIME_API_KEY", "dev-key")

    state = RuntimeState(config, root, api_key)
    fastapi_app = create_app(state)  # create application

    typer.echo(f"meta-improve Runtime API on http://{host}:{port} (x-api-key: {api_key})")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()

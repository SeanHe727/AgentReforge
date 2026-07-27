"""Run the mini agent on commands from argv."""

from __future__ import annotations

import logging
import sys

from .agent import MiniAgent


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(
        level=logging.INFO,
        format="[mini-agent] %(message)s",
        stream=sys.stderr,
    )
    agent = MiniAgent()
    for command in argv:
        result = agent.handle(command)
        logging.info("handled %r -> %r", command, result)
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys

from agent import run_agent


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = input("Ask: ").strip()

    final_text, trace, _ = run_agent(
        [
            {
                "role": "user",
                "content": query,
            }
        ]
    )

    print("\nFinal answer:\n")

    print(final_text)

    print("\nExecution trace:\n")

    for event in trace.events:
        print(
            f"- {event.tool}: "
            f"{event.status}, "
            f"{event.latency_ms:.1f} ms, "
            f"args={event.args}"
        )


if __name__ == "__main__":
    main()

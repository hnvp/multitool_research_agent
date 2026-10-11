from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from agent import run_agent

DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.json")


def percentile(values: list[float], p: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return float("nan")

    xs = sorted(values)
    k = (len(xs) - 1) * p
    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return xs[int(k)]

    return xs[lower] * (upper - k) + xs[upper] * (k - lower)


def check_answer(
    answer: str,
    assertions: dict,
) -> tuple[bool, list[str]]:
    """Evaluate deterministic answer assertions."""
    answer_lower = answer.lower()
    failures: list[str] = []

    for item in assertions.get("contains", []):
        if item.lower() not in answer_lower:
            failures.append(f"missing expected text: {item!r}")

    for item in assertions.get("not_contains", []):
        if item.lower() in answer_lower:
            failures.append(f"contains forbidden text: {item!r}")

    for alternatives in assertions.get("contains_any", []):
        if not any(alternative.lower() in answer_lower for alternative in alternatives):
            failures.append("missing any expected text: " f"{alternatives!r}")

    return not failures, failures


def check_tool_selection(
    expected_tools: list[str],
    actual_tools: list[str],
) -> bool:
    """Check whether the agent selected exactly the expected tool set."""
    return set(expected_tools) == set(actual_tools)


def check_tool_sequence(
    expected_sequence: list[str] | None,
    actual_tools: list[str],
) -> bool | None:
    """Check tool order only when a sequence is explicitly defined."""
    if expected_sequence is None:
        return None

    return expected_sequence == actual_tools


def main() -> None:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        cases = [c for c in cases if c["id"] in wanted]

    if not cases:
        raise SystemExit("No matching test cases found.")

    rows: list[dict] = []
    end_to_end_latencies: list[float] = []
    tool_latencies: dict[str, list[float]] = {}
    category_rows: dict[str, list[dict]] = {}

    for case in cases:
        case_id = case["id"]
        category = case["category"]
        query = case["query"]

        print()
        print(f"[{case_id}] {query}")

        started = perf_counter()

        try:
            answer, trace, _ = run_agent(
                [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            )

            end_to_end_latency = (perf_counter() - started) * 1000

            actual_tools = [event.tool for event in trace.events]

            expected_tools = case.get(
                "expected_tools",
                [],
            )

            selection_ok = check_tool_selection(
                expected_tools,
                actual_tools,
            )

            expected_sequence = case.get(
                "expected_tool_sequence",
            )

            sequence_ok = check_tool_sequence(
                expected_sequence,
                actual_tools,
            )

            execution_ok = all(event.status == "success" for event in trace.events)

            assertions = case.get(
                "answer_assertions",
                {},
            )

            if assertions:
                answer_ok, answer_failures = check_answer(
                    answer,
                    assertions,
                )
            else:
                answer_ok = None
                answer_failures = []

            tool_latency = sum(event.latency_ms for event in trace.events)

            end_to_end_latencies.append(end_to_end_latency)

            for event in trace.events:
                tool_latencies.setdefault(
                    event.tool,
                    [],
                ).append(event.latency_ms)

            row = {
                "id": case_id,
                "category": category,
                "selection_ok": selection_ok,
                "sequence_ok": sequence_ok,
                "execution_ok": execution_ok,
                "answer_ok": answer_ok,
                "tools_used": actual_tools,
                "end_to_end_latency_ms": round(
                    end_to_end_latency,
                    2,
                ),
                "tool_latency_ms": round(
                    tool_latency,
                    2,
                ),
            }

            if answer_failures:
                row["answer_failures"] = answer_failures

            rows.append(row)

            category_rows.setdefault(
                category,
                [],
            ).append(row)

            print(f"Tools: {actual_tools}")
            print(
                f"Selection OK: {selection_ok}, "
                f"Sequence OK: {sequence_ok}, "
                f"Execution OK: {execution_ok}, "
                f"Answer OK: {answer_ok}"
            )

            if answer_failures:
                print("Answer failures: " + "; ".join(answer_failures))

            print(f"End-to-end latency: " f"{end_to_end_latency:.2f} ms")

            print(f"Tool latency: " f"{tool_latency:.2f} ms")

            print(f"Answer: {answer[:500]}")

        except Exception as exc:
            end_to_end_latency = (perf_counter() - started) * 1000

            row = {
                "id": case_id,
                "category": category,
                "selection_ok": False,
                "sequence_ok": None,
                "execution_ok": False,
                "answer_ok": False,
                "tools_used": [],
                "end_to_end_latency_ms": round(
                    end_to_end_latency,
                    2,
                ),
                "tool_latency_ms": None,
                "error": str(exc),
            }

            rows.append(row)

            category_rows.setdefault(
                category,
                [],
            ).append(row)

            end_to_end_latencies.append(end_to_end_latency)

            print(f"ERROR: {exc}")

    valid = [row for row in rows if row["end_to_end_latency_ms"] is not None]

    selection_accuracy = mean(row["selection_ok"] for row in valid) if valid else 0.0

    execution_success = mean(row["execution_ok"] for row in valid) if valid else 0.0

    sequence_rows = [row for row in valid if row["sequence_ok"] is not None]

    sequence_accuracy = (
        mean(row["sequence_ok"] for row in sequence_rows) if sequence_rows else None
    )

    answer_rows = [row for row in valid if row["answer_ok"] is not None]

    answer_accuracy = (
        mean(row["answer_ok"] for row in answer_rows) if answer_rows else None
    )

    print()
    print("=" * 60)
    print("MULTITOOL AGENT EVALUATION")
    print("=" * 60)

    print(f"Cases: {len(rows)}")

    print(f"Tool selection accuracy: " f"{selection_accuracy:.2%}")

    print(f"Tool execution success: " f"{execution_success:.2%}")

    if sequence_accuracy is not None:
        print(f"Tool sequence accuracy: " f"{sequence_accuracy:.2%}")
        print(f"Sequence cases checked: " f"{len(sequence_rows)}/{len(valid)}")

    if answer_accuracy is not None:
        print(f"Answer correctness: " f"{answer_accuracy:.2%}")
        print(f"Answer cases checked: " f"{len(answer_rows)}/{len(valid)}")
        print(
            "Answer cases passed: "
            f"{sum(row['answer_ok'] is True for row in answer_rows)}/"
            f"{len(answer_rows)}"
        )
    else:
        print("Answer correctness: N/A")

    if end_to_end_latencies:
        print()
        print("End-to-End Latency:")
        print(f"Mean: " f"{mean(end_to_end_latencies):.2f} ms")
        print(f"P50: " f"{percentile(end_to_end_latencies, 0.50):.2f} ms")
        print(f"P95: " f"{percentile(end_to_end_latencies, 0.95):.2f} ms")

    if tool_latencies:
        print()
        print("Tool Latency:")

        for tool_name, values in sorted(tool_latencies.items()):
            print(
                f"{tool_name}: "
                f"mean={mean(values):.2f} ms, "
                f"p50={percentile(values, 0.50):.2f} ms, "
                f"p95={percentile(values, 0.95):.2f} ms"
            )

    print()
    print("Category Results:")

    for category in sorted(category_rows):
        category_data = category_rows[category]

        category_selection = mean(row["selection_ok"] for row in category_data)

        category_execution = mean(row["execution_ok"] for row in category_data)

        category_answers = [
            row for row in category_data if row["answer_ok"] is not None
        ]

        category_answer = (
            mean(row["answer_ok"] for row in category_answers)
            if category_answers
            else None
        )

        print(f"\n  {category}")
        print(f"    Cases: " f"{len(category_data)}")
        print(f"    Selection: " f"{category_selection:.2%}")
        print(f"    Execution: " f"{category_execution:.2%}")

        if category_answer is not None:
            print(f"    Answer: " f"{category_answer:.2%}")

    RESULTS_PATH.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Detailed results saved to " f"{RESULTS_PATH}")


if __name__ == "__main__":
    main()

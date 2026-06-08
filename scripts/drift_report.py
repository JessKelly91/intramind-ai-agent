"""Evidently AI drift report for IntraMind ingestion + retrieval.

Step 5 of the Free RAI Stack. Generates an HTML drift report comparing a
"reference" window (older ingested content) against a "current" window
(newer ingested content) using Evidently. Designed to run unattended on a
weekly schedule from GitHub Actions.

Strategy:
  * Pull a sample of recent search results from the gateway (the API exposes
    content + metadata + score - not raw vectors - so we drift on retrieval
    quality and content shape, not on raw embedding floats).
  * Bucket by ``ingested_at`` metadata into reference / current windows.
  * Compute Evidently DataDrift on:
      - similarity score distribution from a fixed probe set
      - chunk character length
      - PII findings count (Step 3 metadata round-trip)
  * Render an HTML report and update the docs/drift index.

Usage:
    python ai-agent/scripts/drift_report.py \\
        --collection eval_corpus \\
        --output-dir docs/drift
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import html
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "drift"
DEFAULT_PROBE_PATH = (
    Path(__file__).resolve().parent / "drift_probes.json"
)


def _default_probes() -> list[str]:
    """A small, fixed probe set used to measure retrieval-score drift.

    Kept short (and intentionally generic) because we only need a stable
    signal, not an exhaustive evaluation. A drop in average score over time
    is a leading indicator that the corpus has drifted away from the topics
    these probes represent.
    """
    return [
        "quarterly revenue and net income",
        "acceptable use policy violations",
        "onboarding for new engineers",
        "dividend payment date and amount",
        "credentials and security training",
    ]


async def _gather_rows(collection: str, probes: list[str]) -> list[dict[str, Any]]:
    """Run probe searches and flatten results into rows for Evidently."""
    sys.path.insert(0, str(REPO_ROOT / "ai-agent" / "src"))
    from tools.api_client import APIGatewayClient  # type: ignore[import]

    rows: list[dict[str, Any]] = []
    async with APIGatewayClient() as client:
        for probe in probes:
            try:
                resp = await client.search(
                    collection_name=collection, query=probe, limit=10, min_score=0.0
                )
            except Exception as exc:  # noqa: BLE001 - probe failures shouldn't kill the run
                logger.warning("Probe %r failed: %s", probe, exc)
                continue
            for r in resp.results or []:
                meta = r.metadata or {}
                ingested_at = meta.get("ingested_at") or meta.get("ingestion_timestamp")
                rows.append(
                    {
                        "probe": probe,
                        "score": float(r.score) if r.score is not None else 0.0,
                        "char_count": len(r.content or ""),
                        "pii_findings_count": int(meta.get("pii_findings_count", 0) or 0),
                        "ingested_at": ingested_at,
                        "source": meta.get("source") or meta.get("filename") or "unknown",
                    }
                )
    return rows


def _split_windows(
    rows: list[dict[str, Any]], split_at: dt.datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into reference (older) vs current (newer) windows."""
    reference: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        ts_raw = row.get("ingested_at")
        try:
            ts = dt.datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            # Rows with no usable timestamp go into the reference bucket so
            # they at least serve as a baseline.
            reference.append(row)
            continue
        if ts < split_at:
            reference.append(row)
        else:
            current.append(row)
    return reference, current


def _render_report(
    reference: list[dict[str, Any]],
    current: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Build an Evidently HTML drift report and write it to ``out_path``."""
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError as exc:
        raise RuntimeError(
            "Evidently / pandas not installed. `pip install evidently pandas`"
        ) from exc

    if not reference or not current:
        logger.warning(
            "Insufficient data for drift report (reference=%d, current=%d). "
            "Falling back to a single-window quality report.",
            len(reference),
            len(current),
        )
        # Use whichever window has rows as both, so Evidently produces *something*.
        reference = reference or current
        current = current or reference

    cols = ["score", "char_count", "pii_findings_count"]
    ref_df = pd.DataFrame(reference)[cols]
    cur_df = pd.DataFrame(current)[cols]

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    logger.info("Wrote drift report to %s", out_path)


def _render_no_data_report(collection: str, out_path: Path) -> None:
    """Write a lightweight report when probe searches return no rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    collection_html = html.escape(collection)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IntraMind Drift Report - No Data</title>
</head>
<body>
  <h1>IntraMind Drift Report</h1>
  <p><strong>Status:</strong> No retrieval rows were available for this run.</p>
  <p><strong>Collection:</strong> {collection_html}</p>
  <p><strong>Generated at:</strong> {generated_at}</p>
  <p>
    The weekly drift job completed, but all probe searches failed or returned
    no results. In the CI compose profile, semantic vectorizers are disabled, so
    retrieval-dependent drift metrics may be unavailable.
  </p>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    logger.warning("Wrote no-data drift report to %s", out_path)


def _update_index(output_dir: Path, latest: Path) -> None:
    """Maintain a simple docs/drift/index.md listing the most recent reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = sorted(
        [p for p in output_dir.glob("*.html")], key=lambda p: p.name, reverse=True
    )
    lines = [
        "# Drift Reports",
        "",
        "Auto-generated weekly by `.github/workflows/drift-report.yml`. ",
        "Each report compares a reference window (older ingested content) ",
        "against a current window (newer ingested content) using Evidently AI.",
        "",
        f"_Latest:_ [{latest.name}](./{latest.name})",
        "",
        "## All reports",
        "",
    ]
    for r in reports[:50]:  # cap so the index doesn't grow forever
        lines.append(f"- [{r.name}](./{r.name})")
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _async_main(args: argparse.Namespace) -> int:
    probes = _default_probes()
    if args.probe_file and Path(args.probe_file).exists():
        probes = json.loads(Path(args.probe_file).read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir)
    today = dt.date.today().isoformat()
    out_path = output_dir / f"{today}.html"

    rows = await _gather_rows(args.collection, probes)
    if not rows:
        logger.warning(
            "No rows returned from collection %s; drift report will record a no-data run.",
            args.collection,
        )
        print(
            "::warning title=Drift report has no retrieval rows::All probe searches "
            "failed or returned no results. CI disables semantic vectorizers, so "
            "retrieval-dependent drift metrics may be unavailable."
        )
        _render_no_data_report(args.collection, out_path)
        _update_index(output_dir, out_path)
        return 0

    split_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.window_days)
    reference, current = _split_windows(rows, split_at)

    _render_report(reference, current, out_path)
    _update_index(output_dir, out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a weekly drift report")
    parser.add_argument(
        "--collection",
        default=os.environ.get("DRIFT_COLLECTION", "eval_corpus"),
        help="Collection to probe (default: eval_corpus)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write HTML reports (default: docs/drift/)",
    )
    parser.add_argument(
        "--probe-file",
        default=str(DEFAULT_PROBE_PATH),
        help="Optional JSON file with a list of probe queries",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Boundary between reference and current windows (default: 7)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())

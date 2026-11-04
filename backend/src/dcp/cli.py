"""Run the pipeline from the command line, writing JSONL.

    dcp run data/raw -o output/sentences.jsonl
    dcp serve --port 8000
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dcp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Extract, segment and classify files")
    r.add_argument("inputs", nargs="+", help="Files or directories")
    r.add_argument("-o", "--output", default="-", help="JSONL output path (default: stdout)")
    r.add_argument("--classifiers", default="", help="Comma-separated classifier names (default: all)")
    r.add_argument("--usable-only", action="store_true", help="Drop sentences flagged unusable or duplicate")

    s = sub.add_parser("serve", help="Start the API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("dcp.api:app", host=args.host, port=args.port, reload=args.reload)
        return

    from dcp import pipeline

    clf = [c for c in args.classifiers.split(",") if c] or None
    sentences = pipeline.run(args.inputs, clf)
    if args.usable_only:
        sentences = [s for s in sentences if s.quality["usable"] and not s.quality["duplicate"]]
    lines = (json.dumps(asdict(s), ensure_ascii=False) + "\n" for s in sentences)
    if args.output == "-":
        sys.stdout.writelines(lines)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.writelines(lines)
    logging.info("Wrote %d sentences", len(sentences))


if __name__ == "__main__":
    main()

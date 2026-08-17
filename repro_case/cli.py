"""Command-line entry point with machine-readable summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io import fetch_inputs, print_json, repository_root
from .protocol import infer_general_assessments, load_specification, run_attempt
from .release import package_release


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="src-case", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download and verify content-addressed inputs")
    fetch.add_argument("--cache", type=Path, default=Path("build/inputs"))

    run = commands.add_parser("run", help="bind, register, execute, and verify one attempt")
    run.add_argument("--mode", choices=("container", "native"), required=True)
    run.add_argument("--context", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--cache", type=Path, default=Path("build/inputs"))
    run.add_argument("--compile-paper", action="store_true")

    infer = commands.add_parser("infer", help="apply the registered inference rule")
    infer.add_argument("--attempt-root", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)

    package = commands.add_parser(
        "package-release", help="build four assets from supported general assessments"
    )
    package.add_argument("--attempt-root", type=Path, required=True)
    package.add_argument("--general-assessment", type=Path, required=True)
    package.add_argument("--cache", type=Path, default=Path("build/inputs"))
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--commit", required=True)
    package.add_argument("--tag", required=True)
    package.add_argument("--repository", required=True)
    package.add_argument("--image", required=True)
    package.add_argument("--image-digest", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    root = repository_root()
    try:
        if arguments.command == "fetch":
            specification, _ = load_specification(root)
            records = fetch_inputs(specification, arguments.cache.resolve())
            print_json({"command": "fetch", "files": records})
            return 0
        if arguments.command == "run":
            summary, exit_code = run_attempt(
                mode=arguments.mode,
                context=arguments.context,
                output=arguments.output,
                cache=arguments.cache,
                compile_paper=arguments.compile_paper,
                root=root,
            )
            print_json(summary)
            return exit_code
        if arguments.command == "infer":
            record, exit_code = infer_general_assessments(
                arguments.attempt_root, arguments.output, root
            )
            print_json(
                {
                    "command": "infer",
                    "general_assessments": {
                        mode: assessment["decision"]
                        for mode, assessment in record["general_assessments"].items()
                    },
                    "path": str(arguments.output),
                }
            )
            return exit_code
        manifest = package_release(
            attempt_root=arguments.attempt_root,
            general_assessment=arguments.general_assessment,
            cache=arguments.cache,
            output=arguments.output,
            commit=arguments.commit,
            tag=arguments.tag,
            repository=arguments.repository,
            image=arguments.image,
            image_digest=arguments.image_digest,
            root=root,
        )
        print_json(
            {
                "command": "package-release",
                "assets": ["main.pdf", "mnist-12.onnx", "reproduction.crate.zip", "SHA256SUMS"],
                "tag": manifest["release"]["tag"],
                "path": str(arguments.output),
            }
        )
        return 0
    except Exception as error:
        print_json(
            {"command": arguments.command, "error": type(error).__name__, "message": str(error)}
        )
        print(f"src-case: {error}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())

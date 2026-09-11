"""Build a shared Tree-sitter library from local grammar checkouts."""

import argparse
from pathlib import Path
from tree_sitter import Language


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--languages", nargs="+", default=["java", "go", "python"])
    args = parser.parse_args()
    sources = [
        args.grammar_dir / f"tree-sitter-{language}" for language in args.languages
    ]
    for source in sources:
        if not (source / "src/parser.c").is_file():
            raise FileNotFoundError(f"Missing grammar source: {source}/src/parser.c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Language.build_library(str(args.output), [str(source) for source in sources])


if __name__ == "__main__":
    main()

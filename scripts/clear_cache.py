#!/usr/bin/env python3
"""Utility to inspect or clear cached JSON feature files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and clear cached JSON files in a cache directory.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example usage:\n"
            "  python scripts/clear_cache.py --dry-run\n"
            "  python scripts/clear_cache.py --yes\n"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
        help="Cache directory containing JSON cache files (default: ./cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without deleting them",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and delete files immediately",
    )
    return parser.parse_args()


def iter_cache_files(cache_dir: Path) -> Iterable[Tuple[Path, int]]:
    if not cache_dir.exists():
        return []

    files: list[Tuple[Path, int]] = []
    for path in cache_dir.rglob("*.json"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        files.append((path, size))
    return files


def format_size(bytes_count: int) -> str:
    mb = bytes_count / (1024 * 1024)
    return f"{mb:.2f} MB"


def print_summary(cache_dir: Path, count: int, total_bytes: int) -> None:
    print(f"Cache directory: {cache_dir}")
    print(f"Found {count} cache file{'s' if count != 1 else ''} totalling {format_size(total_bytes)}.")


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = Path.cwd() / cache_dir
    cache_dir = cache_dir.resolve()

    files = list(iter_cache_files(cache_dir))
    total_bytes = sum(size for _, size in files)
    count = len(files)

    print_summary(cache_dir, count, total_bytes)

    if args.dry_run:
        if files:
            for path, size in files:
                relative = path
                if path.is_relative_to(cache_dir):
                    relative = path.relative_to(cache_dir)
                print(f"[DRY-RUN] {relative} - {format_size(size)}")
        else:
            print("No cache files to delete.")
        print("Dry run complete; no files deleted.")
        return

    if count == 0:
        print("No cache files to delete.")
        return

    if not args.yes:
        prompt = f"Delete {count} files ({format_size(total_bytes)})? [y/N]: "
        reply = input(prompt).strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted; no files deleted.")
            return

    deleted = 0
    deleted_bytes = 0
    for path, size in files:
        try:
            path.unlink(missing_ok=True)
            deleted += 1
            deleted_bytes += size
        except OSError as exc:
            print(f"Failed to delete {path}: {exc}")

    print(f"Deleted {deleted} files ({format_size(deleted_bytes)})")


if __name__ == "__main__":
    main()
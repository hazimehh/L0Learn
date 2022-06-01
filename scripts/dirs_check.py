#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import filecmp
import pathlib
import sys
from typing import Sequence, Dict, List


class onlydiffdircmp(filecmp.dircmp):
    def report_dict(self) -> Dict[str, List[str]]:
        print("report_dict", self.left, self.right)

        results = {}
        results["left"] = self.left
        results["right"] = self.right
        results["left_only"] = self.left_only.sort()
        results["right_only"] = self.right_only
        results["diff_files"] = self.diff_files
        results["funny_files"] = self.funny_files
        return results


def dirs_check(dirs: Sequence[pathlib.Path], ignore: Sequence[pathlib.Path]):
    first_dir, *other_dirs = dirs

    bad_diffs = []

    for other_dir in other_dirs:
        d = onlydiffdircmp(first_dir, other_dir, ignore=ignore)
        diff_results = d.report_dict()

        if any(
            [
                diff_results["left_only"],
                diff_results["right_only"],
                diff_results["diff_files"],
                diff_results["funny_files"],
            ]
        ):
            d.report()
            bad_diffs.append(other_dir)

    if bad_diffs:
        print(f"dirs {bad_diffs} are mis-matched from '{first_dir}'")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check directories to ensure equivalence"
    )
    parser.add_argument(
        "dirs",
        metavar="paths/to/dirs",
        type=str,
        nargs="*",
        help="paths to check for equivalence between dirs",
    )
    parser.add_argument(
        "--ignore",
        metavar="paths/to/ignore",
        nargs="*",
        help="paths exclude from equivalence between dirs",
    )

    args = parser.parse_args()

    dirs_check(dirs=args.dirs, ignore=args.ignore)
    sys.exit()

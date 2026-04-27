from __future__ import annotations

import argparse

from mapper_tda.bias_audit import display_path
from mapper_tda.region_synthesis import synthesize_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize selected Mapper nodes/components into physical, observational, mixed, or weak regions.",
    )
    parser.add_argument("--outputs-dir", default="outputs/mapper", help="Mapper outputs directory. Default: outputs/mapper.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesis, csv_path, md_path = synthesize_regions(args.outputs_dir)
    print("Mapper region synthesis complete.")
    print(f"regions: {len(synthesis)}")
    print(f"csv: {display_path(csv_path)}")
    print(f"markdown: {display_path(md_path)}")


if __name__ == "__main__":
    main()

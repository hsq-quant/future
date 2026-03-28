from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nbd_search import NbdSearchConfig, combine_nbd_discoveries, discover_nbd_search_urls
from src.utils.io import read_yaml


def main() -> None:
    config = read_yaml(ROOT / "configs/nbd_keyword_sets.yaml").get("nbd_search", {})
    start_date = config["start_date"]
    end_date = config["end_date"]
    page_size = int(config.get("page_size", 50))
    frames = []

    for batch in config.get("batches", []):
        frame = discover_nbd_search_urls(
            NbdSearchConfig(
                keywords=list(batch.get("keywords", [])),
                start_date=start_date,
                end_date=end_date,
                page_size=page_size,
                max_results_per_keyword=int(batch.get("max_results_per_keyword", 300)),
            )
        )
        frames.append(frame)
        print(f"batch={batch.get('name', 'unnamed')} rows={len(frame)}")

    combined = combine_nbd_discoveries(frames)
    output = ROOT / config.get("output", "data/raw/finnewshunter_exports/nbd_search_export_5k.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"rows={len(combined)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()

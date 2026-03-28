from __future__ import annotations

from src.data.hf_dataset_fetch import (
    build_hf_headers,
    choose_parquet_urls,
    missing_download_targets,
    parse_hf_parquet_urls,
    parse_hf_rows_payload,
    parse_hf_splits_payload,
    select_parquet_urls_by_ids,
)


def test_build_hf_headers_uses_bearer_token() -> None:
    headers = build_hf_headers("token-123")

    assert headers["Authorization"] == "Bearer token-123"


def test_parse_hf_splits_payload_extracts_split_names() -> None:
    payload = {
        "splits": [
            {"dataset": "x", "config": "data", "split": "train"},
            {"dataset": "x", "config": "data", "split": "validation"},
        ]
    }

    assert parse_hf_splits_payload(payload) == ["train", "validation"]


def test_parse_hf_rows_payload_extracts_row_dicts() -> None:
    payload = {
        "rows": [
            {"row": {"date": "2020-01-01T00:00:00Z", "text": "Oil moved higher."}},
            {"row": {"date": "2020-01-02T00:00:00Z", "text": "OPEC met."}},
        ]
    }

    rows = parse_hf_rows_payload(payload)

    assert rows == [
        {"date": "2020-01-01T00:00:00Z", "text": "Oil moved higher."},
        {"date": "2020-01-02T00:00:00Z", "text": "OPEC met."},
    ]


def test_parse_hf_parquet_urls_accepts_multiple_response_shapes() -> None:
    payload = [
        {"url": "https://huggingface.co/file-1.parquet"},
        {"url": "https://huggingface.co/file-2.parquet"},
    ]

    assert parse_hf_parquet_urls(payload) == [
        "https://huggingface.co/file-1.parquet",
        "https://huggingface.co/file-2.parquet",
    ]

    nested = {"parquet_files": [{"url": "https://huggingface.co/file-3.parquet"}]}
    assert parse_hf_parquet_urls(nested) == ["https://huggingface.co/file-3.parquet"]

    string_list = [
        "https://huggingface.co/file-4.parquet",
        "https://huggingface.co/file-5.parquet",
    ]
    assert parse_hf_parquet_urls(string_list) == string_list


def test_choose_parquet_urls_supports_head_tail_and_all() -> None:
    urls = [f"https://huggingface.co/{idx}.parquet" for idx in range(6)]

    assert choose_parquet_urls(urls, max_files=2, tail=False, all_files=False) == urls[:2]
    assert choose_parquet_urls(urls, max_files=2, tail=True, all_files=False) == urls[-2:]
    assert choose_parquet_urls(urls, max_files=2, tail=False, all_files=True) == urls


def test_missing_download_targets_skips_existing_files(tmp_path) -> None:
    urls = [
        "https://huggingface.co/api/datasets/x/parquet/train/0.parquet",
        "https://huggingface.co/api/datasets/x/parquet/train/1.parquet",
    ]
    (tmp_path / "0.parquet").write_text("done", encoding="utf-8")

    targets = missing_download_targets(urls, tmp_path)

    assert len(targets) == 1
    assert targets[0][0].endswith("1.parquet")
    assert targets[0][1].name == "1.parquet"


def test_select_parquet_urls_by_ids_returns_only_requested_filenames() -> None:
    urls = [f"https://huggingface.co/{idx}.parquet" for idx in range(6)]

    selected = select_parquet_urls_by_ids(urls, [1, 4, 9])

    assert selected == [
        "https://huggingface.co/1.parquet",
        "https://huggingface.co/4.parquet",
    ]

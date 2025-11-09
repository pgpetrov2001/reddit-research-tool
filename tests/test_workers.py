from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from arctic_shift.workers import WorkerPlan, discover_worker_files, prepare_worker_files, worker_path_for


def create_test_data(created_utc: str, **kwargs) -> dict:
    """Helper to create test data rows."""
    data = {"created_utc": created_utc}
    data.update(kwargs)
    return data


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Helper to write JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Helper to read JSONL file."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_prepare_worker_files_no_existing_files():
    """Test that function creates empty files when no existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
            WorkerPlan(interval=("2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"), expected=20),
        ]
        
        prepare_worker_files("posts", base_dir, plans, [])
        
        assert worker_dir.exists()
        files = list(worker_dir.glob("*.jsonl"))
        assert len(files) == 2
        for f in files:
            assert f.exists()
            assert f.stat().st_size == 0  # Empty files


def test_prepare_worker_files_redistribute_simple():
    """Test basic redistribution when partitions change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        # Create old worker files with data
        old_file1 = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file1, [
            create_test_data("2020-01-01T12:00:00Z", id="1"),
            create_test_data("2020-01-01T18:00:00Z", id="2"),
        ])
        
        old_file2 = worker_dir / "posts_worker02__old2__old3__200.jsonl"
        write_jsonl(old_file2, [
            create_test_data("2020-01-02T12:00:00Z", id="3"),
            create_test_data("2020-01-02T18:00:00Z", id="4"),
        ])
        
        existing = [
            (old_file1, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100),
            (old_file2, "2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z", 200),
        ]
        
        # New partitions that split differently
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-01T15:00:00Z"), expected=50),
            WorkerPlan(interval=("2020-01-01T15:00:00Z", "2020-01-03T00:00:00Z"), expected=150),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        # Check old files are deleted
        assert not old_file1.exists()
        assert not old_file2.exists()
        
        # Check new files exist and contain correct data
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 2
        
        # First partition: 2020-01-01T00:00:00Z to 2020-01-01T15:00:00Z
        # Should contain id="1" (2020-01-01T12:00:00Z)
        file1_data = read_jsonl(new_files[0])
        assert len(file1_data) == 1
        assert file1_data[0]["id"] == "1"
        
        # Second partition: 2020-01-01T15:00:00Z to 2020-01-03T00:00:00Z
        # Should contain id="2", "3", "4"
        file2_data = read_jsonl(new_files[1])
        assert len(file2_data) == 3
        ids = {row["id"] for row in file2_data}
        assert ids == {"2", "3", "4"}


def test_prepare_worker_files_boundary_exact_start():
    """Test handling of timestamps exactly at partition start."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2020-01-02T00:00:00Z", id="1"),  # Exactly at new partition start
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=50),
            WorkerPlan(interval=("2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"), expected=50),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 2
        
        # Timestamp exactly at boundary should go to second partition (since range is [start, end))
        file1_data = read_jsonl(new_files[0])
        file2_data = read_jsonl(new_files[1])
        
        assert len(file1_data) == 0  # First partition is empty
        assert len(file2_data) == 1  # Second partition gets the row
        assert file2_data[0]["id"] == "1"


def test_prepare_worker_files_boundary_exact_end():
    """Test handling of timestamps exactly at partition end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2020-01-02T00:00:00Z", id="1"),  # Exactly at first partition end
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=50),
            WorkerPlan(interval=("2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"), expected=50),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        
        # Timestamp at end boundary should NOT be in that partition (range is [start, end))
        file1_data = read_jsonl(new_files[0])
        file2_data = read_jsonl(new_files[1])
        
        assert len(file1_data) == 0
        assert len(file2_data) == 1
        assert file2_data[0]["id"] == "1"


def test_prepare_worker_files_unbounded_start():
    """Test redistribution with unbounded start partition."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2019-12-31T12:00:00Z", id="1"),
            create_test_data("2020-01-01T12:00:00Z", id="2"),
            create_test_data("2020-01-02T12:00:00Z", id="3"),
        ])
        
        existing = [(old_file, None, "2020-01-03T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=(None, "2020-01-01T00:00:00Z"), expected=50),
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z"), expected=50),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 2
        
        file1_data = read_jsonl(new_files[0])
        file2_data = read_jsonl(new_files[1])
        
        # First partition (unbounded start) should get id="1"
        assert len(file1_data) == 1
        assert file1_data[0]["id"] == "1"
        
        # Second partition should get id="2", "3"
        assert len(file2_data) == 2
        ids = {row["id"] for row in file2_data}
        assert ids == {"2", "3"}


def test_prepare_worker_files_unbounded_end():
    """Test redistribution with unbounded end partition."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2020-01-01T12:00:00Z", id="1"),
            create_test_data("2020-01-02T12:00:00Z", id="2"),
            create_test_data("2020-01-03T12:00:00Z", id="3"),
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", None, 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=50),
            WorkerPlan(interval=("2020-01-02T00:00:00Z", None), expected=50),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 2
        
        file1_data = read_jsonl(new_files[0])
        file2_data = read_jsonl(new_files[1])
        
        # First partition should get id="1"
        assert len(file1_data) == 1
        assert file1_data[0]["id"] == "1"
        
        # Second partition (unbounded end) should get id="2", "3"
        assert len(file2_data) == 2
        ids = {row["id"] for row in file2_data}
        assert ids == {"2", "3"}


def test_prepare_worker_files_malformed_json():
    """Test that malformed JSON lines are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        worker_dir.mkdir(parents=True, exist_ok=True)
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        with old_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(create_test_data("2020-01-01T12:00:00Z", id="1")) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(create_test_data("2020-01-01T18:00:00Z", id="2")) + "\n")
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        # Should only have 2 valid rows, malformed JSON skipped
        assert len(file_data) == 2
        ids = {row["id"] for row in file_data}
        assert ids == {"1", "2"}


def test_prepare_worker_files_missing_created_utc():
    """Test that rows without created_utc are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2020-01-01T12:00:00Z", id="1"),
            {"id": "2", "no_created": True},  # Missing created_utc
            create_test_data("2020-01-01T18:00:00Z", id="3"),
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        # Should only have 2 rows, missing created_utc skipped
        assert len(file_data) == 2
        ids = {row["id"] for row in file_data}
        assert ids == {"1", "3"}


def test_prepare_worker_files_empty_lines():
    """Test that empty lines are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        worker_dir.mkdir(parents=True, exist_ok=True)
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        with old_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(create_test_data("2020-01-01T12:00:00Z", id="1")) + "\n")
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace only
            f.write(json.dumps(create_test_data("2020-01-01T18:00:00Z", id="2")) + "\n")
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        assert len(file_data) == 2
        ids = {row["id"] for row in file_data}
        assert ids == {"1", "2"}


def test_prepare_worker_files_multiple_old_files_to_one_new():
    """Test redistributing from multiple old files into fewer new partitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file1 = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file1, [
            create_test_data("2020-01-01T12:00:00Z", id="1"),
        ])
        
        old_file2 = worker_dir / "posts_worker02__old2__old3__100.jsonl"
        write_jsonl(old_file2, [
            create_test_data("2020-01-02T12:00:00Z", id="2"),
        ])
        
        old_file3 = worker_dir / "posts_worker03__old3__old4__100.jsonl"
        write_jsonl(old_file3, [
            create_test_data("2020-01-03T12:00:00Z", id="3"),
        ])
        
        existing = [
            (old_file1, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100),
            (old_file2, "2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z", 100),
            (old_file3, "2020-01-03T00:00:00Z", "2020-01-04T00:00:00Z", 100),
        ]
        
        # Merge into single partition
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-04T00:00:00Z"), expected=300),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        # All old files should be deleted
        assert not old_file1.exists()
        assert not old_file2.exists()
        assert not old_file3.exists()
        
        # One new file with all data
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        assert len(file_data) == 3
        ids = {row["id"] for row in file_data}
        assert ids == {"1", "2", "3"}


def test_prepare_worker_files_one_old_file_to_multiple_new():
    """Test redistributing from one old file into multiple new partitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old4__300.jsonl"
        write_jsonl(old_file, [
            create_test_data("2020-01-01T12:00:00Z", id="1"),
            create_test_data("2020-01-02T12:00:00Z", id="2"),
            create_test_data("2020-01-03T12:00:00Z", id="3"),
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-04T00:00:00Z", 300)]
        
        # Split into 3 partitions
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=100),
            WorkerPlan(interval=("2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"), expected=100),
            WorkerPlan(interval=("2020-01-03T00:00:00Z", "2020-01-04T00:00:00Z"), expected=100),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        # Old file should be deleted
        assert not old_file.exists()
        
        # Three new files
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 3
        
        file1_data = read_jsonl(new_files[0])
        file2_data = read_jsonl(new_files[1])
        file3_data = read_jsonl(new_files[2])
        
        assert len(file1_data) == 1
        assert file1_data[0]["id"] == "1"
        
        assert len(file2_data) == 1
        assert file2_data[0]["id"] == "2"
        
        assert len(file3_data) == 1
        assert file3_data[0]["id"] == "3"


def test_prepare_worker_files_timestamp_outside_bounds():
    """Test handling of timestamps outside partition boundaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2019-12-31T12:00:00Z", id="1"),  # Before first partition
            create_test_data("2020-01-01T12:00:00Z", id="2"),  # In first partition
            create_test_data("2020-01-02T12:00:00Z", id="3"),  # In second partition
            create_test_data("2020-01-03T12:00:00Z", id="4"),  # After last partition
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=50),
            WorkerPlan(interval=("2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"), expected=50),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 2
        
        file1_data = read_jsonl(new_files[0])
        file2_data = read_jsonl(new_files[1])
        
        # Only rows within partitions should be included (id="1" and "4" should be excluded)
        all_ids = {row["id"] for row in file1_data + file2_data}
        assert all_ids == {"2", "3"}


def test_prepare_worker_files_timestamp_outside_bounds_with_unbounded():
    """Test that timestamps outside bounds are assigned to unbounded partitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2019-12-31T12:00:00Z", id="1"),  # Before first partition (unbounded start)
            create_test_data("2020-01-01T12:00:00Z", id="2"),  # In first partition
            create_test_data("2020-01-02T12:00:00Z", id="3"),  # In second partition (unbounded end)
            create_test_data("2020-01-03T12:00:00Z", id="4"),  # After last partition (unbounded end)
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=(None, "2020-01-01T00:00:00Z"), expected=50),  # Unbounded start
            WorkerPlan(interval=("2020-01-01T00:00:00Z", None), expected=50),  # Unbounded end
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 2
        
        file1_data = read_jsonl(new_files[0])
        file2_data = read_jsonl(new_files[1])
        
        # id="1" should go to first partition (unbounded start)
        # id="2" should go to second partition
        # id="3" and "4" should go to second partition (unbounded end)
        assert len(file1_data) == 1
        assert file1_data[0]["id"] == "1"
        
        assert len(file2_data) == 3
        ids = {row["id"] for row in file2_data}
        assert ids == {"2", "3", "4"}


def test_prepare_worker_files_empty_old_files():
    """Test handling of empty old files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        old_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.touch()  # Empty file
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        # Old file should be deleted
        assert not old_file.exists()
        
        # New file should exist and be empty
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        assert new_files[0].exists()
        assert len(read_jsonl(new_files[0])) == 0


def test_prepare_worker_files_preserves_all_valid_data():
    """Test that all valid data from old files is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        old_file = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        # Create data with various timestamp formats
        rows = [
            create_test_data("2020-01-01T00:00:00Z", id="1"),
            create_test_data("2020-01-01T06:00:00Z", id="2"),
            create_test_data("2020-01-01T12:00:00Z", id="3"),
            create_test_data("2020-01-01T18:00:00Z", id="4"),
        ]
        write_jsonl(old_file, rows)
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=100),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        assert len(file_data) == 4
        ids = {row["id"] for row in file_data}
        assert ids == {"1", "2", "3", "4"}


def test_prepare_worker_files_different_kind():
    """Test that function works for different kinds (posts vs comments)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "comments_workers"
        
        old_file = worker_dir / "comments_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file, [
            create_test_data("2020-01-01T12:00:00Z", id="1"),
        ])
        
        existing = [(old_file, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100)]
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
        ]
        
        prepare_worker_files("comments", base_dir, plans, existing)
        
        new_files = list(worker_dir.glob("comments_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        assert len(file_data) == 1
        assert file_data[0]["id"] == "1"


def test_prepare_worker_files_from_final_output_file():
    """Test redistribution from final output file when no worker files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        # Create final output file instead of worker files
        final_file = base_dir / "testsubreddit.posts.jsonl"
        write_jsonl(final_file, [
            create_test_data("2020-01-01T06:00:00Z", id="1"),
            create_test_data("2020-01-01T12:00:00Z", id="2"),
            create_test_data("2020-01-01T18:00:00Z", id="3"),
            create_test_data("2020-01-02T12:00:00Z", id="4"),
        ])
        
        # No existing worker files
        existing = []
        
        # New partitions that split the data
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-01T15:00:00Z"), expected=50),
            WorkerPlan(interval=("2020-01-01T15:00:00Z", "2020-01-03T00:00:00Z"), expected=50),
        ]
        
        prepare_worker_files(
            "posts",
            base_dir,
            plans,
            existing,
            subreddit="testsubreddit",
        )
        
        # Final file should still exist (not deleted)
        assert final_file.exists()
        
        # New worker files should be created
        new_files = sorted(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 2
        
        # First partition: 2020-01-01T00:00:00Z to 2020-01-01T15:00:00Z
        # Should contain id="1" (06:00) and id="2" (12:00)
        file1_data = read_jsonl(new_files[0])
        assert len(file1_data) == 2
        ids1 = {row["id"] for row in file1_data}
        assert ids1 == {"1", "2"}
        
        # Verify sorted
        timestamps1 = [row["created_utc"] for row in file1_data]
        assert timestamps1 == sorted(timestamps1)
        
        # Second partition: 2020-01-01T15:00:00Z to 2020-01-03T00:00:00Z
        # Should contain id="3" (18:00) and id="4" (2020-01-02T12:00:00Z)
        file2_data = read_jsonl(new_files[1])
        assert len(file2_data) == 2
        ids2 = {row["id"] for row in file2_data}
        assert ids2 == {"3", "4"}
        
        # Verify sorted
        timestamps2 = [row["created_utc"] for row in file2_data]
        assert timestamps2 == sorted(timestamps2)


def test_prepare_worker_files_from_final_output_file_no_match():
    """Test that final output file is not used if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        # No files exist at all
        existing = []
        
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
        ]
        
        prepare_worker_files(
            "posts",
            base_dir,
            plans,
            existing,
            subreddit="testsubreddit",
        )
        
        # Should create empty worker files
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        assert len(read_jsonl(new_files[0])) == 0


def test_prepare_worker_files_from_final_output_file_outside_range():
    """Test that only data within partition boundaries is redistributed from final file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        # Final file contains data both inside and outside the requested range
        final_file = base_dir / "testsubreddit.posts.jsonl"
        write_jsonl(final_file, [
            create_test_data("2019-12-31T12:00:00Z", id="0"),  # Before range
            create_test_data("2020-01-01T06:00:00Z", id="1"),  # In range
            create_test_data("2020-01-01T12:00:00Z", id="2"),  # In range
            create_test_data("2020-01-02T12:00:00Z", id="3"),  # After range
        ])
        
        existing = []
        
        # Requested range: 2020-01-01T00:00:00Z to 2020-01-02T00:00:00Z
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=10),
        ]
        
        prepare_worker_files(
            "posts",
            base_dir,
            plans,
            existing,
            subreddit="testsubreddit",
        )
        
        # Only rows within range should be redistributed
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        assert len(file_data) == 2
        ids = {row["id"] for row in file_data}
        assert ids == {"1", "2"}  # id="0" and "3" should be excluded


def test_prepare_worker_files_preserves_sorting():
    """Test that redistributed data is sorted within each partition."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        worker_dir = base_dir / "posts_workers"
        
        # Create old files with unsorted data (different files, different orders)
        old_file1 = worker_dir / "posts_worker01__old1__old2__100.jsonl"
        write_jsonl(old_file1, [
            create_test_data("2020-01-01T18:00:00Z", id="3"),  # Latest in this file
            create_test_data("2020-01-01T06:00:00Z", id="1"),  # Earliest in this file
        ])
        
        old_file2 = worker_dir / "posts_worker02__old2__old3__100.jsonl"
        write_jsonl(old_file2, [
            create_test_data("2020-01-01T12:00:00Z", id="2"),  # Middle timestamp
        ])
        
        existing = [
            (old_file1, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100),
            (old_file2, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", 100),
        ]
        
        # Merge into single partition - all data should end up in one file, sorted
        plans = [
            WorkerPlan(interval=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), expected=200),
        ]
        
        prepare_worker_files("posts", base_dir, plans, existing)
        
        new_files = list(worker_dir.glob("posts_worker*.jsonl"))
        assert len(new_files) == 1
        
        file_data = read_jsonl(new_files[0])
        assert len(file_data) == 3
        
        # Data should be sorted by created_utc
        timestamps = [row["created_utc"] for row in file_data]
        assert timestamps == sorted(timestamps), "Data should be sorted chronologically"
        
        # Verify IDs in correct order
        assert file_data[0]["id"] == "1"
        assert file_data[1]["id"] == "2"
        assert file_data[2]["id"] == "3"


from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .workers import WorkerPlan


class ProgressTracker:
    def __init__(self, label: str, plans: List[WorkerPlan], total: Optional[int] = None, initial_counts: Optional[Dict[int, int]] = None) -> None:
        self.label = label
        self.plans = plans
        self.total = total
        self.counts: Dict[int, int] = initial_counts.copy() if initial_counts else {i: 0 for i in range(len(plans))}
        self.initial_counts: Dict[int, int] = initial_counts.copy() if initial_counts else {i: 0 for i in range(len(plans))}
        self.start_time: float = time.time()
        self.worker_start_times: Dict[int, float] = {}  # Track when each worker first reports progress
        self.worker_completed: Dict[int, bool] = {i: False for i in range(len(plans))}  # Track completion status
        self.lock = threading.Lock()
        self._printed = False
        self._num_lines = 0  # Track how many lines we've printed
        self.use_colors = self._supports_color()

    def _supports_color(self) -> bool:
        """Check if terminal supports color output."""
        # Check if running in a terminal that supports colors
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("TERM") == "dumb":
            return False
        # Windows 10+ supports ANSI escape codes
        if sys.platform == "win32":
            return True
        # Unix-like systems
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _colored(self, text: str, color: str) -> str:
        """Add color to text if terminal supports it."""
        if not self.use_colors:
            return text
        
        colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'gray': '\033[90m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def update(self, worker_id: int, delta: int) -> None:
        """Update progress for a specific worker."""
        with self.lock:
            worker_idx = worker_id - 1
            if worker_idx not in self.worker_start_times:
                # First update from this worker - record start time
                self.worker_start_times[worker_idx] = time.time()
            if worker_idx in self.counts:
                self.counts[worker_idx] += delta
            self._print_progress()

    def mark_completed(self, worker_id: int) -> None:
        """Mark a worker as completed."""
        with self.lock:
            worker_idx = worker_id - 1
            self.worker_completed[worker_idx] = True
            self._print_progress()

    def print_initial(self) -> None:
        """Print initial progress state."""
        self._print_progress()

    def _make_progress_bar(self, current: int, total: Optional[int], width: int = 20, is_completed: bool = False) -> str:
        """Create a visual progress bar like [=====>    ]"""
        if not total or total <= 0:
            # Animated spinner for unknown progress
            if is_completed:
                return self._colored("[✓ complete]".ljust(width + 2), "green")
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            idx = int(time.time() * 2) % len(spinner_chars)
            return f"[{spinner_chars[idx]} working...]"
        
        filled = int((current / total) * width)
        filled = min(filled, width)
        
        if is_completed:
            # Show as complete regardless of percentage
            bar = "=" * width
            return self._colored(f"[{bar}]", "green")
        elif filled >= width:
            # Over 100% but not marked complete yet
            bar = "=" * width
            return self._colored(f"[{bar}]", "yellow")
        elif filled > 0:
            bar = "=" * (filled - 1) + ">" + " " * (width - filled)
            return self._colored(f"[{bar}]", "cyan")
        else:
            bar = " " * width
            return f"[{bar}]"

    def _format_time(self, seconds: float) -> str:
        """Format elapsed time as HH:MM:SS."""
        if seconds < 0:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 99:
            return f"{hours}h"
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_eta(self, remaining: int, speed: float) -> str:
        """Format estimated time to completion."""
        if speed <= 0 or remaining <= 0:
            return "--:--:--"
        
        seconds = remaining / speed
        if seconds > 359999:  # > 99 hours
            days = seconds / 86400
            return f"{days:.1f}d"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_speed(self, items_per_sec: float) -> str:
        """Format speed as items/second."""
        if items_per_sec >= 1000:
            return f"{items_per_sec/1000:.1f}k/s"
        elif items_per_sec >= 10:
            return f"{items_per_sec:.1f}/s"
        elif items_per_sec >= 1:
            return f"{items_per_sec:.2f}/s"
        else:
            return f"{items_per_sec:.2f}/s"

    def _get_worker_speed(self, worker_idx: int, fetched: int, current_time: float) -> float:
        """Get worker speed in items/second."""
        worker_start = self.worker_start_times.get(worker_idx, self.start_time)
        elapsed = current_time - worker_start
        return fetched / elapsed if elapsed > 0 else 0

    def _clear_previous_output(self) -> None:
        """Clear the previously printed lines."""
        if self._num_lines > 0:
            for _ in range(self._num_lines):
                sys.stdout.write("\033[F")  # Move cursor up one line
                sys.stdout.write("\033[K")  # Clear the line

    def _print_progress(self) -> None:
        """Print multi-line progress with one line per worker."""
        if self._printed:
            self._clear_previous_output()
        else:
            self._printed = True
        
        current_time = time.time()
        total_elapsed = current_time - self.start_time
        
        lines = []
        
        # Header line
        elapsed_str = self._colored(self._format_time(total_elapsed), "gray")
        lines.append(f"{self._colored('●', 'green')} [{self.label}] Progress (elapsed: {elapsed_str})")
        
        # Per-worker lines
        for i, plan in enumerate(self.plans):
            count = self.counts.get(i, 0)
            initial_count = self.initial_counts.get(i, 0)
            fetched_since_start = count - initial_count
            is_completed = self.worker_completed.get(i, False)
            
            expected = plan.expected
            
            # Build progress bar
            bar = self._make_progress_bar(count, expected, width=20, is_completed=is_completed)
            
            # Calculate speed
            speed = self._get_worker_speed(i, fetched_since_start, current_time)
            speed_str = self._format_speed(speed)
            
            if expected is not None and expected > 0:
                pct = (count / expected) * 100
                
                # Completion marker or ETA
                if is_completed:
                    status = self._colored("✓", "green")
                else:
                    eta = self._format_eta(expected - count, speed)
                    status = f"ETA: {eta}"
                
                # Format with resume info if applicable
                if initial_count > 0:
                    resume_info = self._colored(f"(+{fetched_since_start})", "yellow")
                    lines.append(f"  W{i+1}: {bar} {count:>6}/{expected:<6} {pct:>5.1f}% {resume_info} │ {speed_str:>8} │ {status}")
                else:
                    lines.append(f"  W{i+1}: {bar} {count:>6}/{expected:<6} {pct:>5.1f}% │ {speed_str:>8} │ {status}")
            else:
                # Unknown total
                if is_completed:
                    status = self._colored("✓", "green")
                else:
                    status = "working..."
                
                if initial_count > 0:
                    resume_info = self._colored(f"(+{fetched_since_start})", "yellow")
                    lines.append(f"  W{i+1}: {bar} {count:>6} {resume_info} │ {speed_str:>8} │ {status}")
                else:
                    lines.append(f"  W{i+1}: {bar} {count:>6} │ {speed_str:>8} │ {status}")
        
        # Total line
        total_fetched = sum(self.counts.values())
        total_initial = sum(self.initial_counts.values())
        total_fetched_since_start = total_fetched - total_initial
        total_expected = sum(p.expected for p in self.plans if p.expected is not None)
        
        if total_expected and total_expected > 0:
            total_pct = (total_fetched / total_expected) * 100
            bar = self._make_progress_bar(total_fetched, total_expected, width=20)
            total_speed = total_fetched_since_start / total_elapsed if total_elapsed > 0 else 0
            speed_str = self._format_speed(total_speed)
            eta = self._format_eta(total_expected - total_fetched, total_speed)
            
            # Separator line
            lines.append(f"  {'─' * 70}")
            
            if total_initial > 0:
                resume_info = self._colored(f"(+{total_fetched_since_start})", "yellow")
                lines.append(f"  {self._colored('Total', 'cyan')}: {bar} {total_fetched:>6}/{total_expected:<6} {total_pct:>5.1f}% {resume_info} │ {speed_str:>8} │ ETA: {eta}")
            else:
                lines.append(f"  {self._colored('Total', 'cyan')}: {bar} {total_fetched:>6}/{total_expected:<6} {total_pct:>5.1f}% │ {speed_str:>8} │ ETA: {eta}")
        else:
            # Unknown total
            lines.append(f"  {'─' * 70}")
            total_speed = total_fetched_since_start / total_elapsed if total_elapsed > 0 else 0
            speed_str = self._format_speed(total_speed)
            
            if total_initial > 0:
                resume_info = self._colored(f"(+{total_fetched_since_start})", "yellow")
                lines.append(f"  {self._colored('Total', 'cyan')}: {total_fetched:>6} {resume_info} │ {speed_str:>8}")
            else:
                lines.append(f"  {self._colored('Total', 'cyan')}: {total_fetched:>6} │ {speed_str:>8}")
        
        # Print all lines
        output = "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        
        self._num_lines = len(lines)

    def close(self) -> None:
        """Finalize progress display."""
        if self._printed:
            # Don't clear - leave final progress visible
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._printed = False
        self._num_lines = 0


def get_resume_counts(kind: str, out_dir: Path, plans: List["WorkerPlan"]) -> Dict[int, int]:
    """Get resume counts for each worker by checking existing files."""
    from .workers import worker_path_for
    
    counts: Dict[int, int] = {}
    for idx, plan in enumerate(plans):
        worker_id = idx + 1
        path = worker_path_for(kind, out_dir, worker_id, plan.interval, plan.expected)
        if path.exists():
            count = 0
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            count += 1
            except OSError:
                count = 0
            counts[idx] = count
        else:
            counts[idx] = 0
    return counts


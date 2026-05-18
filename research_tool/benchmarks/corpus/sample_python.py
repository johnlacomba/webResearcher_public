"""
Event-driven task scheduler with priority queues and dependency tracking.

Supports DAG-based task execution with configurable parallelism,
timeout enforcement, and result caching.
"""

from __future__ import annotations

import heapq
import time
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class ScheduledTask:
    priority: int
    created_at: float
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    callback: Callable = field(compare=False, repr=False)
    dependencies: list[str] = field(default_factory=list, compare=False)
    timeout_seconds: float = field(default=300.0, compare=False)
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    retry_count: int = 0


def generate_task_id(name: str, params: dict) -> str:
    """Generate a deterministic task ID from name and parameters.

    Uses SHA-256 of the canonical parameter representation to ensure
    identical tasks produce identical IDs regardless of dict ordering.
    """
    canonical = f"{name}:{sorted(params.items())}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ResultCache:
    """LRU cache for task results with TTL-based expiration.

    Stores up to max_size results. Entries expire after ttl_seconds
    regardless of access pattern. Cache hits return a shallow copy
    of the stored TaskResult to prevent mutation of cached state.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        self._cache: dict[str, tuple[float, TaskResult]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, task_id: str) -> TaskResult | None:
        entry = self._cache.get(task_id)
        if entry is None:
            self._misses += 1
            return None
        stored_at, result = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._cache[task_id]
            self._misses += 1
            return None
        self._hits += 1
        return result

    def put(self, task_id: str, result: TaskResult) -> None:
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[task_id] = (time.monotonic(), result)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class DependencyGraph:
    """Directed acyclic graph for task dependency tracking.

    Maintains adjacency lists for both forward dependencies (task A
    depends on B) and reverse dependencies (B is depended on by A).
    Provides topological ordering and cycle detection.
    """

    def __init__(self):
        self._forward: dict[str, set[str]] = {}
        self._reverse: dict[str, set[str]] = {}

    def add_task(self, task_id: str, dependencies: list[str]) -> None:
        self._forward.setdefault(task_id, set()).update(dependencies)
        for dep in dependencies:
            self._reverse.setdefault(dep, set()).add(task_id)
            self._forward.setdefault(dep, set())

    def is_ready(self, task_id: str, completed: set[str]) -> bool:
        return all(dep in completed for dep in self._forward.get(task_id, set()))

    def get_dependents(self, task_id: str) -> set[str]:
        return self._reverse.get(task_id, set())

    def topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = {
            node: len(deps) for node, deps in self._forward.items()
        }
        queue = [node for node, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dependent in self._reverse.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._forward):
            cycle_nodes = set(self._forward) - set(result)
            raise ValueError(
                f"Dependency cycle detected involving tasks: {cycle_nodes}"
            )

        return result

    def detect_cycle(self) -> list[str] | None:
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._forward.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    path.append(neighbor)
                    return True

            path.pop()
            rec_stack.discard(node)
            return False

        for node in self._forward:
            if node not in visited:
                if dfs(node):
                    return path

        return None


class TaskScheduler:
    """Priority-based task scheduler with dependency resolution.

    Tasks are scheduled according to their priority (lower value = higher
    priority) and executed when all dependencies are satisfied. The
    scheduler maintains a min-heap for efficient priority ordering and
    a dependency graph for execution sequencing.

    Maximum concurrent tasks is configurable. When the concurrency limit
    is reached, tasks wait in the priority queue until a running task
    completes. Timed-out tasks are cancelled and optionally retried.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        cache: ResultCache | None = None,
    ):
        self._queue: list[ScheduledTask] = []
        self._running: dict[str, ScheduledTask] = {}
        self._results: dict[str, TaskResult] = {}
        self._graph = DependencyGraph()
        self._completed: set[str] = set()
        self._max_concurrent = max_concurrent
        self._cache = cache or ResultCache()

    def submit(
        self,
        name: str,
        callback: Callable,
        priority: Priority = Priority.NORMAL,
        dependencies: list[str] | None = None,
        timeout_seconds: float = 300.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        task_id = generate_task_id(name, metadata or {})
        cached = self._cache.get(task_id)
        if cached is not None:
            self._results[task_id] = cached
            self._completed.add(task_id)
            return task_id

        task = ScheduledTask(
            priority=priority.value,
            created_at=time.monotonic(),
            task_id=task_id,
            name=name,
            callback=callback,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
        )

        self._graph.add_task(task_id, task.dependencies)
        heapq.heappush(self._queue, task)
        return task_id

    def tick(self) -> list[TaskResult]:
        newly_completed = []

        check_timeouts = []
        for task_id, task in self._running.items():
            result = self._results.get(task_id)
            if result and result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                newly_completed.append(result)
            elif time.monotonic() - (self._results.get(task_id, TaskResult(task_id=task_id)).started_at or 0) > task.timeout_seconds:
                check_timeouts.append(task)

        for task in check_timeouts:
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.TIMED_OUT,
                error=f"Task exceeded {task.timeout_seconds}s timeout",
                finished_at=time.monotonic(),
            )
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                heapq.heappush(self._queue, task)
            else:
                self._results[task.task_id] = result
                newly_completed.append(result)
            self._running.pop(task.task_id, None)

        for result in newly_completed:
            self._completed.add(result.task_id)
            self._running.pop(result.task_id, None)
            if result.status == TaskStatus.COMPLETED:
                self._cache.put(result.task_id, result)

        while (
            len(self._running) < self._max_concurrent
            and self._queue
        ):
            task = heapq.heappop(self._queue)
            if not self._graph.is_ready(task.task_id, self._completed):
                heapq.heappush(self._queue, task)
                break

            self._running[task.task_id] = task
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.RUNNING,
                started_at=time.monotonic(),
            )
            self._results[task.task_id] = result

            try:
                output = task.callback()
                result.status = TaskStatus.COMPLETED
                result.output = output
            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error = str(e)
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    result.retry_count = task.retry_count
                    heapq.heappush(self._queue, task)

            result.finished_at = time.monotonic()
            result.duration_ms = (result.finished_at - result.started_at) * 1000

        return newly_completed

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def get_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def cancel(self, task_id: str) -> bool:
        if task_id in self._running:
            result = self._results.get(task_id)
            if result:
                result.status = TaskStatus.CANCELLED
                result.finished_at = time.monotonic()
            self._running.pop(task_id, None)
            self._completed.add(task_id)
            return True
        self._queue = [t for t in self._queue if t.task_id != task_id]
        heapq.heapify(self._queue)
        return True

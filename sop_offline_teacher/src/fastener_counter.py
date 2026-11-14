"""Causal event association and de-duplicated fastener counting.

The counter deliberately sits after visual inference.  It consumes per-tick
signals and turns them into auditable events instead of treating every tool
press as a new fastener.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Iterable


class CounterState(str, Enum):
    WAITING = "waiting"
    FASTENER_PLACED = "fastener_placed"
    FASTENING = "fastening"
    COMPLETE = "complete"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CounterConfig:
    place_threshold: float = 0.55
    tool_threshold: float = 0.60
    release_threshold: float = 0.45
    min_fastening_ticks: int = 2
    max_ticks_without_tool: int = 25
    unknown_timeout_ticks: int = 15


@dataclass(frozen=True)
class FastenerEvent:
    event_id: int
    start_timestamp: float
    end_timestamp: float
    state: str
    placement_score: float
    tool_score: float
    repeated_presses: int
    confidence: float


class FastenerCounter:
    """Causal state machine for placement -> tool work -> release.

    ``placement`` starts a candidate. Tool activity is associated with that
    candidate until a release is observed. Repeated presses within the same
    candidate are accumulated but never create another count.
    """

    def __init__(self, config: CounterConfig | None = None) -> None:
        self.config = config or CounterConfig()
        self.reset()

    def reset(self) -> None:
        self.state = CounterState.WAITING
        self.count = 0
        self.tick = 0
        self._candidate_start = 0.0
        self._placement_score = 0.0
        self._tool_score = 0.0
        self._tool_ticks = 0
        self._idle_ticks = 0
        self._unknown_ticks = 0
        self._presses = 0
        self._events: list[FastenerEvent] = []

    @property
    def events(self) -> tuple[FastenerEvent, ...]:
        return tuple(self._events)

    def update(self, sample: dict[str, Any]) -> dict[str, Any]:
        timestamp = float(sample.get("timestamp", sample.get("timestamp_sec", self.tick)))
        placement = float(sample.get("placement", sample.get("placement_score", 0.0)))
        tool = float(sample.get("tool", sample.get("tool_score", sample.get("probability", 0.0))))
        release = float(sample.get("release", sample.get("release_score", 0.0)))
        visible = bool(sample.get("visible", sample.get("input_valid", True)))
        self.tick += 1
        cfg = self.config

        if not visible:
            self._unknown_ticks += 1
            if self.state in (CounterState.FASTENER_PLACED, CounterState.FASTENING):
                if self._unknown_ticks >= cfg.unknown_timeout_ticks:
                    self.state = CounterState.UNKNOWN
            return self._result(timestamp, placement, tool)

        self._unknown_ticks = 0
        if self.state in (CounterState.WAITING, CounterState.COMPLETE, CounterState.UNKNOWN):
            if placement >= cfg.place_threshold:
                self.state = CounterState.FASTENER_PLACED
                self._candidate_start = timestamp
                self._placement_score = placement
                self._tool_score = 0.0
                self._tool_ticks = 0
                self._idle_ticks = 0
                self._presses = 0
        elif self.state == CounterState.FASTENER_PLACED:
            if tool >= cfg.tool_threshold:
                self.state = CounterState.FASTENING
                self._tool_ticks = 1
                self._tool_score = tool
                self._idle_ticks = 0
            elif release >= cfg.release_threshold:
                self._discard_candidate()
            else:
                self._idle_ticks += 1
                if self._idle_ticks > cfg.max_ticks_without_tool:
                    self._discard_candidate()
        elif self.state == CounterState.FASTENING:
            if tool >= cfg.tool_threshold:
                self._tool_ticks += 1
                self._tool_score = max(self._tool_score, tool)
                self._presses += 1
                self._idle_ticks = 0
            else:
                self._idle_ticks += 1
                if release >= cfg.release_threshold or self._idle_ticks > cfg.max_ticks_without_tool:
                    if self._tool_ticks >= cfg.min_fastening_ticks:
                        self._complete(timestamp)
                    else:
                        self._discard_candidate()

        return self._result(timestamp, placement, tool)

    def _complete(self, timestamp: float) -> None:
        confidence = min(1.0, (self._placement_score + self._tool_score) / 2.0)
        self.count += 1
        self._events.append(FastenerEvent(
            self.count, self._candidate_start, timestamp, CounterState.COMPLETE.value,
            self._placement_score, self._tool_score, self._presses, confidence,
        ))
        self.state = CounterState.COMPLETE
        self._idle_ticks = 0

    def _discard_candidate(self) -> None:
        self.state = CounterState.WAITING
        self._tool_ticks = 0
        self._idle_ticks = 0
        self._presses = 0

    def _result(self, timestamp: float, placement: float, tool: float) -> dict[str, Any]:
        return {"timestamp": timestamp, "state": self.state.value, "count": self.count,
                "placement_score": placement, "tool_score": tool,
                "event_count": len(self._events)}

    def finish(self, timestamp: float | None = None) -> dict[str, Any]:
        """Flush a candidate at end-of-stream, preserving minimum tick rules."""
        if self.state == CounterState.FASTENING and self._tool_ticks >= self.config.min_fastening_ticks:
            self._complete(float(timestamp if timestamp is not None else self.tick))
        return {"count": self.count, "events": [asdict(event) for event in self._events],
                "state": self.state.value}


def run_counter(samples: Iterable[dict[str, Any]], config: CounterConfig | None = None) -> dict[str, Any]:
    counter = FastenerCounter(config)
    timeline = [counter.update(sample) for sample in samples]
    summary = counter.finish()
    summary["timeline"] = timeline
    return summary

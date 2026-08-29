"""Repeat the labeled evaluator corpus and calibrate evaluation defaults."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import browser_evaluator, evaluator_benchmark


@dataclass(frozen=True)
class CaseCalibration:
    id: str
    domain: str
    expected_pass: bool
    outcomes: tuple[bool, ...]

    @property
    def mismatches(self) -> int:
        return sum(outcome != self.expected_pass for outcome in self.outcomes)

    @property
    def mismatch_rate(self) -> float:
        return self.mismatches / len(self.outcomes) if self.outcomes else 0.0

    @property
    def flake_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        passed = sum(self.outcomes)
        return min(passed, len(self.outcomes) - passed) / len(self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mismatches"] = self.mismatches
        payload["mismatch_rate"] = self.mismatch_rate
        payload["flake_rate"] = self.flake_rate
        return payload


@dataclass(frozen=True)
class CalibrationReport:
    corpus_version: int
    actor: str
    repetitions: int
    alpha: float
    cases: tuple[CaseCalibration, ...]
    recommended_trials: int
    composition: dict[str, Any]
    confidence_level: float = evaluator_benchmark.DEFAULT_CONFIDENCE_LEVEL

    @property
    def attempts(self) -> int:
        return sum(len(case.outcomes) for case in self.cases)

    @property
    def mismatches(self) -> int:
        return sum(case.mismatches for case in self.cases)

    @property
    def accuracy(self) -> float:
        return 1.0 - self.mismatches / self.attempts if self.attempts else 0.0

    @property
    def max_flake_rate(self) -> float:
        return max((case.flake_rate for case in self.cases), default=0.0)

    @property
    def unstable_cases(self) -> int:
        return sum(len(set(case.outcomes)) > 1 for case in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": evaluator_benchmark.REPORT_VERSION,
            "corpus_version": self.corpus_version,
            "actor": self.actor,
            "repetitions": self.repetitions,
            "alpha": self.alpha,
            "attempts": self.attempts,
            "mismatches": self.mismatches,
            "accuracy": self.accuracy,
            "max_flake_rate": self.max_flake_rate,
            "unstable_cases": self.unstable_cases,
            "recommended_trials": self.recommended_trials,
            "composition": self.composition,
            "uncertainty": {
                "method": "wilson-score",
                "confidence_level": self.confidence_level,
                "accuracy": evaluator_benchmark.proportion_interval(
                    self.attempts - self.mismatches,
                    self.attempts,
                    self.confidence_level,
                ),
                "unstable_case_rate": evaluator_benchmark.proportion_interval(
                    self.unstable_cases,
                    len(self.cases),
                    self.confidence_level,
                ),
                "scope": evaluator_benchmark.INTERVAL_SCOPE,
            },
            "cases": [case.to_dict() for case in self.cases],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def minimum_discordant_wins(alpha: float) -> int:
    """Wins needed for an all-win exact sign test to clear ``alpha``."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(0.5))


def majority_error_probability(trials: int, flake_rate: float) -> float:
    """Probability that flakes form a strict majority across odd ``trials``."""
    if trials < 1 or trials % 2 == 0:
        raise ValueError("trials must be a positive odd number")
    if not 0 <= flake_rate <= 0.5:
        raise ValueError("flake_rate must be between 0 and 0.5")
    threshold = trials // 2 + 1
    return sum(
        math.comb(trials, count) * (flake_rate**count) * ((1 - flake_rate) ** (trials - count))
        for count in range(threshold, trials + 1)
    )


def recommend_trials(flake_rate: float, alpha: float, *, maximum: int = 49) -> int:
    """Smallest odd trial count satisfying sign-test power and majority stability."""
    minimum = minimum_discordant_wins(alpha)
    trials = minimum if minimum % 2 else minimum + 1
    while trials <= maximum:
        if majority_error_probability(trials, flake_rate) <= alpha:
            return trials
        trials += 2
    return maximum


async def run_calibration(
    cases: Iterable[evaluator_benchmark.BenchmarkCase] = evaluator_benchmark.BENCHMARK_CASES,
    *,
    repetitions: int = 3,
    alpha: float = 0.05,
    actor: str = "semantic-v4",
    confidence_level: float = evaluator_benchmark.DEFAULT_CONFIDENCE_LEVEL,
    evaluator: evaluator_benchmark.Evaluator = browser_evaluator.evaluate,
) -> CalibrationReport:
    if repetitions < 2 or repetitions > 20:
        raise ValueError("repetitions must be between 2 and 20")
    case_items = tuple(cases)
    evaluator_benchmark.proportion_interval(0, 0, confidence_level)
    case_results: list[CaseCalibration] = []
    for case in case_items:
        evaluation = await evaluator(case.html, tasks=(case.task,), trials_per_task=repetitions)
        attempts = sorted(evaluation.tasks, key=lambda result: result.trial)
        if len(attempts) != repetitions:
            raise RuntimeError(
                f"evaluator returned {len(attempts)} attempts for {case.id}; expected {repetitions}"
            )
        outcomes = [attempt.passed for attempt in attempts]
        case_results.append(
            CaseCalibration(
                id=case.id,
                domain=case.domain,
                expected_pass=case.expected_pass,
                outcomes=tuple(outcomes),
            )
        )
    max_flake = max((case.flake_rate for case in case_results), default=0.0)
    return CalibrationReport(
        corpus_version=evaluator_benchmark.CORPUS_VERSION,
        actor=actor,
        repetitions=repetitions,
        alpha=alpha,
        cases=tuple(case_results),
        recommended_trials=recommend_trials(max_flake, alpha),
        composition=evaluator_benchmark.corpus_composition(case_items),
        confidence_level=confidence_level,
    )

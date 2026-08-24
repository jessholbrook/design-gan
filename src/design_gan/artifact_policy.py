"""Versioned boundary for the mutable design artifact."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactPolicy:
    """The only artifact boundary supported by this roadmap."""

    kind: str = "standalone-html"
    version: int = 1
    max_bytes: int = 512 * 1024
    network_access: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_ARTIFACT_POLICY = ArtifactPolicy()


@dataclass
class ArtifactValidation:
    passed: bool
    size_bytes: int
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def feedback(self) -> str:
        if self.passed:
            return f"Artifact boundary: PASS ({self.size_bytes} bytes, standalone/offline)."
        return "Artifact boundary: FAIL — " + "; ".join(self.violations)


_URL_ATTRIBUTE = re.compile(
    r"\b(src|srcset|href|action|poster)\s*=\s*(['\"])(.*?)\2",
    re.IGNORECASE | re.DOTALL,
)
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE | re.DOTALL)
_CSS_IMPORT = re.compile(r"@import\b", re.IGNORECASE)
_NETWORK_SCRIPT = re.compile(
    r"\b(?:fetch\s*\(|XMLHttpRequest\b|WebSocket\s*\(|EventSource\s*\(|"
    r"navigator\s*\.\s*sendBeacon\s*\(|import\s*\()",
    re.IGNORECASE,
)
_EMBEDDED_DOCUMENT = re.compile(r"<(?:iframe|object|embed)\b", re.IGNORECASE)


def _attribute_requires_network(name: str, value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    if name == "srcset":
        return any(
            _attribute_requires_network("src", candidate.strip().split()[0])
            for candidate in value.split(",")
            if candidate.strip()
        )
    lowered = value.lower()
    if name in {"src", "poster"}:
        return not lowered.startswith("data:")
    if name == "action":
        return value != "#" and not lowered.startswith("javascript:")
    return not (
        value.startswith("#") or lowered.startswith("data:") or lowered.startswith("javascript:")
    )


def validate_html(
    html: str,
    policy: ArtifactPolicy = DEFAULT_ARTIFACT_POLICY,
) -> ArtifactValidation:
    """Validate the bounded artifact without executing it."""
    if policy.kind != "standalone-html" or policy.version != 1:
        raise ValueError(f"unsupported artifact policy: {policy.kind}@{policy.version}")

    size = len(html.encode("utf-8"))
    violations: list[str] = []
    if size > policy.max_bytes:
        violations.append(f"artifact is {size} bytes; limit is {policy.max_bytes}")
    lowered = html.lower()
    if "<html" not in lowered or "<body" not in lowered:
        violations.append("artifact must contain one complete HTML document")
    if not policy.network_access:
        if any(
            _attribute_requires_network(match.group(1).lower(), match.group(3))
            for match in _URL_ATTRIBUTE.finditer(html)
        ):
            violations.append("external URL found in an HTML resource or navigation attribute")
        if _CSS_IMPORT.search(html) or any(
            not match.group(2).strip().lower().startswith(("data:", "#"))
            for match in _CSS_URL.finditer(html)
        ):
            violations.append("external URL found in CSS")
        if _NETWORK_SCRIPT.search(html):
            violations.append("network API found in inline script")
        if _EMBEDDED_DOCUMENT.search(html):
            violations.append("embedded documents are outside the artifact boundary")
    return ArtifactValidation(
        passed=not violations,
        size_bytes=size,
        violations=violations,
    )

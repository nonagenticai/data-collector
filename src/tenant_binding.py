"""Classification + in-process tally for the ``X-Tenant-Id`` → tenant binding.

``X-Tenant-Id`` is normally injected by an upstream gateway from a validated
licence token. Direct in-cluster callers can also send the header themselves, so
the service treats it as a hint unless the deployment provides peer identity at
the mesh layer. This module classifies which branch a request took and keeps an
in-process tally.

**Why this module counts rather than rejects.** Two independent reasons:

1. *No peer identity is available to reject on.* In a deployment without L7
   policy (Istio Ambient, no waypoint), no principal reaches the application, so
   the process cannot distinguish a gateway-injected header from a caller-invented
   one.
2. *The break-set for requiring a tenant is not empty.* The service supports
   fleet-global (``tenant_id IS NULL``) samples, and that bucket is still written.
   Rejecting an absent header would break the global training path.

The actual fix is deployment-side (L7 authorization keyed on the source
principal), not in this module.

**The instrument.** The tally is in-process and surfaced through a structured log
event. The WARNING fires **once per distinct (outcome, tenant) per process**;
every request is counted on every branch regardless, and the running tally rides
along on each emitted line so the counts are recoverable from logs alone.
"""

from __future__ import annotations

from enum import StrEnum
from threading import Lock
from uuid import UUID

import structlog

logger = structlog.stdlib.get_logger(__name__)


class BindOutcome(StrEnum):
    """Which branch of the binding a request took."""

    ABSENT = "absent"
    """No ``X-Tenant-Id`` at all → NULL/global sample. The flywheel's normal path."""

    BOUND = "bound"
    """A UUID was presented and bound as the caller's tenant, on a connection whose
    peer identity this process cannot verify."""

    UNPARSEABLE = "unparseable"
    """``X-Tenant-Id`` present but not a UUID (e.g. an unresolved licence token) →
    treated as absent → global."""


class BindCredential(StrEnum):
    """Whether the request carried a credential this service cannot introspect.

    Separates "no credential at all" from "holds a licence this service cannot see" —
    the latter is a caller that *is* authenticated somewhere, just not here, and the
    two need different client-side remediation.
    """

    BEARER_PRESENT = "bearer_present"
    BEARER_ABSENT = "bearer_absent"


# Cap the first-seen set so a hostile or buggy caller cycling UUIDs cannot grow it
# without bound. Once the cap is hit we stop emitting the per-tenant WARNING but keep
# counting — the counters stay correct, only the novelty signal degrades.
_MAX_TRACKED_KEYS = 512

_lock = Lock()
_counts: dict[tuple[str, str], int] = {}
_seen_keys: set[tuple[str, str, str]] = set()
_tracking_saturated = False


def reset_for_test() -> None:
    """Clear the module-level tally. Test-support only."""
    global _tracking_saturated  # noqa: PLW0603
    with _lock:
        _counts.clear()
        _seen_keys.clear()
        _tracking_saturated = False


def snapshot() -> dict[str, int]:
    """Return a copy of the tally keyed ``"<outcome>/<credential>"``."""
    with _lock:
        return {f"{outcome}/{credential}": n for (outcome, credential), n in _counts.items()}


def classify(raw_tenant: str | None, authorization: str | None) -> tuple[BindOutcome, str | None]:
    """Classify the binding a request asks for, without performing it.

    Returns the outcome and the normalised tenant UUID string (``None`` unless the
    outcome is :attr:`BindOutcome.BOUND`). Mirrors the parse in ``routers.samples.
    _parse_tenant`` — a ``Bearer `` prefix is stripped because license-authz is
    documented to inject a bare tenant and an unresolved token arrives prefixed.
    """
    del authorization  # classification of the credential is the caller's concern
    if not raw_tenant:
        return BindOutcome.ABSENT, None
    try:
        return BindOutcome.BOUND, str(UUID(raw_tenant.removeprefix("Bearer ").strip()))
    except ValueError:
        return BindOutcome.UNPARSEABLE, None


def record(outcome: BindOutcome, tenant: str | None, authorization: str | None) -> None:
    """Count this request on its branch and emit the appropriate log line.

    Every branch is counted. A WARNING is emitted only the first time this process
    sees a given ``(outcome, credential, tenant)`` triple, so the line stays a signal
    instead of firing on every one of the ~5,800 legitimately-tenanted requests a day.
    """
    credential = BindCredential.BEARER_PRESENT if authorization else BindCredential.BEARER_ABSENT
    count_key = (str(outcome), str(credential))
    novelty_key = (str(outcome), str(credential), tenant or "")

    with _lock:
        _counts[count_key] = _counts.get(count_key, 0) + 1
        total = _counts[count_key]
        first_seen = novelty_key not in _seen_keys
        if first_seen and not _tracking_saturated:
            _seen_keys.add(novelty_key)
            if len(_seen_keys) >= _MAX_TRACKED_KEYS:
                _mark_saturated()
        else:
            first_seen = False
        tally = {f"{o}/{c}": n for (o, c), n in _counts.items()}

    if outcome is BindOutcome.BOUND and first_seen:
        # The one line worth waking someone for: a tenant UUID this pod has not seen
        # before was accepted from a connection whose peer this process cannot verify.
        logger.warning(
            "tenant_bound_from_unverified_peer",
            outcome=str(outcome),
            credential=str(credential),
            tenant_id=tenant,
            occurrences=total,
            tally=tally,
        )
        return

    logger.debug(
        "tenant_binding",
        outcome=str(outcome),
        credential=str(credential),
        tenant_id=tenant,
        occurrences=total,
        tally=tally,
    )


def _mark_saturated() -> None:
    """Stop tracking novelty once the key cap is reached (caller holds ``_lock``)."""
    global _tracking_saturated  # noqa: PLW0603
    _tracking_saturated = True
    logger.warning(
        "tenant_binding_novelty_tracking_saturated",
        tracked_keys=len(_seen_keys),
        note="counters remain accurate; per-tenant first-seen warnings are suppressed",
    )

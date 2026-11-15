"""Tenant scope resolution for the READ surface, and the tally that measures it.

**The exposure this closes.** Every read handler took ``get_owner_session`` and no
tenant predicate. ``TENANT_RLS_MODE=enforce`` is live and the ``tenant_isolation_read``
policy exists, but RLS on ``inference_samples`` is ENABLE-not-FORCE
(``relforcerowsecurity=f``, verified live 2026-09-07) and ``data_collector_user`` owns
the table — so the owner bypasses the policy outright. Measured on prod the same day:
339,293 rows, 257,319 of them tenant-tagged across **19** tenants, and 80% of rows sit
in a ``schema_id`` that spans more than one tenant. The cross-tenant read was therefore
active, not latent, and ``GET /api/samples`` returns full ``messages`` +
``response_content`` — the customer content itself.

**The mechanism, and why not the alternative.** Reads that carry a tenant now run on
``get_session`` — the restricted ``data_collector_app`` role, which is
``NOBYPASSRLS``/non-owner and has carried ``SELECT`` since the Stage C cutover — so the
existing ``tenant_isolation_read`` policy (``tenant_id IS NULL OR tenant_id = GUC``)
binds with no DDL at all. The alternative, ``ALTER TABLE … FORCE ROW LEVEL SECURITY``,
was rejected: FORCE applies RLS to the **owner** too, and this service depends on the
owner keeping its bypass in three places that would break silently or loudly —
the flywheel threshold counts in ``sample_service`` (GUC unset on the owner engine ⇒ 0
rows ⇒ the trigger never fires), global ``tenant_id IS NULL`` INSERTs (96 in the 24h to
2026-09-07; the restricted role's INSERT ``WITH CHECK`` rejects them), and Alembic. It
is also the explicit doctrine of ``rls``: *"ENABLE, never
FORCE. … Applying FORCE would 0-row the owner and break them."* Choosing the role
instead of FORCE means this change needs **no migration**: it is pure routing, and
reverting the commit restores the previous behaviour with nothing to undo in the DB.

**Why an explicit predicate as well as RLS.** ``rls`` describes RLS as the
backstop *beneath* each service's application-layer ``WHERE tenant_id = …``. Keeping
both matters here for a second reason: the default test harness overrides both session
dependencies onto one owner-connected session, so an RLS-only fix would be invisible to
320 of the 321 existing tests. The predicate is provable on any connection; RLS is what
still holds when a future handler forgets it.

**NULL-tenant rows do not move.** 81,974 rows carry ``tenant_id IS NULL`` (fleet-global)
and 96 more were written in the last 24h. Both the RLS SELECT policy and the predicate
here are *global-or-own* — ``tenant_id IS NULL OR tenant_id = <caller>`` — so every
caller keeps seeing the whole global bucket exactly as before. The break-set for those
rows is empty, and that is deliberate: they are genuine fleet-wide training data.

**What this does NOT close, and why it cannot be closed here.** An *untenanted* caller
is indistinguishable from ``training-data-generator``. This namespace runs Istio Ambient
with no waypoint, so no peer identity reaches the process (see ``tenant_binding``), and
tdg sends no tenant header on any of its four call sites. So the untenanted arm of
``/count``, ``/export`` and ``/api/schemas`` — tdg's flywheel lifeline, one of which is
its readiness probe — stays cross-tenant and is instead **counted and warned**, so the
residue has a denominator and the follow-up flip is provable rather than hopeful.
``GET /api/samples`` and ``/stats``, which tdg does not call, fail closed today.
"""

from __future__ import annotations

from enum import StrEnum
from threading import Lock
from uuid import UUID

import structlog
from fastapi import HTTPException

logger = structlog.stdlib.get_logger(__name__)


class ReadScope(StrEnum):
    """Which population a read is allowed to see."""

    TENANT = "tenant"
    """The caller's own rows plus the fleet-global (NULL-tenant) bucket. Runs on the
    restricted role, so the ``tenant_isolation_read`` policy binds as well."""

    ALL_TENANTS = "all_tenants"
    """Every tenant's rows. Runs on the owner (RLS-bypass) session. Never a default —
    it must be asked for by name, exactly like ``purge``'s cross-tenant arm."""


class UntenantedReads(StrEnum):
    """What an endpoint does when the caller presents no tenant and does not ask for
    ``all_tenants``.

    The two values encode a measured difference in break-set, not a preference.
    """

    REJECT = "reject"
    """Fail closed with 400. For endpoints no live untenanted caller depends on."""

    CROSS_TENANT = "cross_tenant"
    """Keep today's cross-tenant read, but warn and count it. For the three endpoints
    ``training-data-generator`` calls untenanted; rejecting them is a flywheel outage,
    so the exposure is measured here until the caller passes ``all_tenants=true``."""


# Cap the first-seen set so a caller cycling paths cannot grow it without bound. Past
# the cap the per-key WARNING stops but the counters stay correct — the same trade
# ``tenant_binding`` makes.
_MAX_TRACKED_KEYS = 512

_lock = Lock()
_counts: dict[tuple[str, str], int] = {}
_seen_keys: set[tuple[str, str]] = set()
_tracking_saturated = False


def reset_for_test() -> None:
    """Clear the module-level tally. Test-support only."""
    global _tracking_saturated  # noqa: PLW0603
    with _lock:
        _counts.clear()
        _seen_keys.clear()
        _tracking_saturated = False


def snapshot() -> dict[str, int]:
    """Return a copy of the tally keyed ``"<endpoint>/<scope>"``."""
    with _lock:
        return {f"{endpoint}/{scope}": n for (endpoint, scope), n in _counts.items()}


def _record(endpoint: str, scope: str) -> tuple[int, bool]:
    """Count one resolution; report the running total and whether it is newly seen."""
    global _tracking_saturated  # noqa: PLW0603
    with _lock:
        key = (endpoint, scope)
        _counts[key] = _counts.get(key, 0) + 1
        first_seen = False
        if key not in _seen_keys:
            if len(_seen_keys) < _MAX_TRACKED_KEYS:
                _seen_keys.add(key)
                first_seen = True
            else:
                _tracking_saturated = True
        return _counts[key], first_seen


def resolve_read_scope(
    *,
    tenant: UUID | None,
    all_tenants: bool,
    endpoint: str,
    untenanted: UntenantedReads,
) -> ReadScope:
    """Decide which population ``endpoint`` may read, or raise 400.

    The contract mirrors ``purge_samples`` (PR #40) so the read and delete surfaces
    cannot drift apart:

    * ``all_tenants=true`` **and** a tenant present → 400. A caller that presents an
      identity and simultaneously asks to ignore it is contradicting itself; picking
      the wider reading silently is how the original hole stayed invisible.
    * ``all_tenants=true`` and no tenant → :attr:`ReadScope.ALL_TENANTS`, logged at
      WARNING. This is the named platform/flywheel arm.
    * a tenant present → :attr:`ReadScope.TENANT`. **This is the close**: every caller
      that arrives through the gateway with a license-resolved tenant, and every
      in-cluster caller that forwards one, now sees only its own rows plus the global
      bucket.
    * no tenant and no ``all_tenants`` → decided by ``untenanted``: 400 under
      :attr:`UntenantedReads.REJECT`, or a counted+warned cross-tenant read under
      :attr:`UntenantedReads.CROSS_TENANT`.

    ``endpoint`` is a stable label (e.g. ``"GET /api/samples"``) used only for the
    tally, so the residual cross-tenant surface can be attributed per route.
    """
    if all_tenants and tenant is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "all_tenants=true is a platform-maintenance path and cannot be combined "
                "with a tenant-scoped caller (X-Tenant-Id present). Drop one."
            ),
        )

    if all_tenants:
        scope = ReadScope.ALL_TENANTS
        total, first_seen = _record(endpoint, "all_tenants_explicit")
        if first_seen:
            logger.warning(
                "cross_tenant_read_requested",
                endpoint=endpoint,
                scope=str(scope),
                occurrences=total,
                tally=snapshot(),
            )
        return scope

    if tenant is not None:
        _record(endpoint, "tenant_scoped")
        return ReadScope.TENANT

    if untenanted is UntenantedReads.REJECT:
        _record(endpoint, "rejected_untenanted")
        raise HTTPException(
            status_code=400,
            detail=(
                "This read is tenant-scoped and no tenant was presented. Send the "
                "gateway-injected X-Tenant-Id, or ask for the cross-tenant platform "
                "read by name with all_tenants=true."
            ),
        )

    # CROSS_TENANT: today's behaviour, preserved so the flywheel keeps running — but
    # no longer silent. This branch IS the residual exposure; the tally is its
    # denominator and the WARNING is how it gets noticed.
    scope = ReadScope.ALL_TENANTS
    total, first_seen = _record(endpoint, "untenanted_cross_tenant")
    if first_seen:
        logger.warning(
            "untenanted_cross_tenant_read",
            endpoint=endpoint,
            scope=str(scope),
            occurrences=total,
            tally=snapshot(),
            detail=(
                "Caller presented no X-Tenant-Id and did not ask for all_tenants; served "
                "cross-tenant to keep the flywheel running. Migrate the caller to "
                "all_tenants=true, then flip this endpoint to UntenantedReads.REJECT."
            ),
        )
    return scope

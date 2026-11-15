"""Tests for the unauthenticated ``X-Tenant-Id`` binding: its tally and its GUC agreement.

Two properties, both of which were unmeasured before:

* **Every branch of the binding is counted**, and the tally separates "no credential at
  all" from "holds a licence this service cannot introspect" — the two need different
  client-side remediation. The WARNING that names a newly-seen tenant fires once per
  process per triple, so it stays a signal instead of firing on every legitimately
  tenanted request.
* **The RLS GUC agrees with the tenant the row is stamped with.** The middleware binds
  the context from the raw ``X-Tenant-Id`` only; ``resolve_tenant`` may resolve a
  *different* tenant via ``X-Effective-Tenant-Id``. Under ``TENANT_RLS_MODE=enforce``
  that disagreement makes ``tenant_isolation_ins``'s ``WITH CHECK`` reject the insert.
"""

from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from httpx import AsyncClient

import tenant_binding
from db import tenant_ctx
from routers.samples import resolve_tenant
from settings import Settings
from tenant_binding import BindCredential, BindOutcome


@pytest.fixture(autouse=True)
def _clean_tally():
    """Each test starts from an empty tally (module-level state)."""
    tenant_binding.reset_for_test()
    yield
    tenant_binding.reset_for_test()


def _delegation_settings(*delegating: str) -> Settings:
    return Settings(delegating_tenants=",".join(delegating))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassify:
    def test_absent_header_is_absent(self) -> None:
        assert tenant_binding.classify(None, None) == (BindOutcome.ABSENT, None)

    def test_empty_header_is_absent(self) -> None:
        assert tenant_binding.classify("", None) == (BindOutcome.ABSENT, None)

    def test_uuid_binds(self) -> None:
        t = uuid4()
        assert tenant_binding.classify(str(t), None) == (BindOutcome.BOUND, str(t))

    def test_bearer_prefix_is_stripped(self) -> None:
        """license-authz injects a bare tenant; an unresolved token arrives prefixed."""
        t = uuid4()
        assert tenant_binding.classify(f"Bearer {t}", None) == (BindOutcome.BOUND, str(t))

    def test_non_uuid_is_unparseable(self) -> None:
        assert tenant_binding.classify("Bearer not-a-uuid", None) == (
            BindOutcome.UNPARSEABLE,
            None,
        )

    def test_classification_matches_the_router_parse(self) -> None:
        """The middleware and the router must agree on what counts as a tenant, or the
        tally would describe a different population than the one that is bound."""
        from routers.samples import _parse_tenant

        for raw in (None, "", "not-a-uuid", str(uuid4()), f"Bearer {uuid4()}"):
            outcome, tenant = tenant_binding.classify(raw, None)
            parsed = _parse_tenant(raw)
            assert (tenant is None) == (parsed is None), raw
            if parsed is not None:
                assert tenant == str(parsed), raw
            assert (outcome is BindOutcome.BOUND) == (parsed is not None), raw


# ---------------------------------------------------------------------------
# The tally: every branch counted, credential separated
# ---------------------------------------------------------------------------


class TestTally:
    def test_every_branch_is_counted(self) -> None:
        t = str(uuid4())
        tenant_binding.record(BindOutcome.BOUND, t, None)
        tenant_binding.record(BindOutcome.ABSENT, None, None)
        tenant_binding.record(BindOutcome.UNPARSEABLE, None, None)

        snap = tenant_binding.snapshot()
        assert snap[f"{BindOutcome.BOUND}/{BindCredential.BEARER_ABSENT}"] == 1
        assert snap[f"{BindOutcome.ABSENT}/{BindCredential.BEARER_ABSENT}"] == 1
        assert snap[f"{BindOutcome.UNPARSEABLE}/{BindCredential.BEARER_ABSENT}"] == 1

    def test_bearer_presence_is_a_separate_series(self) -> None:
        """ "No credential at all" and "holds a licence this service cannot see" are
        different exposures and must not be summed into one number."""
        t = str(uuid4())
        tenant_binding.record(BindOutcome.BOUND, t, "Bearer abc")
        tenant_binding.record(BindOutcome.BOUND, t, None)

        snap = tenant_binding.snapshot()
        assert snap[f"{BindOutcome.BOUND}/{BindCredential.BEARER_PRESENT}"] == 1
        assert snap[f"{BindOutcome.BOUND}/{BindCredential.BEARER_ABSENT}"] == 1

    def test_repeat_requests_accumulate(self) -> None:
        t = str(uuid4())
        for _ in range(5):
            tenant_binding.record(BindOutcome.BOUND, t, None)
        assert tenant_binding.snapshot()[f"{BindOutcome.BOUND}/{BindCredential.BEARER_ABSENT}"] == 5

    def test_warning_fires_once_per_new_tenant_then_stays_quiet(self) -> None:
        """The signal is a tenant this pod has not seen; steady traffic must not warn."""
        t1, t2 = str(uuid4()), str(uuid4())
        with patch.object(tenant_binding.logger, "warning") as warn:
            for _ in range(3):
                tenant_binding.record(BindOutcome.BOUND, t1, None)
            assert warn.call_count == 1
            tenant_binding.record(BindOutcome.BOUND, t2, None)
            assert warn.call_count == 2

        assert tenant_binding.snapshot()[f"{BindOutcome.BOUND}/{BindCredential.BEARER_ABSENT}"] == 4

    def test_absent_branch_does_not_warn(self) -> None:
        """The flywheel's normal untenanted path is not an anomaly."""
        with patch.object(tenant_binding.logger, "warning") as warn:
            for _ in range(3):
                tenant_binding.record(BindOutcome.ABSENT, None, None)
        assert warn.call_count == 0

    def test_novelty_tracking_saturates_without_losing_counts(self) -> None:
        """A caller cycling UUIDs must not grow the tracked set without bound, and the
        counters must stay correct after novelty tracking gives up."""
        for _ in range(tenant_binding._MAX_TRACKED_KEYS + 50):
            tenant_binding.record(BindOutcome.BOUND, str(uuid4()), None)

        assert len(tenant_binding._seen_keys) <= tenant_binding._MAX_TRACKED_KEYS
        assert (
            tenant_binding.snapshot()[f"{BindOutcome.BOUND}/{BindCredential.BEARER_ABSENT}"]
            == tenant_binding._MAX_TRACKED_KEYS + 50
        )


# ---------------------------------------------------------------------------
# GUC / row agreement — the enforce-mode correctness fix
# ---------------------------------------------------------------------------


class TestGucAgreesWithResolvedTenant:
    async def test_plain_caller_binds_its_own_tenant(self) -> None:
        t = uuid4()
        token = tenant_ctx.set(None)
        try:
            assert await resolve_tenant(x_tenant_id=str(t), x_effective_tenant_id=None) == t
            assert tenant_ctx.get() == str(t)
        finally:
            tenant_ctx.reset(token)

    async def test_absent_header_leaves_guc_null(self) -> None:
        token = tenant_ctx.set("stale-value-from-a-previous-request")
        try:
            assert await resolve_tenant(x_tenant_id=None, x_effective_tenant_id=None) is None
            assert tenant_ctx.get() is None
        finally:
            tenant_ctx.reset(token)

    async def test_delegated_write_binds_the_effective_tenant_not_the_caller(self) -> None:
        """The regression this fixes: the row was stamped `customer` while the GUC still
        held `sa`, so enforce-mode's INSERT WITH CHECK rejected every delegated write."""
        sa, customer = uuid4(), uuid4()
        token = tenant_ctx.set(None)
        try:
            with patch(
                "routers.samples.get_settings",
                return_value=_delegation_settings(str(sa)),
            ):
                resolved = await resolve_tenant(
                    x_tenant_id=str(sa), x_effective_tenant_id=str(customer)
                )
            assert resolved == customer
            # The GUC must name the SAME tenant the row will carry.
            assert tenant_ctx.get() == str(customer)
        finally:
            tenant_ctx.reset(token)

    async def test_refused_delegation_binds_the_caller_not_the_asserted_tenant(self) -> None:
        """A spoof attempt must leave the GUC on the caller's own tenant."""
        caller, victim = uuid4(), uuid4()
        token = tenant_ctx.set(None)
        try:
            with patch(
                "routers.samples.get_settings",
                return_value=_delegation_settings(str(uuid4())),
            ):
                resolved = await resolve_tenant(
                    x_tenant_id=str(caller), x_effective_tenant_id=str(victim)
                )
            assert resolved == caller
            assert tenant_ctx.get() == str(caller)
        finally:
            tenant_ctx.reset(token)


# ---------------------------------------------------------------------------
# End-to-end: the value the DB layer actually sees during a write
# ---------------------------------------------------------------------------


class TestGucVisibleToTheWrite:
    async def test_delegated_request_exposes_effective_tenant_to_create_sample(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """Capture ``tenant_ctx`` at the moment ``create_sample`` runs — that is the
        value ``register_tenant_guc`` reads at transaction-begin."""
        from services import sample_service

        sa, customer = uuid4(), uuid4()
        seen: dict[str, Any] = {}
        original = sample_service.create_sample

        async def _capture(*args: Any, **kwargs: Any) -> Any:
            seen["guc"] = tenant_ctx.get(None)
            return await original(*args, **kwargs)

        with (
            patch(
                "routers.samples.get_settings",
                return_value=_delegation_settings(str(sa)),
            ),
            patch.object(sample_service, "create_sample", _capture),
        ):
            resp = await client.post(
                "/api/samples",
                json=sample_payload,
                headers={
                    "X-Tenant-Id": str(sa),
                    "X-Effective-Tenant-Id": str(customer),
                },
            )

        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(customer)
        # Before the fix this was str(sa) — the caller's tenant — and enforce-mode's
        # WITH CHECK would have rejected the row stamped with `customer`.
        assert seen["guc"] == str(customer)

    @pytest.mark.parametrize("path", ["/", "/health", "/health/live"])
    async def test_probe_paths_are_not_counted(self, client: AsyncClient, path: str) -> None:
        """kubelet probes must not enter the tally.

        They carry no tenant and fire every few seconds, so counting them would swamp
        the ``absent`` branch — and that branch's size is exactly what a future
        fail-closed has to be sized against. A denominator inflated by probe traffic
        would make the untenanted caller population look far larger than it is.
        """
        resp = await client.get(path)
        assert resp.status_code == 200
        assert tenant_binding.snapshot() == {}

    async def test_a_probe_path_with_a_tenant_header_is_still_not_counted(
        self, client: AsyncClient
    ) -> None:
        """The exclusion is by path, not by whether a header happened to be sent."""
        resp = await client.get("/health", headers={"X-Tenant-Id": str(uuid4())})
        assert resp.status_code == 200
        assert tenant_binding.snapshot() == {}

    async def test_plain_request_counts_the_bind(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """A normal tenanted request is counted on the BOUND branch by the middleware."""
        t = uuid4()
        resp = await client.post(
            "/api/samples", json=sample_payload, headers={"X-Tenant-Id": str(t)}
        )
        assert resp.status_code == 201
        assert tenant_binding.snapshot()[f"{BindOutcome.BOUND}/{BindCredential.BEARER_ABSENT}"] >= 1

    async def test_untenanted_request_counts_the_absent_branch(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """The flywheel's global path is counted too — a fail-closed needs its size."""
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201
        assert (
            tenant_binding.snapshot()[f"{BindOutcome.ABSENT}/{BindCredential.BEARER_ABSENT}"] >= 1
        )

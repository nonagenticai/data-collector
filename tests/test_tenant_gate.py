"""Tests for the multi-tenancy hard-gate (dormant landing).

Covers the schema + plumbing that lands the gate without turning enforcement on:
- per-tenant dedup via ``UNIQUE(tenant_id, content_hash) NULLS NOT DISTINCT``,
- tenant stamping from the ``X-Tenant-Id`` header (with body fallback + precedence),
- RLS policies applied by ``register_rls_ddl`` on ``create_all``,
- dormant behavior: with ``TENANT_RLS_MODE`` unset the owner connection bypasses RLS,
  so reads stay cross-tenant (no scoping) — exactly what the flywheel export needs.
"""

from typing import Any
from unittest.mock import patch
from uuid import uuid4

from httpx import AsyncClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.db import InferenceSampleDB
from models.domain import SampleCreate
from services import sample_service
from settings import Settings


def _payload(**overrides: Any) -> SampleCreate:
    uid = uuid4().hex[:12]
    defaults: dict[str, Any] = {
        "schema_id": "ks_TenantGate",
        "job_id": "job-tenant-001",
        "inference_id": f"inf-{uuid4().hex[:12]}",
        "messages": [{"role": "user", "content": "hi"}],
        "response_content": f'{{"r": "{uid}"}}',
        "provider": "Qwen3-4B-AWQ",
        "validation_passed": True,
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 10,
        "failover_chain_id": "chain-tenant-001",
    }
    defaults.update(overrides)
    return SampleCreate(**defaults)


# ---------------------------------------------------------------------------
# Per-tenant dedup semantics (service layer)
# ---------------------------------------------------------------------------


class TestPerTenantDedup:
    async def test_same_content_different_tenants_both_persist(self, session: AsyncSession) -> None:
        """Two tenants may independently hold byte-identical content."""
        t1, t2 = uuid4(), uuid4()
        # Identical content (fixed, so content_hash collides across the two).
        p = _payload(response_content='{"shared": "content"}')
        r1 = await sample_service.create_sample(session, p, t1)
        r2 = await sample_service.create_sample(session, p, t2)

        assert r1.id != r2.id
        assert r1.tenant_id == t1
        assert r2.tenant_id == t2

        count = await session.scalar(
            select(func.count())
            .select_from(InferenceSampleDB)
            .where(InferenceSampleDB.schema_id == p.schema_id)
        )
        assert count == 2

    async def test_same_content_same_tenant_dedups(self, session: AsyncSession) -> None:
        t1 = uuid4()
        p = _payload(response_content='{"shared": "content"}')
        r1 = await sample_service.create_sample(session, p, t1)
        r2 = await sample_service.create_sample(session, p, t1)

        assert r1.id == r2.id  # deduped: same row returned
        count = await session.scalar(
            select(func.count())
            .select_from(InferenceSampleDB)
            .where(InferenceSampleDB.schema_id == p.schema_id)
        )
        assert count == 1

    async def test_null_tenant_same_content_dedups_globally(self, session: AsyncSession) -> None:
        """NULL-tenant (global) rows still dedup on content_hash (NULLS NOT DISTINCT)."""
        p = _payload(response_content='{"shared": "content"}')
        r1 = await sample_service.create_sample(session, p, None)
        r2 = await sample_service.create_sample(session, p, None)

        assert r1.id == r2.id
        assert r1.tenant_id is None
        count = await session.scalar(
            select(func.count())
            .select_from(InferenceSampleDB)
            .where(InferenceSampleDB.schema_id == p.schema_id)
        )
        assert count == 1

    async def test_null_and_tagged_same_content_coexist(self, session: AsyncSession) -> None:
        """A global (NULL) row and a tenant-tagged row with identical content coexist."""
        t1 = uuid4()
        p = _payload(response_content='{"shared": "content"}')
        r_global = await sample_service.create_sample(session, p, None)
        r_tagged = await sample_service.create_sample(session, p, t1)

        assert r_global.id != r_tagged.id
        count = await session.scalar(
            select(func.count())
            .select_from(InferenceSampleDB)
            .where(InferenceSampleDB.schema_id == p.schema_id)
        )
        assert count == 2


# ---------------------------------------------------------------------------
# Tenant resolution from the request (API layer)
# ---------------------------------------------------------------------------


class TestTenantResolution:
    async def test_x_tenant_id_header_stamps_row(
        self, client: AsyncClient, session: AsyncSession, sample_payload: dict[str, Any]
    ) -> None:
        tenant = uuid4()
        resp = await client.post(
            "/api/samples", json=sample_payload, headers={"X-Tenant-Id": str(tenant)}
        )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(tenant)

        row = await session.scalar(
            select(InferenceSampleDB).where(
                InferenceSampleDB.inference_id == sample_payload["inference_id"]
            )
        )
        assert row is not None
        assert str(row.tenant_id) == str(tenant)

    async def test_no_header_is_global_null(
        self, client: AsyncClient, session: AsyncSession, sample_payload: dict[str, Any]
    ) -> None:
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] is None

        row = await session.scalar(
            select(InferenceSampleDB).where(
                InferenceSampleDB.inference_id == sample_payload["inference_id"]
            )
        )
        assert row is not None
        assert row.tenant_id is None

    async def test_header_takes_precedence_over_body(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        header_tenant = uuid4()
        body_tenant = uuid4()
        payload = {**sample_payload, "tenant_id": str(body_tenant)}
        resp = await client.post(
            "/api/samples", json=payload, headers={"X-Tenant-Id": str(header_tenant)}
        )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(header_tenant)

    async def test_body_tenant_used_when_no_header(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        body_tenant = uuid4()
        payload = {**sample_payload, "tenant_id": str(body_tenant)}
        resp = await client.post("/api/samples", json=payload)
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(body_tenant)

    async def test_non_uuid_header_treated_as_global(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """An unresolved/garbage X-Tenant-Id must not fail the write (dormant → global)."""
        resp = await client.post(
            "/api/samples", json=sample_payload, headers={"X-Tenant-Id": "Bearer not-a-uuid"}
        )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] is None


# ---------------------------------------------------------------------------
# RLS applied but dormant
# ---------------------------------------------------------------------------


class TestRlsDormant:
    async def test_rls_policies_present_on_inference_samples(self, session: AsyncSession) -> None:
        """register_rls_ddl applied the global-or-own policies via create_all."""
        policies = (
            (
                await session.execute(
                    text("SELECT policyname FROM pg_policies WHERE tablename = 'inference_samples'")
                )
            )
            .scalars()
            .all()
        )
        # Nullable tenant_id → the four "global-or-own" split policies.
        assert {
            "tenant_isolation_read",
            "tenant_isolation_ins",
            "tenant_isolation_upd",
            "tenant_isolation_del",
        } <= set(policies)

    async def test_trigger_progress_has_no_rls(self, session: AsyncSession) -> None:
        """trigger_progress carries no tenant_id → excluded → flywheel counts stay global."""
        policies = (
            (
                await session.execute(
                    text("SELECT policyname FROM pg_policies WHERE tablename = 'trigger_progress'")
                )
            )
            .scalars()
            .all()
        )
        assert list(policies) == []

    async def test_reads_cross_tenant_while_dormant(self, session: AsyncSession) -> None:
        """With TENANT_RLS_MODE unset the owner connection bypasses RLS (ENABLE-not-FORCE),
        so a read sees every tenant's rows — the cross-tenant flywheel export path."""
        t1, t2 = uuid4(), uuid4()
        await sample_service.create_sample(session, _payload(schema_id="ks_X"), t1)
        await sample_service.create_sample(session, _payload(schema_id="ks_X"), t2)
        await sample_service.create_sample(session, _payload(schema_id="ks_X"), None)

        rows = (
            (
                await session.execute(
                    select(InferenceSampleDB).where(InferenceSampleDB.schema_id == "ks_X")
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 3
        assert {r.tenant_id for r in rows} == {t1, t2, None}


# ---------------------------------------------------------------------------
# Effective-tenant delegation (API layer)
# ---------------------------------------------------------------------------


def _delegation_settings(*delegating: str) -> Settings:
    """Build a Settings whose delegation allowlist holds the given tenants."""
    return Settings(delegating_tenants=",".join(delegating))


class TestEffectiveTenantDelegation:
    """A trusted service account may stamp a run's real tenant.

    A background run reaches this service through a service account, so the
    mesh resolves ``X-Tenant-Id`` to the SA's tenant and the customer's tenant
    is lost. ``X-Effective-Tenant-Id`` carries it — but only from a caller whose
    own (mesh-injected, unforgeable) tenant is on the allowlist.
    """

    async def test_allowlisted_caller_delegates_to_effective_tenant(
        self, client: AsyncClient, session: AsyncSession, sample_payload: dict[str, Any]
    ) -> None:
        sa_tenant, customer_tenant = uuid4(), uuid4()
        with patch(
            "routers.samples.get_settings",
            return_value=_delegation_settings(str(sa_tenant)),
        ):
            resp = await client.post(
                "/api/samples",
                json=sample_payload,
                headers={
                    "X-Tenant-Id": str(sa_tenant),
                    "X-Effective-Tenant-Id": str(customer_tenant),
                },
            )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(customer_tenant)

        row = await session.scalar(
            select(InferenceSampleDB).where(
                InferenceSampleDB.inference_id == sample_payload["inference_id"]
            )
        )
        assert row is not None
        assert str(row.tenant_id) == str(customer_tenant)

    async def test_non_allowlisted_caller_cannot_delegate(
        self, client: AsyncClient, session: AsyncSession, sample_payload: dict[str, Any]
    ) -> None:
        """A caller not on the allowlist keeps its OWN tenant, not the asserted one."""
        caller_tenant, victim_tenant = uuid4(), uuid4()
        with patch(
            "routers.samples.get_settings",
            return_value=_delegation_settings(str(uuid4())),
        ):
            resp = await client.post(
                "/api/samples",
                json=sample_payload,
                headers={
                    "X-Tenant-Id": str(caller_tenant),
                    "X-Effective-Tenant-Id": str(victim_tenant),
                },
            )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(caller_tenant)

        row = await session.scalar(
            select(InferenceSampleDB).where(
                InferenceSampleDB.inference_id == sample_payload["inference_id"]
            )
        )
        assert row is not None
        assert str(row.tenant_id) == str(caller_tenant)

    async def test_empty_allowlist_is_a_no_op(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """Default config ⇒ delegation never applies (behaviour unchanged)."""
        caller_tenant, victim_tenant = uuid4(), uuid4()
        with patch("routers.samples.get_settings", return_value=_delegation_settings()):
            resp = await client.post(
                "/api/samples",
                json=sample_payload,
                headers={
                    "X-Tenant-Id": str(caller_tenant),
                    "X-Effective-Tenant-Id": str(victim_tenant),
                },
            )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(caller_tenant)

    async def test_effective_header_without_caller_tenant_is_rejected(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """No X-Tenant-Id ⇒ no established identity ⇒ delegation impossible.

        This is the anonymous-spoof path: without it, any unauthenticated caller
        could write into an arbitrary tenant simply by naming it.
        """
        victim_tenant = uuid4()
        with patch(
            "routers.samples.get_settings",
            return_value=_delegation_settings(str(uuid4())),
        ):
            resp = await client.post(
                "/api/samples",
                json=sample_payload,
                headers={"X-Effective-Tenant-Id": str(victim_tenant)},
            )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] is None

    async def test_effective_equal_to_caller_is_benign(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """The standard S2S hook stamps the header unconditionally; equal is a no-op."""
        caller_tenant = uuid4()
        with patch("routers.samples.get_settings", return_value=_delegation_settings()):
            resp = await client.post(
                "/api/samples",
                json=sample_payload,
                headers={
                    "X-Tenant-Id": str(caller_tenant),
                    "X-Effective-Tenant-Id": str(caller_tenant),
                },
            )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(caller_tenant)

    async def test_unparseable_effective_header_falls_back_to_caller(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        caller_tenant = uuid4()
        with patch(
            "routers.samples.get_settings",
            return_value=_delegation_settings(str(caller_tenant)),
        ):
            resp = await client.post(
                "/api/samples",
                json=sample_payload,
                headers={
                    "X-Tenant-Id": str(caller_tenant),
                    "X-Effective-Tenant-Id": "not-a-uuid",
                },
            )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == str(caller_tenant)


class TestDelegatingTenantsParsing:
    def test_allowlist_normalises_case_and_whitespace(self) -> None:
        tenant = uuid4()
        s = Settings(delegating_tenants=f"  {str(tenant).upper()} , ")
        assert s.delegating_tenants_set == frozenset({str(tenant)})

    def test_unparseable_entries_are_dropped_not_kept_literally(self) -> None:
        """A typo'd entry must not widen the allowlist to a literal string."""
        tenant = uuid4()
        s = Settings(delegating_tenants=f"nonsense,{tenant},")
        assert s.delegating_tenants_set == frozenset({str(tenant)})

    def test_default_is_empty(self) -> None:
        assert Settings().delegating_tenants_set == frozenset()

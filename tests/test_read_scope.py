"""The READ surface is tenant-scoped — contract + isolation tests.

**What these prove, and why on the OWNER session.** The ``client`` fixture overrides
both session dependencies onto a single owner-connected session, where RLS is bypassed
(ENABLE-not-FORCE). That is deliberate here: it means every isolation assertion below
can only pass because of the explicit ``_scope_reads_to_tenant`` predicate, never
because Postgres quietly filtered the rows. Delete the predicate and these go red on a
database that is still applying the policy correctly.

``tests/test_rls_enforce.py::TestReadScopeUnderEnforce`` covers the other half — that
the same reads also bind the ``tenant_isolation_read`` policy when they actually run on
the restricted role — so the backstop is proven too.

The three populations used throughout: a row owned by the caller (``T1``), a row owned
by a stranger (``T2``), and a fleet-global ``tenant_id IS NULL`` row. Correct scoping is
``{T1, None}``; the pre-fix behaviour was ``{T1, T2, None}``.
"""

from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

import read_scope
from models.domain import SampleCreate
from services import sample_service

SCHEMA = "ks_ReadScope"


def _payload(**overrides: Any) -> SampleCreate:
    uid = uuid4().hex[:12]
    defaults: dict[str, Any] = {
        "schema_id": SCHEMA,
        "job_id": "job-readscope-001",
        "inference_id": f"inf-{uuid4().hex[:12]}",
        "messages": [{"role": "user", "content": "hi"}],
        "response_content": f'{{"r": "{uid}"}}',
        "provider": "Qwen3-4B-AWQ",
        "validation_passed": True,
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 10,
        "failover_chain_id": "chain-readscope-001",
    }
    defaults.update(overrides)
    return SampleCreate(**defaults)


@pytest.fixture()
async def three_tenants(session: AsyncSession) -> tuple[UUID, UUID]:
    """Seed one row for T1, one for a stranger T2, and one fleet-global NULL row."""
    read_scope.reset_for_test()
    t1, t2 = uuid4(), uuid4()
    for tid in (t1, t2, None):
        await sample_service.create_sample(session, _payload(), tid)
    return t1, t2


def _tenant_header(tenant: UUID) -> dict[str, str]:
    return {"X-Tenant-Id": str(tenant)}


# ---------------------------------------------------------------------------
# The close: a caller that presents a tenant never sees a stranger's rows
# ---------------------------------------------------------------------------


class TestTenantedReadsAreScoped:
    async def test_list_excludes_foreign_tenant(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """GET /api/samples returns own + global only.

        This is the endpoint that returns ``messages`` and ``response_content``, so it
        is the one that leaked customer content.
        """
        t1, t2 = three_tenants
        resp = await client.get("/api/samples", headers=_tenant_header(t1))
        assert resp.status_code == 200
        seen = {row["tenant_id"] for row in resp.json()}
        assert seen == {str(t1), None}
        assert str(t2) not in seen

    async def test_count_excludes_foreign_tenant(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        t1, _ = three_tenants
        resp = await client.get("/api/samples/count", headers=_tenant_header(t1))
        assert resp.status_code == 200
        # T1's row + the global row; never the stranger's third row.
        assert resp.json()["count"] == 2

    async def test_export_excludes_foreign_tenant(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        t1, t2 = three_tenants
        resp = await client.get(
            "/api/samples/export", params={"schema_id": SCHEMA}, headers=_tenant_header(t1)
        )
        assert resp.status_code == 200
        lines = [ln for ln in resp.text.strip().split("\n") if ln.strip()]
        assert len(lines) == 2
        assert str(t2) not in resp.text

    async def test_stats_excludes_foreign_tenant(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """Both the aggregate and the provider breakdown must be scoped.

        The breakdown is a second query; leaving it unscoped would have re-published
        the counts the outer aggregate just hid.
        """
        t1, _ = three_tenants
        resp = await client.get("/api/samples/stats", headers=_tenant_header(t1))
        assert resp.status_code == 200
        stats = next(s for s in resp.json() if s["schema_id"] == SCHEMA)
        assert stats["total_samples"] == 2
        assert sum(stats["provider_breakdown"].values()) == 2

    async def test_schemas_excludes_a_foreign_only_schema(
        self, client: AsyncClient, session: AsyncSession, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """A schema id only a stranger has used must not appear in the caller's list."""
        t1, t2 = three_tenants
        await sample_service.create_sample(session, _payload(schema_id="ks_StrangerOnly"), t2)

        resp = await client.get("/api/schemas", headers=_tenant_header(t1))
        assert resp.status_code == 200
        ids = resp.json()["schema_ids"]
        assert SCHEMA in ids
        assert "ks_StrangerOnly" not in ids


# ---------------------------------------------------------------------------
# The 81,974 NULL-tenant rows must stay visible to everyone
# ---------------------------------------------------------------------------


class TestGlobalRowsStayVisible:
    async def test_global_row_visible_to_every_tenant(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """global-or-own, not own-only.

        Scoping to strict equality would silently remove 81,974 fleet-global rows
        (42,712 of them validated-and-unused) from every caller's view — a larger
        break than the exposure being closed.
        """
        t1, t2 = three_tenants
        for tenant in (t1, t2):
            resp = await client.get("/api/samples", headers=_tenant_header(tenant))
            assert resp.status_code == 200
            assert None in {row["tenant_id"] for row in resp.json()}

    async def test_global_row_visible_to_untenanted_cross_tenant_arm(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """The flywheel's untenanted export still sees the whole corpus."""
        resp = await client.get("/api/samples/export", params={"schema_id": SCHEMA})
        assert resp.status_code == 200
        lines = [ln for ln in resp.text.strip().split("\n") if ln.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# The named cross-tenant arm — mirrors purge's all_tenants contract
# ---------------------------------------------------------------------------


class TestAllTenantsArm:
    async def test_all_tenants_returns_every_tenant(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        t1, t2 = three_tenants
        resp = await client.get("/api/samples", params={"all_tenants": "true"})
        assert resp.status_code == 200
        assert {row["tenant_id"] for row in resp.json()} == {str(t1), str(t2), None}

    @pytest.mark.parametrize(
        ("path", "params"),
        [
            ("/api/samples", {}),
            ("/api/samples/count", {}),
            ("/api/samples/export", {"schema_id": SCHEMA}),
            ("/api/samples/stats", {}),
            ("/api/schemas", {}),
        ],
    )
    async def test_all_tenants_with_a_tenant_caller_is_rejected(
        self,
        client: AsyncClient,
        three_tenants: tuple[UUID, UUID],
        path: str,
        params: dict[str, str],
    ) -> None:
        """A caller that presents an identity AND asks to ignore it is contradicting
        itself. Fail loud rather than silently picking the wider reading — that silent
        widening is how the original hole stayed invisible."""
        t1, _ = three_tenants
        resp = await client.get(
            path, params={**params, "all_tenants": "true"}, headers=_tenant_header(t1)
        )
        assert resp.status_code == 400
        assert "all_tenants" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# The untenanted arm: which endpoints fail closed, and which are measured
# ---------------------------------------------------------------------------


class TestUntenantedArm:
    @pytest.mark.parametrize("path", ["/api/samples", "/api/samples/stats"])
    async def test_content_endpoints_fail_closed(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID], path: str
    ) -> None:
        """No live untenanted caller depends on these two, so they reject.

        model-evaluator has a client, a manifest, an AuthorizationPolicy and a
        NetworkPolicy slot for GET /api/samples — and zero traffic in 28 days. fr's
        stats binding has no in-repo caller at all.
        """
        resp = await client.get(path)
        assert resp.status_code == 400
        assert "all_tenants=true" in resp.json()["detail"]

    @pytest.mark.parametrize(
        ("path", "params"),
        [
            ("/api/samples/count", {}),
            ("/api/samples/export", {"schema_id": SCHEMA}),
            ("/api/schemas", {}),
        ],
    )
    async def test_flywheel_endpoints_stay_cross_tenant(
        self,
        client: AsyncClient,
        three_tenants: tuple[UUID, UUID],
        path: str,
        params: dict[str, str],
    ) -> None:
        """training-data-generator calls these three untenanted and sends no header on
        any of its call sites; one of them backs its readiness probe. Rejecting them
        would be a flywheel outage, not a fix — so they are measured instead."""
        resp = await client.get(path, params=params)
        assert resp.status_code == 200

    async def test_the_residual_exposure_is_counted(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """The branch that is deliberately left open must carry its own denominator.

        This is what makes the follow-up flip provable: when tdg moves to
        ``all_tenants=true`` these counters go to zero, and only then is switching
        those endpoints to ``REJECT`` a measurement rather than a hope.
        """
        read_scope.reset_for_test()
        await client.get("/api/samples/count")
        await client.get("/api/schemas")
        await client.get("/api/samples/count", headers=_tenant_header(three_tenants[0]))

        tally = read_scope.snapshot()
        assert tally["GET /api/samples/count/untenanted_cross_tenant"] == 1
        assert tally["GET /api/schemas/untenanted_cross_tenant"] == 1
        assert tally["GET /api/samples/count/tenant_scoped"] == 1


# ---------------------------------------------------------------------------
# Unparseable / delegated header handling reuses the write path's resolver
# ---------------------------------------------------------------------------


class TestHeaderParsing:
    async def test_unparseable_tenant_is_treated_as_absent(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """An unresolved licence token is not a tenant. It must fall to the untenanted
        arm — never be accepted as an identity, and never crash the cast."""
        resp = await client.get("/api/samples", headers={"X-Tenant-Id": "not-a-uuid"})
        assert resp.status_code == 400

    async def test_bearer_prefixed_tenant_is_accepted(
        self, client: AsyncClient, three_tenants: tuple[UUID, UUID]
    ) -> None:
        """``resolve_tenant`` strips a ``Bearer `` prefix; the read path must inherit
        that rather than re-implement parsing and drift from the write path."""
        t1, _ = three_tenants
        resp = await client.get("/api/samples", headers={"X-Tenant-Id": f"Bearer {t1}"})
        assert resp.status_code == 200
        assert {row["tenant_id"] for row in resp.json()} == {str(t1), None}


# ---------------------------------------------------------------------------
# Which CONNECTION a read lands on — the half RLS depends on
# ---------------------------------------------------------------------------


class TestSessionRouting:
    """The router must send a scoped read to the RUNTIME session and a cross-tenant
    read to the OWNER session.

    Without this, ``_pick_session`` is untested: every other test in this file calls
    the service layer through a fixture that maps both dependencies onto one session,
    and ``test_rls_enforce.py`` picks its session by hand. A mutation that routed every
    read back to the owner therefore SURVIVED — it silently removes the RLS backstop
    while leaving the explicit predicate (and so every assertion) intact. That is
    exactly the failure this class exists to catch: the predicate is the visible
    control, the connection is the one that still holds when the predicate is wrong.
    """

    @pytest.fixture()
    def routed(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """Capture which session object the router handed to the service layer."""
        seen: dict[str, object] = {}
        real = sample_service.list_samples

        async def spy(session, filters, **kwargs):  # noqa: ANN001, ANN202
            seen["session"] = session
            seen["all_tenants"] = kwargs.get("all_tenants")
            seen["tenant_id"] = kwargs.get("tenant_id")
            return await real(session, filters, **kwargs)

        monkeypatch.setattr(sample_service, "list_samples", spy)
        return seen

    @pytest.fixture()
    async def split_client(self, engine, routed):  # noqa: ANN001, ANN201
        """A client whose two session dependencies yield DISTINCT session objects.

        The shared ``client`` fixture maps both onto one session, which is why it
        cannot see routing at all.
        """
        from sqlalchemy.ext.asyncio import async_sessionmaker

        from db import get_owner_session, get_session
        from main import app

        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        runtime, owner = factory(), factory()

        async def _runtime():  # noqa: ANN202
            yield runtime

        async def _owner():  # noqa: ANN202
            yield owner

        app.dependency_overrides[get_session] = _runtime
        app.dependency_overrides[get_owner_session] = _owner
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac, runtime, owner
        app.dependency_overrides.clear()
        await runtime.close()
        await owner.close()

    async def test_tenanted_read_uses_the_runtime_session(
        self, split_client, routed: dict[str, object]
    ) -> None:  # noqa: ANN001
        """A scoped read must run on the restricted role, so ``tenant_isolation_read``
        binds beneath the predicate."""
        ac, runtime, owner = split_client
        resp = await ac.get("/api/samples", headers=_tenant_header(uuid4()))
        assert resp.status_code == 200
        assert routed["session"] is runtime
        assert routed["session"] is not owner
        assert routed["all_tenants"] is False

    async def test_all_tenants_read_uses_the_owner_session(
        self, split_client, routed: dict[str, object]
    ) -> None:  # noqa: ANN001
        """The named cross-tenant arm must run on the owner — it is the only connection
        that legitimately bypasses RLS (ENABLE-not-FORCE)."""
        ac, runtime, owner = split_client
        resp = await ac.get("/api/samples", params={"all_tenants": "true"})
        assert resp.status_code == 200
        assert routed["session"] is owner
        assert routed["session"] is not runtime
        assert routed["all_tenants"] is True

"""Enforce-mode RLS tests — the two-role split the enforce cutover flips on.

Unlike ``test_tenant_gate.py`` (dormant: one owner connection, RLS bypassed), these
tests stand up the real posture: a **restricted non-owner role** bound by RLS plus
the **owner** (RLS-bypass) connection, with ``TENANT_RLS_MODE=enforce`` forced on the
runtime engine. They prove the properties the cutover depends on:

* the runtime role is isolated — it sees only NULL(global)+own rows and cannot write
  another tenant's or a global (NULL) row (``WITH CHECK``);
* global (NULL) writes succeed only on the owner connection — which is exactly why
  ``create_sample`` routes them there;
* the flywheel's cross-tenant COUNT must run on the owner connection (the runtime
  role would scope it to one tenant and silently under-count);
* anti-drift: RLS is ENABLE-not-FORCE and the restricted role is NOSUPERUSER /
  NOBYPASSRLS / LOGIN.

The restricted role is emulated with ``SET ROLE`` bound at transaction-begin (same
DSN, no password/pg_hba dependency) so this runs anywhere the owner can CREATE ROLE;
it ``pytest.skip``s cleanly where it cannot.
"""

from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import Base
from models.db import InferenceSampleDB  # noqa: F401 -- registers model + RLS after_create
from models.domain import SampleCreate, SampleFilter
from rls import register_tenant_guc
from services import sample_service
from tests.conftest import TEST_DATABASE_URL, require_database  # noqa: E402
from tests.db_url import mask, running_in_ci  # noqa: E402

_RESTRICTED_ROLE = "dc_rls_test_app"


def _payload(**overrides: Any) -> SampleCreate:
    uid = uuid4().hex[:12]
    defaults: dict[str, Any] = {
        "schema_id": "ks_Enforce",
        "job_id": "job-enforce-001",
        "inference_id": f"inf-{uuid4().hex[:12]}",
        "messages": [{"role": "user", "content": "hi"}],
        "response_content": f'{{"r": "{uid}"}}',
        "provider": "Qwen3-4B-AWQ",
        "validation_passed": True,
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 10,
        "failover_chain_id": "chain-enforce-001",
    }
    defaults.update(overrides)
    return SampleCreate(**defaults)


class _Env:
    """Handles + factories for the two-role enforce harness."""

    def __init__(self, owner_engine, runtime_engine, tenant_ctx) -> None:  # noqa: ANN001
        self.owner_engine = owner_engine
        self.runtime_engine = runtime_engine
        self.tenant_ctx = tenant_ctx
        self.OwnerSession = async_sessionmaker(bind=owner_engine, expire_on_commit=False)
        self.RuntimeSession = async_sessionmaker(bind=runtime_engine, expire_on_commit=False)


@pytest.fixture()
async def enforce_env(request: pytest.FixtureRequest):  # noqa: ANN201
    """Owner engine + a runtime engine that acts as a restricted non-owner role.

    The runtime engine uses the same DSN but a transaction-begin listener issues
    ``SET ROLE`` (restricted, RLS-bound) and ``register_tenant_guc(mode='enforce')``
    binds ``app.current_tenant`` from ``tenant_ctx``.

    ⚠️ Two DIFFERENT skips live in this fixture and only one of them is still a
    skip. Losing CREATE ROLE / SET ROLE is a CAPABILITY limit — a legitimate
    conditional skip, kept below. "PostgreSQL not available" was not: it caught
    every connect failure into one outcome, so a database that simply did not
    exist on a live server vanished as `14 skipped, exit 0` — against this file,
    which is the repo's RLS tenant-isolation proof, and whose whole value is that
    it RAN. `require_database` splits the two states: ABSENT is loud everywhere,
    an absent SERVER still skips locally and fails in CI.
    """
    require_database("tests/test_rls_enforce.py (the RLS tenant-isolation proof)")
    owner_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    try:
        async with owner_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        await owner_engine.dispose()
        # The probe above said PRESENT, so this is a genuine failure of something
        # other than existence or reachability. Same posture as conftest.engine.
        if running_in_ci():
            pytest.fail(
                f"Test database at {mask(TEST_DATABASE_URL)} answered a probe, but the "
                f"RLS proof could not use it: {e}",
                pytrace=False,
            )
        pytest.skip(f"cannot use PostgreSQL at {mask(TEST_DATABASE_URL)}: {e}")

    # Fresh schema (register_rls_ddl applies the global-or-own policies on create_all).
    async with owner_engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Provision the restricted role + grants; probe SET ROLE. Skip if not permitted.
    try:
        async with owner_engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DO $$
                    BEGIN
                        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_RESTRICTED_ROLE}') THEN
                            EXECUTE 'DROP OWNED BY {_RESTRICTED_ROLE}';
                            EXECUTE 'DROP ROLE {_RESTRICTED_ROLE}';
                        END IF;
                        CREATE ROLE {_RESTRICTED_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS
                            NOCREATEROLE NOCREATEDB;
                    END
                    $$;
                    """
                )
            )
            await conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_RESTRICTED_ROLE}"))
            await conn.execute(
                text(
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                    f"IN SCHEMA public TO {_RESTRICTED_ROLE}"
                )
            )
            # Ensure the connecting user may SET ROLE to it (superuser already can).
            await conn.execute(text(f"GRANT {_RESTRICTED_ROLE} TO CURRENT_USER"))
            # Probe: SET ROLE must actually work in this environment.
            await conn.execute(text(f"SET LOCAL ROLE {_RESTRICTED_ROLE}"))
    except (ProgrammingError, DBAPIError) as e:
        await owner_engine.dispose()
        pytest.skip(f"restricted-role harness unavailable: {e}")

    tenant_ctx: ContextVar[str | None] = ContextVar("enforce_tenant_ctx", default=None)
    runtime_engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Act as the restricted role for every runtime transaction, then bind the GUC.
    @event.listens_for(runtime_engine.sync_engine, "begin")
    def _assume_restricted_role(conn) -> None:  # noqa: ANN001
        conn.exec_driver_sql(f"SET ROLE {_RESTRICTED_ROLE}")

    register_tenant_guc(runtime_engine, lambda: tenant_ctx.get(None), mode="enforce")

    env = _Env(owner_engine, runtime_engine, tenant_ctx)
    try:
        yield env
    finally:
        await runtime_engine.dispose()
        async with owner_engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.execute(text(f"DROP OWNED BY {_RESTRICTED_ROLE}"))
            await conn.execute(text(f"DROP ROLE IF EXISTS {_RESTRICTED_ROLE}"))
        await owner_engine.dispose()


# ---------------------------------------------------------------------------
# Anti-drift: ENABLE-not-FORCE + restricted role attributes
# ---------------------------------------------------------------------------


class TestAntiDrift:
    async def test_rls_enabled_not_forced(self, enforce_env: _Env) -> None:
        """inference_samples must be ENABLE'd (so the restricted role binds) but never
        FORCE'd (so the owner keeps its cross-tenant bypass)."""
        async with enforce_env.OwnerSession() as s:
            row = (
                await s.execute(
                    text(
                        "SELECT relrowsecurity, relforcerowsecurity "
                        "FROM pg_class WHERE relname = 'inference_samples'"
                    )
                )
            ).one()
        assert row.relrowsecurity is True
        assert row.relforcerowsecurity is False

    async def test_trigger_progress_not_under_rls(self, enforce_env: _Env) -> None:
        """The high-water marker table stays cross-tenant (no RLS)."""
        async with enforce_env.OwnerSession() as s:
            enabled = (
                await s.execute(
                    text("SELECT relrowsecurity FROM pg_class WHERE relname = 'trigger_progress'")
                )
            ).scalar_one()
        assert enabled is False

    async def test_restricted_role_attributes(self, enforce_env: _Env) -> None:
        """The runtime role must be login-capable, non-super, non-bypassrls."""
        async with enforce_env.OwnerSession() as s:
            row = (
                await s.execute(
                    text(
                        "SELECT rolsuper, rolbypassrls, rolcanlogin "
                        f"FROM pg_roles WHERE rolname = '{_RESTRICTED_ROLE}'"
                    )
                )
            ).one()
        assert row.rolsuper is False
        assert row.rolbypassrls is False
        assert row.rolcanlogin is True


# ---------------------------------------------------------------------------
# Read isolation + global visibility under the restricted role
# ---------------------------------------------------------------------------


class TestReadIsolation:
    async def _seed(self, enforce_env: _Env, t1, t2) -> None:  # noqa: ANN001
        """Seed a T1 row, a T2 row, and a NULL(global) row via the owner."""
        async with enforce_env.OwnerSession() as s:
            for tid in (t1, t2, None):
                await sample_service.create_sample(s, _payload(schema_id="ks_Iso"), tid)

    async def test_runtime_sees_only_own_and_global(self, enforce_env: _Env) -> None:
        t1, t2 = uuid4(), uuid4()
        await self._seed(enforce_env, t1, t2)

        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as s:
            rows = (
                (
                    await s.execute(
                        select(InferenceSampleDB).where(InferenceSampleDB.schema_id == "ks_Iso")
                    )
                )
                .scalars()
                .all()
            )
        # T1's own row + the NULL global row; never T2's.
        assert {r.tenant_id for r in rows} == {t1, None}

    async def test_owner_sees_all_tenants(self, enforce_env: _Env) -> None:
        t1, t2 = uuid4(), uuid4()
        await self._seed(enforce_env, t1, t2)

        async with enforce_env.OwnerSession() as s:
            rows = (
                (
                    await s.execute(
                        select(InferenceSampleDB).where(InferenceSampleDB.schema_id == "ks_Iso")
                    )
                )
                .scalars()
                .all()
            )
        assert {r.tenant_id for r in rows} == {t1, t2, None}


# ---------------------------------------------------------------------------
# Write isolation (WITH CHECK) + why global writes must use the owner
# ---------------------------------------------------------------------------


class TestWriteIsolation:
    async def test_runtime_can_write_own_tenant(self, enforce_env: _Env) -> None:
        t1 = uuid4()
        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as runtime, enforce_env.OwnerSession() as owner:
            result = await sample_service.create_sample(
                runtime, _payload(), t1, owner_session=owner
            )
        assert result.tenant_id == t1

    async def test_runtime_cannot_write_foreign_tenant(self, enforce_env: _Env) -> None:
        """GUC=T1 but a row tagged T2 must be rejected by the INSERT WITH CHECK."""
        t1, t2 = uuid4(), uuid4()
        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as s:
            s.add(_orm_row(t2))
            with pytest.raises(DBAPIError):
                await s.commit()

    async def test_runtime_cannot_write_global_null(self, enforce_env: _Env) -> None:
        """A NULL(global) row fails WITH CHECK on the restricted role — the reason
        create_sample routes global writes to the owner connection."""
        t1 = uuid4()
        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as s:
            s.add(_orm_row(None))
            with pytest.raises(DBAPIError):
                await s.commit()

    async def test_create_sample_routes_global_write_to_owner(self, enforce_env: _Env) -> None:
        """create_sample(tenant_id=None) succeeds by routing the write to the owner,
        and the global row is then visible to a scoped runtime read."""
        t1 = uuid4()
        # No tenant bound: a global write. Runtime session would fail WITH CHECK,
        # but create_sample must route it to owner_session.
        async with enforce_env.RuntimeSession() as runtime, enforce_env.OwnerSession() as owner:
            result = await sample_service.create_sample(
                runtime, _payload(schema_id="ks_Global"), None, owner_session=owner
            )
        assert result.tenant_id is None

        # The global row is readable by any tenant (global-or-own SELECT policy).
        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as s:
            count = await s.scalar(
                select(func.count())
                .select_from(InferenceSampleDB)
                .where(InferenceSampleDB.schema_id == "ks_Global")
            )
        assert count == 1


# ---------------------------------------------------------------------------
# The flywheel COUNT must run on the owner connection
# ---------------------------------------------------------------------------


class TestFlywheelCountNeedsOwner:
    async def test_owner_count_is_cross_tenant_runtime_is_scoped(self, enforce_env: _Env) -> None:
        t1, t2 = uuid4(), uuid4()
        async with enforce_env.OwnerSession() as s:
            for tid in (t1, t2, None):
                await sample_service.create_sample(s, _payload(schema_id="ks_Count"), tid)

        # Owner: the true cross-tenant flywheel count (what get_count / the threshold
        # checks rely on) sees all three.
        async with enforce_env.OwnerSession() as s:
            owner_count = await s.scalar(
                select(func.count())
                .select_from(InferenceSampleDB)
                .where(InferenceSampleDB.validation_passed.is_(True))
                .where(InferenceSampleDB.used_in_training.is_(False))
            )
        assert owner_count == 3

        # Runtime scoped to T1: only T1 + NULL — proving a naive flip that left the
        # count on the runtime role would silently under-count the flywheel.
        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as s:
            runtime_count = await s.scalar(
                select(func.count())
                .select_from(InferenceSampleDB)
                .where(InferenceSampleDB.validation_passed.is_(True))
                .where(InferenceSampleDB.used_in_training.is_(False))
            )
        assert runtime_count == 2


def _orm_row(tenant_id) -> InferenceSampleDB:  # noqa: ANN001
    """A minimal valid ORM row tagged with the given tenant_id."""
    from uuid import uuid7

    return InferenceSampleDB(
        id=str(uuid7()),
        tenant_id=tenant_id,
        schema_id="ks_Write",
        job_id="job-w",
        inference_id=f"inf-{uuid4().hex[:12]}",
        messages=[{"role": "user", "content": "hi"}],
        response_content='{"r": "x"}',
        provider="Qwen3-4B-AWQ",
        validation_passed=True,
        fallback_used=False,
        fallback_reason=None,
        latency_ms=10,
        failover_chain_id="chain-w",
        used_in_training=False,
        training_batch_id=None,
        content_hash=None,
    )


# ---------------------------------------------------------------------------
# The read close, proven on the RESTRICTED role (the RLS half)
# ---------------------------------------------------------------------------


class TestReadScopeUnderEnforce:
    """``tests/test_read_scope.py`` proves the explicit predicate on an owner
    connection, where RLS is bypassed. These prove the other half: when a scoped read
    actually runs on the restricted role, ``tenant_isolation_read`` binds underneath
    it — so a handler that ever loses its predicate still returns zero foreign rows.

    This is the property that makes changing the ROLE, rather than applying
    ``FORCE ROW LEVEL SECURITY``, a real control and not just a refactor.
    """

    async def _seed(self, enforce_env: _Env, t1, t2) -> None:  # noqa: ANN001
        async with enforce_env.OwnerSession() as s:
            for tid in (t1, t2, None):
                await sample_service.create_sample(s, _payload(schema_id="ks_ReadRls"), tid)

    async def test_scoped_read_with_no_predicate_still_excludes_foreign_tenant(
        self, enforce_env: _Env
    ) -> None:
        """The backstop: ``all_tenants=True`` drops the application predicate entirely,
        yet the restricted role still cannot see T2's row."""
        t1, t2 = uuid4(), uuid4()
        await self._seed(enforce_env, t1, t2)

        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as s:
            rows = await sample_service.list_samples(
                s,
                SampleFilter(schema_id="ks_ReadRls", limit=100, offset=0),
                tenant_id=t1,
                all_tenants=True,  # predicate deliberately disabled — RLS must still hold
            )
        assert {r.tenant_id for r in rows} == {t1, None}
        assert t2 not in {r.tenant_id for r in rows}

    async def test_predicate_and_policy_agree(self, enforce_env: _Env) -> None:
        """Predicate-on and predicate-off must return the SAME rows on the restricted
        role. A disagreement means the two have drifted and one of them is wrong."""
        t1, t2 = uuid4(), uuid4()
        await self._seed(enforce_env, t1, t2)

        enforce_env.tenant_ctx.set(str(t1))
        async with enforce_env.RuntimeSession() as s:
            with_predicate = await sample_service.list_samples(
                s, SampleFilter(schema_id="ks_ReadRls", limit=100, offset=0), tenant_id=t1
            )
            without_predicate = await sample_service.list_samples(
                s,
                SampleFilter(schema_id="ks_ReadRls", limit=100, offset=0),
                tenant_id=t1,
                all_tenants=True,
            )
        assert {r.id for r in with_predicate} == {r.id for r in without_predicate}

    async def test_untenanted_restricted_read_sees_only_the_global_bucket(
        self, enforce_env: _Env
    ) -> None:
        """With no GUC bound the fail-closed policy leaves exactly the NULL/global rows.

        This is what an untenanted read WOULD return if ``/count``, ``/export`` and
        ``/api/schemas`` were flipped to ``UntenantedReads.REJECT``'s scoped sibling —
        i.e. the measured size of the break that migrating tdg has to absorb.
        """
        t1, t2 = uuid4(), uuid4()
        await self._seed(enforce_env, t1, t2)

        enforce_env.tenant_ctx.set(None)
        async with enforce_env.RuntimeSession() as s:
            rows = await sample_service.list_samples(
                s, SampleFilter(schema_id="ks_ReadRls", limit=100, offset=0), tenant_id=None
            )
        assert {r.tenant_id for r in rows} == {None}

    async def test_owner_all_tenants_still_sees_everything(self, enforce_env: _Env) -> None:
        """The named platform arm must keep working — it is what the flywheel uses, and
        it is the reason RLS is ENABLE-not-FORCE rather than FORCE."""
        t1, t2 = uuid4(), uuid4()
        await self._seed(enforce_env, t1, t2)

        async with enforce_env.OwnerSession() as s:
            rows = await sample_service.list_samples(
                s,
                SampleFilter(schema_id="ks_ReadRls", limit=100, offset=0),
                all_tenants=True,
            )
        assert {r.tenant_id for r in rows} == {t1, t2, None}

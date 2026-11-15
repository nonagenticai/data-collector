"""Direct unit tests for services/sample_service.py -- covering uncovered branches.

All tests use the real async PostgreSQL/SQLAlchemy stack (no mocks for DB operations).
Payload uniqueness uses uuid4 instead of global counters to eliminate
cross-test ordering dependencies.
"""

import asyncio
import inspect
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from models.db import InferenceSampleDB
from models.domain import ExportParams, MarkUsedRequest, SampleCreate, SampleFilter
from services import sample_service
from services.sample_service import _compute_content_hash, _row_to_domain


def _make_payload(**overrides: Any) -> SampleCreate:
    """Helper to build a SampleCreate with sensible defaults.

    Each call generates unique response_content and inference_id via uuid4
    to avoid dedup unless explicitly overridden.
    """
    uid = uuid4().hex[:12]
    defaults: dict[str, Any] = {
        "schema_id": "ks_TestAgent",
        "job_id": "job-test-001",
        "inference_id": f"inf-{uuid4().hex[:12]}",
        "messages": [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Test input."},
        ],
        "response_content": f'{{"result": "ok", "uid": "{uid}"}}',
        "provider": "Qwen3-4B-AWQ",
        "validation_passed": True,
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 100,
        "failover_chain_id": "chain-test-001",
    }
    defaults.update(overrides)
    return SampleCreate(**defaults)


# ---------------------------------------------------------------------------
# create_sample
# ---------------------------------------------------------------------------


class TestCreateSample:
    async def test_create_sample_persists_all_fields(self, session: AsyncSession) -> None:
        """create_sample should persist every field and return a complete domain model."""
        payload = _make_payload(
            schema_id="ks_Persist",
            job_id="job-persist-42",
            provider="claude",
            validation_passed=False,
            fallback_used=True,
            fallback_reason="timeout on primary",
            latency_ms=250,
            failover_chain_id="chain-persist-001",
        )
        result = await sample_service.create_sample(session, payload)

        # Verify domain model fields match input
        assert result.schema_id == "ks_Persist"
        assert result.job_id == "job-persist-42"
        assert result.provider == "claude"
        assert result.validation_passed is False
        assert result.fallback_used is True
        assert result.fallback_reason == "timeout on primary"
        assert result.latency_ms == 250
        assert result.failover_chain_id == "chain-persist-001"
        assert result.used_in_training is False
        assert result.training_batch_id is None
        assert isinstance(result.id, UUID)
        assert isinstance(result.created_at, datetime)

        # Verify actually persisted in DB by querying directly
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(result.id))
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.schema_id == "ks_Persist"
        assert row.job_id == "job-persist-42"
        assert row.provider == "claude"
        assert row.validation_passed is False
        assert row.fallback_used is True
        assert row.fallback_reason == "timeout on primary"
        assert row.latency_ms == 250
        assert row.content_hash is not None

    async def test_create_sample_deduplication_returns_same_record(
        self, session: AsyncSession
    ) -> None:
        """create_sample should return the existing sample on duplicate content hash."""
        fixed_content = '{"dedup": "test-value"}'
        payload = _make_payload(response_content=fixed_content)
        first = await sample_service.create_sample(session, payload)
        second = await sample_service.create_sample(session, payload)

        # Same ID returned
        assert first.id == second.id
        # All fields identical
        assert first.schema_id == second.schema_id
        assert first.provider == second.provider

        # Only one row in DB
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.schema_id == payload.schema_id)
        db_result = await session.execute(stmt)
        rows = db_result.scalars().all()
        assert len(rows) == 1

    async def test_create_sample_content_hash_computed_correctly(
        self, session: AsyncSession
    ) -> None:
        """The content_hash stored in DB should match _compute_content_hash."""
        payload = _make_payload()
        result = await sample_service.create_sample(session, payload)

        expected_hash = _compute_content_hash(
            payload.schema_id, payload.job_id, payload.messages, payload.response_content
        )

        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(result.id))
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.content_hash == expected_hash

    async def test_create_sample_handles_preexisting_duplicates(
        self, session: AsyncSession
    ) -> None:
        """create_sample should not crash when duplicate content_hash rows already exist.

        Simulates the pre-migration state by dropping the unique index and
        inserting two rows with the same content_hash via raw SQL (bypasses
        ORM sentinel tracking which is strict on UUID type matching).
        """
        import json as _json

        # Drop the live unique index so we can insert duplicates
        await session.execute(text("DROP INDEX IF EXISTS ix_samples_tenant_content_hash_unique"))
        await session.commit()

        fixed_content = '{"preexisting": "dup-test"}'
        schema_id = "ks_TestAgent"
        job_id = "job-dup-shared"
        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Test input."},
        ]
        content_hash = _compute_content_hash(schema_id, job_id, messages, fixed_content)

        # Insert two rows via raw SQL to bypass ORM sentinel restrictions
        for i in range(2):
            row_id = str(uuid4())
            await session.execute(
                text(
                    """
                    INSERT INTO inference_samples
                        (id, schema_id, job_id, inference_id, messages, response_content,
                         provider, validation_passed, fallback_used,
                         fallback_reason, latency_ms, failover_chain_id,
                         used_in_training, training_batch_id, content_hash)
                    VALUES
                        (:id, :schema_id, :job_id, :inference_id, :messages, :response_content,
                         :provider, :validation_passed, :fallback_used,
                         :fallback_reason, :latency_ms, :failover_chain_id,
                         :used_in_training, :training_batch_id, :content_hash)
                    """
                ),
                {
                    "id": row_id,
                    "schema_id": schema_id,
                    "job_id": job_id,
                    "inference_id": f"inf-dup-{i}",
                    "messages": _json.dumps(messages),
                    "response_content": fixed_content,
                    "provider": "Qwen3-4B-AWQ",
                    "validation_passed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "latency_ms": 100,
                    "failover_chain_id": "chain-dup",
                    "used_in_training": False,
                    "training_batch_id": None,
                    "content_hash": content_hash,
                },
            )
        await session.commit()

        # Verify two rows exist
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.content_hash == content_hash)
        result = await session.execute(stmt)
        assert len(result.scalars().all()) == 2

        # create_sample should return the oldest existing row, not crash
        payload = _make_payload(
            schema_id=schema_id,
            job_id=job_id,
            messages=messages,
            response_content=fixed_content,
        )
        sample = await sample_service.create_sample(session, payload)
        assert sample.schema_id == schema_id
        assert sample.response_content == fixed_content

    async def test_create_sample_messages_stored_as_json(self, session: AsyncSession) -> None:
        """Messages field should be stored and retrieved as a list of dicts."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User query"},
            {"role": "assistant", "content": "Response"},
        ]
        payload = _make_payload(messages=messages)
        result = await sample_service.create_sample(session, payload)

        assert result.messages == messages
        assert len(result.messages) == 3
        assert result.messages[0]["role"] == "system"
        assert result.messages[2]["role"] == "assistant"


# ---------------------------------------------------------------------------
# outcome / finish_reason / tool_calls must stay OUT of the dedup hash (R22)
# ---------------------------------------------------------------------------


class TestContentHashIgnoresOutcomeFields:
    """``_compute_content_hash`` feeds the UNIQUE index
    ``ix_samples_tenant_content_hash_unique``. Adding any field to it silently
    changes dedup volume -- and therefore *training* volume -- across the whole
    corpus. These pin the hash inputs at exactly four."""

    async def test_content_hash_ignores_new_fields(self, session: AsyncSession) -> None:
        """Two payloads differing ONLY in outcome/finish_reason/tool_calls dedup
        to a single row."""
        fixed_content = '{"dedup": "r22-hash-pin"}'
        shared: dict[str, Any] = {
            "schema_id": "ks_HashPin",
            "job_id": "job-hash-pin-001",
            "inference_id": "inf-hash-pin-001",
            "response_content": fixed_content,
        }
        tool_calls = [
            {
                "id": "call_hashpin01",
                "type": "function",
                "function": {"name": "noop", "arguments": '{"a": 1}'},
            }
        ]
        first_payload = _make_payload(
            **shared, outcome="ok", finish_reason="tool_calls", tool_calls=tool_calls
        )
        second_payload = _make_payload(
            **shared, outcome="empty", finish_reason="length", tool_calls=None
        )

        # NON-VACUITY: the two payloads must genuinely differ on the three
        # fields. If SampleCreate silently dropped them (pydantic extra=ignore)
        # the payloads would be identical and "they dedup" would prove nothing.
        assert getattr(first_payload, "outcome", "<no-such-field>") == "ok"
        assert getattr(second_payload, "outcome", "<no-such-field>") == "empty"
        assert getattr(first_payload, "finish_reason", "<no-such-field>") == "tool_calls"
        assert getattr(second_payload, "finish_reason", "<no-such-field>") == "length"
        assert getattr(first_payload, "tool_calls", "<no-such-field>") == tool_calls
        assert getattr(second_payload, "tool_calls", "<no-such-field>") is None
        # ...and everything the hash IS allowed to see is identical.
        assert first_payload.schema_id == second_payload.schema_id
        assert first_payload.job_id == second_payload.job_id
        assert first_payload.messages == second_payload.messages
        assert first_payload.response_content == second_payload.response_content

        first = await sample_service.create_sample(session, first_payload)
        second = await sample_service.create_sample(session, second_payload)

        # Deduped: same row returned, exactly one row on disk.
        assert first.id == second.id
        db_result = await session.execute(
            select(InferenceSampleDB).where(InferenceSampleDB.schema_id == "ks_HashPin")
        )
        rows = db_result.scalars().all()
        assert len(rows) == 1, f"dedup volume changed: {len(rows)} rows, expected 1"

        # First writer wins -- the surviving row keeps the FIRST payload's fields.
        assert getattr(rows[0], "outcome", "<no-such-column>") == "ok"
        assert getattr(rows[0], "finish_reason", "<no-such-column>") == "tool_calls"
        assert getattr(rows[0], "tool_calls", "<no-such-column>") == tool_calls

        # The stored hash is exactly the four-input hash, unchanged.
        assert rows[0].content_hash == _compute_content_hash(
            first_payload.schema_id,
            first_payload.job_id,
            first_payload.messages,
            first_payload.response_content,
        )

    def test_compute_content_hash_signature_is_pinned(self) -> None:
        """Structural guard: the hash takes exactly these four inputs.

        A new parameter here silently re-buckets every future sample against
        every historical one. If you are changing this, you are changing
        training volume -- do it deliberately, not as a side effect."""
        params = list(inspect.signature(_compute_content_hash).parameters)
        assert params == ["schema_id", "job_id", "messages", "response_content"]


# ---------------------------------------------------------------------------
# content_hash UNIQUE index (DB-enforced dedup) + migration 005
# ---------------------------------------------------------------------------


class TestContentHashUniqueIndex:
    """The unique index on content_hash makes dedup a DB invariant, not just an
    app-side best-effort — and activates the IntegrityError fallback."""

    async def test_content_hash_unique_index_enforced(self, session: AsyncSession) -> None:
        """Inserting a second row with an identical content_hash must raise IntegrityError."""
        payload = _make_payload(response_content='{"unique_idx": "row-one"}')
        first = await sample_service.create_sample(session, payload)
        shared_hash = _compute_content_hash(
            payload.schema_id, payload.job_id, payload.messages, payload.response_content
        )

        # A completely different row that merely collides on content_hash must
        # still be rejected by the DB (global uniqueness while tenant-less).
        clash = InferenceSampleDB(
            id=str(uuid4()),
            schema_id="ks_Other",
            job_id="job-other",
            inference_id=f"inf-{uuid4().hex[:12]}",
            messages=[{"role": "user", "content": "totally different"}],
            response_content="different response body",
            provider="claude",
            validation_passed=True,
            fallback_used=False,
            fallback_reason=None,
            latency_ms=5,
            failover_chain_id="chain-x",
            used_in_training=False,
            training_batch_id=None,
            content_hash=shared_hash,
        )
        session.add(clash)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # The original row is still the only one with that hash.
        rows = (
            (
                await session.execute(
                    select(InferenceSampleDB).where(InferenceSampleDB.content_hash == shared_hash)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert str(rows[0].id) == str(first.id)

    async def test_create_sample_concurrent_duplicate_returns_single_row(
        self, engine: AsyncEngine
    ) -> None:
        """Two concurrent create_sample calls with identical payloads must converge
        on one row via the IntegrityError fallback (sample_service.py:241-252)."""
        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        payload = _make_payload(response_content='{"concurrent": "dup-race"}')
        shared_hash = _compute_content_hash(
            payload.schema_id, payload.job_id, payload.messages, payload.response_content
        )

        async with factory() as s1, factory() as s2:
            r1, r2 = await asyncio.gather(
                sample_service.create_sample(s1, payload),
                sample_service.create_sample(s2, payload),
            )

        assert r1.id == r2.id

        async with factory() as s3:
            rows = (
                (
                    await s3.execute(
                        select(InferenceSampleDB).where(
                            InferenceSampleDB.content_hash == shared_hash
                        )
                    )
                )
                .scalars()
                .all()
            )
        assert len(rows) == 1
        assert str(rows[0].id) == str(r1.id)

    async def test_integrity_error_fallback_returns_existing_row(
        self, session: AsyncSession
    ) -> None:
        """Deterministically drive the create_sample IntegrityError fallback (241-252):
        force the pre-check to miss (simulating a concurrent insert landing after it),
        so the row insert collides with the unique index and the fallback re-query
        returns the already-committed row instead of raising."""
        payload = _make_payload(response_content='{"fallback": "race-window"}')
        canonical = await sample_service.create_sample(session, payload)

        real_execute = session.execute
        calls = {"n": 0}

        async def _fake_execute(stmt: Any, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                # First execute in create_sample is the dedup pre-check. Return a
                # genuinely-empty result so the code proceeds to INSERT and hits
                # the live unique index; the fallback re-query (call 2) runs real.
                return await real_execute(
                    select(InferenceSampleDB).where(InferenceSampleDB.id == str(uuid4()))
                )
            return await real_execute(stmt, *args, **kwargs)

        with patch.object(session, "execute", side_effect=_fake_execute):
            result = await sample_service.create_sample(session, payload)

        assert result.id == canonical.id

        rows = (
            (
                await session.execute(
                    select(InferenceSampleDB).where(
                        InferenceSampleDB.content_hash
                        == _compute_content_hash(
                            payload.schema_id,
                            payload.job_id,
                            payload.messages,
                            payload.response_content,
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1

    async def test_migration_005_dedups_keeping_oldest(self, session: AsyncSession) -> None:
        """Migration 005's DELETE keeps the oldest row per content_hash, after which
        the UNIQUE index can be created."""
        # Drop the live unique index so duplicate content_hash rows can be inserted,
        # simulating the pre-migration (003 non-unique) state.
        await session.execute(text("DROP INDEX IF EXISTS ix_samples_tenant_content_hash_unique"))
        await session.commit()

        shared_hash = "d" * 64
        base = datetime.now(UTC)
        # t0 < t1 < t2 (oldest first); distinct ids.
        rows_spec = [
            (str(uuid4()), base - timedelta(minutes=2)),  # t0 -- should survive
            (str(uuid4()), base - timedelta(minutes=1)),  # t1
            (str(uuid4()), base),  # t2
        ]
        oldest_id = rows_spec[0][0]

        for i, (row_id, ts) in enumerate(rows_spec):
            await session.execute(
                text(
                    """
                    INSERT INTO inference_samples
                        (id, schema_id, job_id, inference_id, messages, response_content,
                         provider, validation_passed, fallback_used, fallback_reason,
                         latency_ms, failover_chain_id, used_in_training,
                         training_batch_id, content_hash, created_at)
                    VALUES
                        (:id, :schema_id, :job_id, :inference_id, :messages, :response_content,
                         :provider, :validation_passed, :fallback_used, :fallback_reason,
                         :latency_ms, :failover_chain_id, :used_in_training,
                         :training_batch_id, :content_hash, :created_at)
                    """
                ),
                {
                    "id": row_id,
                    "schema_id": "ks_Mig005",
                    "job_id": "job-mig",
                    "inference_id": f"inf-mig-{i}",
                    "messages": json.dumps([{"role": "user", "content": "x"}]),
                    "response_content": "resp",
                    "provider": "Qwen3-4B-AWQ",
                    "validation_passed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "latency_ms": 10,
                    "failover_chain_id": "chain-mig",
                    "used_in_training": False,
                    "training_batch_id": None,
                    "content_hash": shared_hash,
                    "created_at": ts,
                },
            )
        await session.commit()

        # Run migration 005's exact keep-oldest DELETE.
        await session.execute(
            text(
                """
                DELETE FROM inference_samples
                WHERE id NOT IN (
                    SELECT DISTINCT ON (content_hash) id
                    FROM inference_samples
                    WHERE content_hash IS NOT NULL
                    ORDER BY content_hash, created_at ASC, id ASC
                )
                AND content_hash IS NOT NULL
                """
            )
        )
        await session.commit()

        surviving = (
            (
                await session.execute(
                    select(InferenceSampleDB).where(InferenceSampleDB.content_hash == shared_hash)
                )
            )
            .scalars()
            .all()
        )
        assert len(surviving) == 1
        assert str(surviving[0].id) == oldest_id

        # With duplicates collapsed, the unique index can now be created.
        await session.execute(
            text(
                "CREATE UNIQUE INDEX ix_samples_content_hash_unique "
                "ON inference_samples (content_hash)"
            )
        )
        await session.commit()


# ---------------------------------------------------------------------------
# list_samples date/filter tests
# ---------------------------------------------------------------------------


class TestListSamplesDateFilters:
    async def test_filter_by_start_date(self, session: AsyncSession) -> None:
        """list_samples with start_date should only return samples created on or after that date."""
        payload = _make_payload()
        created = await sample_service.create_sample(session, payload)

        # Future start_date -- no results
        future_date = datetime.now(UTC) + timedelta(days=1)
        filters = SampleFilter(start_date=future_date)
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 0

        # Past start_date -- should return the sample
        past_date = datetime.now(UTC) - timedelta(days=1)
        filters = SampleFilter(start_date=past_date)
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == created.id

    async def test_filter_by_end_date(self, session: AsyncSession) -> None:
        """list_samples with end_date should only return samples created on or before that date."""
        payload = _make_payload()
        created = await sample_service.create_sample(session, payload)

        # Past end_date -- no results
        past_date = datetime.now(UTC) - timedelta(days=1)
        filters = SampleFilter(end_date=past_date)
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 0

        # Future end_date -- should return the sample
        future_date = datetime.now(UTC) + timedelta(days=1)
        filters = SampleFilter(end_date=future_date)
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == created.id

    async def test_filter_by_start_and_end_date(self, session: AsyncSession) -> None:
        """list_samples with both start and end date range."""
        payload = _make_payload()
        created = await sample_service.create_sample(session, payload)

        past = datetime.now(UTC) - timedelta(days=1)
        future = datetime.now(UTC) + timedelta(days=1)
        filters = SampleFilter(start_date=past, end_date=future)
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == created.id

    async def test_filter_by_used_in_training(self, session: AsyncSession) -> None:
        """list_samples with used_in_training filter."""
        payload = _make_payload()
        created = await sample_service.create_sample(session, payload)

        filters = SampleFilter(used_in_training=False)
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == created.id
        assert results[0].used_in_training is False

        filters = SampleFilter(used_in_training=True)
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 0

    async def test_list_samples_ordering(self, session: AsyncSession) -> None:
        """list_samples should return results ordered by created_at desc."""
        s1 = await sample_service.create_sample(session, _make_payload())
        s2 = await sample_service.create_sample(session, _make_payload())
        s3 = await sample_service.create_sample(session, _make_payload())

        results = await sample_service.list_samples(session, SampleFilter())
        assert len(results) == 3
        # Most recently created first
        assert results[0].id == s3.id
        assert results[1].id == s2.id
        assert results[2].id == s1.id

    async def test_list_samples_pagination(self, session: AsyncSession) -> None:
        """list_samples should correctly paginate with limit and offset."""
        for _ in range(5):
            await sample_service.create_sample(session, _make_payload())

        page1 = await sample_service.list_samples(session, SampleFilter(limit=2, offset=0))
        assert len(page1) == 2

        page2 = await sample_service.list_samples(session, SampleFilter(limit=2, offset=2))
        assert len(page2) == 2

        page3 = await sample_service.list_samples(session, SampleFilter(limit=2, offset=4))
        assert len(page3) == 1

        # No overlap between pages
        all_ids = {s.id for s in page1} | {s.id for s in page2} | {s.id for s in page3}
        assert len(all_ids) == 5


# ---------------------------------------------------------------------------
# get_count
# ---------------------------------------------------------------------------


class TestGetCount:
    async def test_get_count_basic(self, session: AsyncSession) -> None:
        """get_count should return correct count and schema_id."""
        await sample_service.create_sample(session, _make_payload())
        await sample_service.create_sample(session, _make_payload())

        result = await sample_service.get_count(session, "ks_TestAgent")
        assert result.count == 2
        assert result.schema_id == "ks_TestAgent"

    async def test_get_count_only_validated(self, session: AsyncSession) -> None:
        """get_count with only_validated should exclude failed samples."""
        await sample_service.create_sample(session, _make_payload(validation_passed=True))
        await sample_service.create_sample(session, _make_payload(validation_passed=False))

        result = await sample_service.get_count(session, "ks_TestAgent", only_validated=True)
        assert result.count == 1

        result_all = await sample_service.get_count(session, "ks_TestAgent", only_validated=False)
        assert result_all.count == 2

    async def test_get_count_only_unused(self, session: AsyncSession) -> None:
        """get_count with only_unused should exclude used samples."""
        sample = await sample_service.create_sample(session, _make_payload())
        # Mark one as used
        mark_req = MarkUsedRequest(
            sample_ids=[sample.id],
            training_batch_id="batch-test",
        )
        await sample_service.mark_used(session, mark_req)
        # Create another (unused)
        await sample_service.create_sample(session, _make_payload())

        result = await sample_service.get_count(session, "ks_TestAgent", only_unused=True)
        assert result.count == 1

        result_all = await sample_service.get_count(session, "ks_TestAgent", only_unused=False)
        assert result_all.count == 2

    async def test_get_count_different_schemas_isolated(self, session: AsyncSession) -> None:
        """get_count should only count samples for the specified schema."""
        await sample_service.create_sample(session, _make_payload(schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_B"))

        result_a = await sample_service.get_count(session, "ks_A")
        assert result_a.count == 2
        result_b = await sample_service.get_count(session, "ks_B")
        assert result_b.count == 1

    async def test_get_count_aggregates_when_schema_id_none(self, session: AsyncSession) -> None:
        """get_count without schema_id aggregates across all schemas (flywheel global trigger)."""
        await sample_service.create_sample(session, _make_payload(schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_B"))

        result = await sample_service.get_count(session)
        assert result.count == 3
        assert result.schema_id is None


# ---------------------------------------------------------------------------
# list_schema_ids
# ---------------------------------------------------------------------------


class TestListSchemaIds:
    async def test_list_schema_ids_empty(self, session: AsyncSession) -> None:
        """Should return empty list for empty database."""
        result = await sample_service.list_schema_ids(session)
        assert result == []

    async def test_list_schema_ids_deduplicated_and_sorted(self, session: AsyncSession) -> None:
        """Should return unique, sorted schema_ids."""
        await sample_service.create_sample(session, _make_payload(schema_id="ks_C"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_B"))

        result = await sample_service.list_schema_ids(session)
        assert result == ["ks_A", "ks_B", "ks_C"]


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    async def test_get_stats_with_provider_breakdown(self, session: AsyncSession) -> None:
        """get_stats should include accurate provider breakdown."""
        await sample_service.create_sample(session, _make_payload(provider="Qwen3-4B-AWQ"))
        await sample_service.create_sample(session, _make_payload(provider="claude"))
        await sample_service.create_sample(session, _make_payload(provider="claude"))

        stats = await sample_service.get_stats(session, schema_id="ks_TestAgent")
        assert len(stats) == 1
        s = stats[0]
        assert s.total_samples == 3
        assert s.provider_breakdown["Qwen3-4B-AWQ"] == 1
        assert s.provider_breakdown["claude"] == 2
        assert sum(s.provider_breakdown.values()) == s.total_samples

    async def test_get_stats_all_schemas(self, session: AsyncSession) -> None:
        """get_stats without schema_id should return stats for all schemas."""
        await sample_service.create_sample(session, _make_payload(schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_B"))

        stats = await sample_service.get_stats(session)
        assert len(stats) == 2
        schema_ids = {s.schema_id for s in stats}
        assert schema_ids == {"ks_A", "ks_B"}

    async def test_get_stats_with_mixed_validation(self, session: AsyncSession) -> None:
        """get_stats should correctly count validated/failed/available."""
        await sample_service.create_sample(session, _make_payload(validation_passed=True))
        await sample_service.create_sample(session, _make_payload(validation_passed=False))
        sample = await sample_service.create_sample(session, _make_payload(validation_passed=True))
        # Mark one validated sample as used
        await sample_service.mark_used(
            session,
            MarkUsedRequest(sample_ids=[sample.id], training_batch_id="batch-x"),
        )

        stats = await sample_service.get_stats(session, schema_id="ks_TestAgent")
        assert len(stats) == 1
        s = stats[0]
        assert s.total_samples == 3
        assert s.validated_samples == 2
        assert s.failed_samples == 1
        assert s.used_in_training == 1
        assert s.available_for_training == 1
        # Invariant: validated = used_in_training + available_for_training
        assert s.validated_samples == s.used_in_training + s.available_for_training

    async def test_get_stats_fallback_rate(self, session: AsyncSession) -> None:
        """get_stats should compute fallback_rate as fraction of fallback_used=True."""
        await sample_service.create_sample(session, _make_payload(fallback_used=False))
        await sample_service.create_sample(session, _make_payload(fallback_used=True))

        stats = await sample_service.get_stats(session, schema_id="ks_TestAgent")
        assert len(stats) == 1
        assert stats[0].fallback_rate == pytest.approx(0.5)

    async def test_get_stats_avg_latency(self, session: AsyncSession) -> None:
        """get_stats should compute correct average latency."""
        await sample_service.create_sample(session, _make_payload(latency_ms=100))
        await sample_service.create_sample(session, _make_payload(latency_ms=200))
        await sample_service.create_sample(session, _make_payload(latency_ms=300))

        stats = await sample_service.get_stats(session, schema_id="ks_TestAgent")
        assert len(stats) == 1
        assert stats[0].avg_latency_ms == pytest.approx(200.0)

    async def test_get_stats_empty(self, session: AsyncSession) -> None:
        """get_stats should return empty list for empty database."""
        stats = await sample_service.get_stats(session)
        assert stats == []


# ---------------------------------------------------------------------------
# mark_used
# ---------------------------------------------------------------------------


class TestMarkUsed:
    async def test_mark_used_returns_updated_count(self, session: AsyncSession) -> None:
        """mark_used should return the number of updated rows."""
        s1 = await sample_service.create_sample(session, _make_payload())
        s2 = await sample_service.create_sample(session, _make_payload())

        req = MarkUsedRequest(
            sample_ids=[s1.id, s2.id],
            training_batch_id="batch-123",
        )
        updated = await sample_service.mark_used(session, req)
        assert updated == 2

    async def test_mark_used_sets_training_fields(self, session: AsyncSession) -> None:
        """mark_used should set used_in_training=True and training_batch_id on the rows."""
        sample = await sample_service.create_sample(session, _make_payload())
        assert sample.used_in_training is False

        req = MarkUsedRequest(
            sample_ids=[sample.id],
            training_batch_id="batch-verify",
        )
        await sample_service.mark_used(session, req)

        # Verify in DB
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(sample.id))
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.used_in_training is True
        assert row.training_batch_id == "batch-verify"

    async def test_mark_used_nonexistent_ids(self, session: AsyncSession) -> None:
        """mark_used with non-existent IDs should return 0."""
        req = MarkUsedRequest(
            sample_ids=[uuid4()],
            training_batch_id="batch-nope",
        )
        updated = await sample_service.mark_used(session, req)
        assert updated == 0

    async def test_mark_used_idempotent(self, session: AsyncSession) -> None:
        """Marking the same sample used twice should still update (overwrite same values)."""
        sample = await sample_service.create_sample(session, _make_payload())
        req = MarkUsedRequest(
            sample_ids=[sample.id],
            training_batch_id="batch-first",
        )
        first_count = await sample_service.mark_used(session, req)
        assert first_count == 1

        req2 = MarkUsedRequest(
            sample_ids=[sample.id],
            training_batch_id="batch-second",
        )
        second_count = await sample_service.mark_used(session, req2)
        assert second_count == 1

        # Verify the training_batch_id was overwritten
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(sample.id))
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.training_batch_id == "batch-second"


# ---------------------------------------------------------------------------
# export_samples_jsonl
# ---------------------------------------------------------------------------


class TestExportSamplesJsonl:
    async def test_export_basic(self, session: AsyncSession) -> None:
        """export_samples_jsonl should yield valid JSONL lines with correct fields."""
        await sample_service.create_sample(session, _make_payload())
        await sample_service.create_sample(session, _make_payload())

        params = ExportParams(schema_id="ks_TestAgent")
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(line)

        assert len(lines) == 2
        for line in lines:
            assert line.endswith("\n")
            parsed = json.loads(line)
            assert parsed["schema_id"] == "ks_TestAgent"
            assert "id" in parsed
            assert "messages" in parsed
            assert "response_content" in parsed
            assert parsed["validation_passed"] is True
            assert parsed["used_in_training"] is False

    async def test_export_with_provider_filter(self, session: AsyncSession) -> None:
        """export_samples_jsonl should filter by provider."""
        await sample_service.create_sample(session, _make_payload(provider="Qwen3-4B-AWQ"))
        await sample_service.create_sample(session, _make_payload(provider="claude"))

        params = ExportParams(schema_id="ks_TestAgent", provider="claude")
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(line)
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["provider"] == "claude"

    async def test_export_respects_limit(self, session: AsyncSession) -> None:
        """export_samples_jsonl with limit should cap output."""
        for _ in range(5):
            await sample_service.create_sample(session, _make_payload())

        params = ExportParams(schema_id="ks_TestAgent", limit=2)
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(line)
        assert len(lines) == 2

    async def test_export_ordering(self, session: AsyncSession) -> None:
        """export_samples_jsonl should order by created_at asc (oldest first)."""
        s1 = await sample_service.create_sample(session, _make_payload())
        s2 = await sample_service.create_sample(session, _make_payload())

        params = ExportParams(schema_id="ks_TestAgent")
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(json.loads(line))

        assert len(lines) == 2
        # First line should be oldest (s1)
        assert lines[0]["id"] == str(s1.id)
        assert lines[1]["id"] == str(s2.id)


# ---------------------------------------------------------------------------
# _row_to_domain
# ---------------------------------------------------------------------------


class TestRowToDomain:
    """Tests for _row_to_domain using real DB rows instead of mocks."""

    async def test_row_to_domain_with_real_db_row(self, session: AsyncSession) -> None:
        """_row_to_domain should correctly convert a real DB row to domain model."""
        payload = _make_payload(
            fallback_used=True,
            fallback_reason="test reason",
        )
        created = await sample_service.create_sample(session, payload)

        # Fetch the raw DB row
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(created.id))
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()

        domain = _row_to_domain(row)
        assert isinstance(domain.id, UUID)
        assert domain.id == created.id
        assert domain.schema_id == payload.schema_id
        assert domain.job_id == payload.job_id
        assert domain.provider == payload.provider
        assert domain.fallback_used is True
        assert domain.fallback_reason == "test reason"
        assert isinstance(domain.created_at, datetime)


# ---------------------------------------------------------------------------
# _compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    def test_same_input_same_hash(self) -> None:
        """Identical inputs should produce identical hashes."""
        h1 = _compute_content_hash(
            "schema", "job-1", [{"role": "user", "content": "hi"}], "response"
        )
        h2 = _compute_content_hash(
            "schema", "job-1", [{"role": "user", "content": "hi"}], "response"
        )
        assert h1 == h2

    def test_different_schema_different_hash(self) -> None:
        """Different schema_id should produce different hash."""
        h1 = _compute_content_hash(
            "schema_a", "job-1", [{"role": "user", "content": "hi"}], "response"
        )
        h2 = _compute_content_hash(
            "schema_b", "job-1", [{"role": "user", "content": "hi"}], "response"
        )
        assert h1 != h2

    def test_different_job_id_different_hash(self) -> None:
        """Different job_id should produce different hash."""
        h1 = _compute_content_hash(
            "schema", "job-1", [{"role": "user", "content": "hi"}], "response"
        )
        h2 = _compute_content_hash(
            "schema", "job-2", [{"role": "user", "content": "hi"}], "response"
        )
        assert h1 != h2

    def test_different_messages_different_hash(self) -> None:
        """Different messages should produce different hash."""
        h1 = _compute_content_hash(
            "schema", "job-1", [{"role": "user", "content": "hi"}], "response"
        )
        h2 = _compute_content_hash(
            "schema", "job-1", [{"role": "user", "content": "bye"}], "response"
        )
        assert h1 != h2

    def test_different_response_different_hash(self) -> None:
        """Different response_content should produce different hash."""
        h1 = _compute_content_hash(
            "schema", "job-1", [{"role": "user", "content": "hi"}], "response_a"
        )
        h2 = _compute_content_hash(
            "schema", "job-1", [{"role": "user", "content": "hi"}], "response_b"
        )
        assert h1 != h2

    def test_hash_is_hex_sha256(self) -> None:
        """Hash should be a 64-character hex string (SHA-256)."""
        h = _compute_content_hash("s", "j", [{"role": "user", "content": "c"}], "r")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_message_key_order_irrelevant(self) -> None:
        """sort_keys=True means dict key order in messages doesn't affect hash."""
        h1 = _compute_content_hash("s", "j", [{"content": "hi", "role": "user"}], "r")
        h2 = _compute_content_hash("s", "j", [{"role": "user", "content": "hi"}], "r")
        assert h1 == h2


# ---------------------------------------------------------------------------
# purge_samples
# ---------------------------------------------------------------------------


class TestPurgeSamples:
    async def test_purge_by_schema_id(self, session: AsyncSession) -> None:
        """purge_samples should delete matching samples by schema_id."""
        await sample_service.create_sample(session, _make_payload(schema_id="ks_Purge"))
        await sample_service.create_sample(session, _make_payload(schema_id="ks_Keep"))

        result = await sample_service.purge_samples(session, schema_id="ks_Purge")
        assert result.purged_count == 1
        assert result.filters_applied["schema_id"] == "ks_Purge"

        # Verify only ks_Keep remains
        remaining = await sample_service.list_samples(session, SampleFilter())
        assert len(remaining) == 1
        assert remaining[0].schema_id == "ks_Keep"

    async def test_purge_by_provider(self, session: AsyncSession) -> None:
        """purge_samples should delete matching samples by provider."""
        await sample_service.create_sample(session, _make_payload(provider="claude"))
        await sample_service.create_sample(session, _make_payload(provider="Qwen3-4B-AWQ"))

        result = await sample_service.purge_samples(session, provider="claude")
        assert result.purged_count == 1
        assert result.filters_applied["provider"] == "claude"

        remaining = await sample_service.list_samples(session, SampleFilter())
        assert len(remaining) == 1
        assert remaining[0].provider == "Qwen3-4B-AWQ"

    async def test_purge_by_validation_passed(self, session: AsyncSession) -> None:
        """purge_samples should delete matching samples by validation_passed."""
        await sample_service.create_sample(session, _make_payload(validation_passed=True))
        await sample_service.create_sample(session, _make_payload(validation_passed=False))

        result = await sample_service.purge_samples(session, validation_passed=False)
        assert result.purged_count == 1
        assert result.filters_applied["validation_passed"] == "False"

        remaining = await sample_service.list_samples(session, SampleFilter())
        assert len(remaining) == 1
        assert remaining[0].validation_passed is True

    async def test_purge_by_before_date(self, session: AsyncSession) -> None:
        """purge_samples should delete samples before the given date."""
        await sample_service.create_sample(session, _make_payload())

        future = datetime.now(UTC) + timedelta(days=1)
        result = await sample_service.purge_samples(session, before_date=future)
        assert result.purged_count == 1
        assert "before_date" in result.filters_applied

        remaining = await sample_service.list_samples(session, SampleFilter())
        assert len(remaining) == 0

    async def test_purge_no_matches(self, session: AsyncSession) -> None:
        """purge_samples with no matches should return 0 and leave DB unchanged."""
        await sample_service.create_sample(session, _make_payload())

        result = await sample_service.purge_samples(session, schema_id="ks_NonExistent")
        assert result.purged_count == 0

        remaining = await sample_service.list_samples(session, SampleFilter())
        assert len(remaining) == 1

    async def test_purge_multiple_filters(self, session: AsyncSession) -> None:
        """purge_samples with multiple filters combined (AND logic)."""
        await sample_service.create_sample(
            session, _make_payload(schema_id="ks_Multi", provider="claude")
        )
        await sample_service.create_sample(
            session, _make_payload(schema_id="ks_Multi", provider="Qwen3-4B-AWQ")
        )

        result = await sample_service.purge_samples(
            session, schema_id="ks_Multi", provider="claude"
        )
        assert result.purged_count == 1
        assert result.filters_applied["schema_id"] == "ks_Multi"
        assert result.filters_applied["provider"] == "claude"

        remaining = await sample_service.list_samples(session, SampleFilter())
        assert len(remaining) == 1
        assert remaining[0].provider == "Qwen3-4B-AWQ"


# ---------------------------------------------------------------------------
# Training threshold notification
# ---------------------------------------------------------------------------

# Threshold used by all tests in this section -- kept low so we don't need
# to create dozens of samples per test.
_TEST_THRESHOLD = 3


def _threshold_settings(**overrides: Any) -> Any:
    """Return a mock Settings with training trigger fields."""
    from unittest.mock import MagicMock

    s = MagicMock()
    s.training_trigger_enabled = overrides.get("enabled", True)
    s.training_threshold = overrides.get("threshold", _TEST_THRESHOLD)
    s.training_trigger_url = overrides.get("url", "http://tdg.test.local:8152/api/triggers/fire")
    # Explicitly disable the global trigger by default so per-schema threshold
    # tests never accidentally exercise the aggregate path.
    s.global_training_trigger_enabled = overrides.get("global_enabled", False)
    s.global_training_threshold = overrides.get("global_threshold", 10**9)
    s.global_training_trigger_url = overrides.get(
        "global_url", "http://tdg.test.local:8152/api/triggers/fire-global"
    )
    return s


class TestTrainingThreshold:
    """Tests for event-driven training threshold notification."""

    async def test_threshold_fires_when_count_equals_threshold(self, session: AsyncSession) -> None:
        """Webhook should fire when available_for_training exactly equals threshold."""
        schema = f"ks_ThresholdFire_{uuid4().hex[:8]}"

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=_TEST_THRESHOLD),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            # Create threshold-1 samples -- no trigger yet
            for _ in range(_TEST_THRESHOLD - 1):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
            mock_fire.assert_not_called()

            # The threshold-th sample should trigger
            await sample_service.create_sample(
                session, _make_payload(schema_id=schema, validation_passed=True)
            )
            # Yield to event loop so the fire-and-forget task can execute
            await asyncio.sleep(0)
            mock_fire.assert_called_once_with(
                "http://tdg.test.local:8152/api/triggers/fire",
                schema,
                _TEST_THRESHOLD,
            )

    async def test_threshold_does_not_fire_below_threshold(self, session: AsyncSession) -> None:
        """Webhook should NOT fire when count is below threshold."""
        schema = f"ks_BelowThreshold_{uuid4().hex[:8]}"

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=_TEST_THRESHOLD),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            for _ in range(_TEST_THRESHOLD - 1):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
            mock_fire.assert_not_called()

    async def test_threshold_does_not_fire_for_deduplicated(self, session: AsyncSession) -> None:
        """Webhook should NOT fire for deduplicated samples (no new row created)."""
        schema = f"ks_DedupNoFire_{uuid4().hex[:8]}"
        fixed_content = f'{{"dedup_threshold": "{uuid4().hex}"}}'

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=2),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            # Create 1 unique sample
            await sample_service.create_sample(
                session,
                _make_payload(
                    schema_id=schema, validation_passed=True, response_content=fixed_content
                ),
            )
            # Submit the same content again -- dedup returns existing, no threshold check
            await sample_service.create_sample(
                session,
                _make_payload(
                    schema_id=schema, validation_passed=True, response_content=fixed_content
                ),
            )
            mock_fire.assert_not_called()

    async def test_threshold_does_not_fire_when_validation_failed(
        self, session: AsyncSession
    ) -> None:
        """Webhook should NOT fire when validation_passed is False."""
        schema = f"ks_FailedNoFire_{uuid4().hex[:8]}"

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=1),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            await sample_service.create_sample(
                session, _make_payload(schema_id=schema, validation_passed=False)
            )
            mock_fire.assert_not_called()

    async def test_webhook_failure_does_not_crash_sample_creation(
        self, session: AsyncSession
    ) -> None:
        """A failing webhook should not prevent sample creation from succeeding."""
        schema = f"ks_WebhookFail_{uuid4().hex[:8]}"

        mock_post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=1),
            ),
            patch("services.sample_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.post = mock_post
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client_instance

            result = await sample_service.create_sample(
                session, _make_payload(schema_id=schema, validation_passed=True)
            )
            # Let fire-and-forget task run
            await asyncio.sleep(0)

            assert result.schema_id == schema
            assert result.validation_passed is True

            # Verify sample was persisted despite webhook failure
            stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(result.id))
            db_result = await session.execute(stmt)
            row = db_result.scalar_one()
            assert row.schema_id == schema

    async def test_threshold_does_not_fire_above_threshold(self, session: AsyncSession) -> None:
        """Webhook should fire exactly once at the threshold, not for samples above it."""
        schema = f"ks_AboveThreshold_{uuid4().hex[:8]}"

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=_TEST_THRESHOLD),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            for _ in range(_TEST_THRESHOLD + 2):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
            # Yield to event loop so fire-and-forget tasks can execute
            await asyncio.sleep(0)
            # Should have fired exactly once (when count == threshold)
            assert mock_fire.call_count == 1


# ---------------------------------------------------------------------------
# Global (system-wide single-LoRA) training threshold notification
# ---------------------------------------------------------------------------

_GLOBAL_TEST_THRESHOLD = 4


class TestGlobalTrainingThreshold:
    """Tests for the event-driven AGGREGATE training threshold notification.

    Mirrors :class:`TestTrainingThreshold` but for the system-wide single-LoRA
    flywheel: trigger fires when the count of validated/unused samples ACROSS
    ALL schemas exactly equals ``global_training_threshold``.
    """

    async def test_global_threshold_fires_when_aggregate_equals_threshold(
        self, session: AsyncSession
    ) -> None:
        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(
                    enabled=False,  # mute the per-schema trigger; isolate global path
                    global_enabled=True,
                    global_threshold=_GLOBAL_TEST_THRESHOLD,
                ),
            ),
            patch(
                "services.sample_service._fire_global_training_trigger",
                new_callable=AsyncMock,
            ) as mock_fire_global,
        ):
            # Spread inserts across two schemas so we know aggregation is happening.
            schema_a = f"ks_GlobalA_{uuid4().hex[:8]}"
            schema_b = f"ks_GlobalB_{uuid4().hex[:8]}"
            for i in range(_GLOBAL_TEST_THRESHOLD - 1):
                schema = schema_a if i % 2 == 0 else schema_b
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
            mock_fire_global.assert_not_called()

            # The threshold-th sample (regardless of which schema it lands on)
            # crosses the aggregate threshold and should fire.
            await sample_service.create_sample(
                session, _make_payload(schema_id=schema_b, validation_passed=True)
            )
            await asyncio.sleep(0)
            mock_fire_global.assert_called_once_with(
                "http://tdg.test.local:8152/api/triggers/fire-global",
                _GLOBAL_TEST_THRESHOLD,
            )

    async def test_global_threshold_does_not_refire_above_threshold(
        self, session: AsyncSession
    ) -> None:
        """Once the aggregate count exceeds the threshold, subsequent inserts must NOT refire."""
        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(
                    enabled=False,
                    global_enabled=True,
                    global_threshold=_GLOBAL_TEST_THRESHOLD,
                ),
            ),
            patch(
                "services.sample_service._fire_global_training_trigger",
                new_callable=AsyncMock,
            ) as mock_fire_global,
        ):
            schema = f"ks_GlobalRefire_{uuid4().hex[:8]}"
            for _ in range(_GLOBAL_TEST_THRESHOLD + 3):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
                await asyncio.sleep(0)
            assert mock_fire_global.call_count == 1

    async def test_global_threshold_disabled_never_fires(self, session: AsyncSession) -> None:
        """When global_training_trigger_enabled=False, never fire — even at the count."""
        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(
                    enabled=False,
                    global_enabled=False,
                    global_threshold=_GLOBAL_TEST_THRESHOLD,
                ),
            ),
            patch(
                "services.sample_service._fire_global_training_trigger",
                new_callable=AsyncMock,
            ) as mock_fire_global,
        ):
            schema = f"ks_GlobalDisabled_{uuid4().hex[:8]}"
            for _ in range(_GLOBAL_TEST_THRESHOLD + 2):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
                await asyncio.sleep(0)
            mock_fire_global.assert_not_called()

    async def test_global_threshold_skipped_when_url_empty(self, session: AsyncSession) -> None:
        """Empty global_training_trigger_url no-ops even when enabled is True."""
        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(
                    enabled=False,
                    global_enabled=True,
                    global_threshold=_GLOBAL_TEST_THRESHOLD,
                    global_url="",
                ),
            ),
            patch(
                "services.sample_service._fire_global_training_trigger",
                new_callable=AsyncMock,
            ) as mock_fire_global,
        ):
            schema = f"ks_GlobalNoUrl_{uuid4().hex[:8]}"
            for _ in range(_GLOBAL_TEST_THRESHOLD):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
                await asyncio.sleep(0)
            mock_fire_global.assert_not_called()


# ---------------------------------------------------------------------------
# Rising-edge trigger: overshoot fires; re-arms after training consumes samples
# ---------------------------------------------------------------------------


async def _seed_validated(session: AsyncSession, schema_id: str, n: int) -> None:
    """Directly insert n validated/unused rows for schema_id and commit.

    Advances the DB validated/unused count WITHOUT going through create_sample,
    so the trigger_progress marker is NOT advanced — this reproduces the
    concurrent-overshoot condition (the count jumps past the threshold between
    marker observations) that the old exact-equality check silently missed.
    """
    for i in range(n):
        row_id = str(uuid4())
        content_hash = f"{uuid4().hex}{uuid4().hex}"  # 64 hex chars, globally unique
        await session.execute(
            text(
                """
                INSERT INTO inference_samples
                    (id, schema_id, job_id, inference_id, messages, response_content,
                     provider, validation_passed, fallback_used, fallback_reason,
                     latency_ms, failover_chain_id, used_in_training,
                     training_batch_id, content_hash)
                VALUES
                    (:id, :schema_id, :job_id, :inference_id, :messages, :response_content,
                     :provider, :validation_passed, :fallback_used, :fallback_reason,
                     :latency_ms, :failover_chain_id, :used_in_training,
                     :training_batch_id, :content_hash)
                """
            ),
            {
                "id": row_id,
                "schema_id": schema_id,
                "job_id": "job-seed",
                "inference_id": f"inf-seed-{i}-{uuid4().hex[:8]}",
                "messages": json.dumps([{"role": "user", "content": f"seed-{i}"}]),
                "response_content": f"seed-response-{uuid4().hex}",
                "provider": "Qwen3-4B-AWQ",
                "validation_passed": True,
                "fallback_used": False,
                "fallback_reason": None,
                "latency_ms": 10,
                "failover_chain_id": "chain-seed",
                "used_in_training": False,
                "training_batch_id": None,
                "content_hash": content_hash,
            },
        )
    await session.commit()


class TestRisingEdgeTrigger:
    """The trigger fires on the rising edge across the threshold — including a
    single-step overshoot the old ``available != threshold`` check missed — and
    re-arms once the observed count drops back below the threshold."""

    async def test_threshold_fires_on_overshoot(self, session: AsyncSession) -> None:
        """A count that jumps from below to above the threshold in one insert fires."""
        schema = f"ks_Overshoot_{uuid4().hex[:8]}"

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=_TEST_THRESHOLD),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            # Seed threshold+1 rows WITHOUT advancing the marker (marker stays 0),
            # so the next insert observes an overshoot: prev=0 < threshold <= count.
            await _seed_validated(session, schema, _TEST_THRESHOLD + 1)
            mock_fire.assert_not_called()

            await sample_service.create_sample(
                session, _make_payload(schema_id=schema, validation_passed=True)
            )
            await asyncio.sleep(0)
            # available is now threshold+2; old exact-equality check would NOT fire.
            mock_fire.assert_called_once_with(
                "http://tdg.test.local:8152/api/triggers/fire",
                schema,
                _TEST_THRESHOLD + 2,
            )

    async def test_global_threshold_fires_on_overshoot(self, session: AsyncSession) -> None:
        """The aggregate global trigger also fires on a single-step overshoot."""
        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(
                    enabled=False,  # mute per-schema path; isolate the global path
                    global_enabled=True,
                    global_threshold=_GLOBAL_TEST_THRESHOLD,
                ),
            ),
            patch(
                "services.sample_service._fire_global_training_trigger",
                new_callable=AsyncMock,
            ) as mock_fire_global,
        ):
            schema = f"ks_GlobalOvershoot_{uuid4().hex[:8]}"
            await _seed_validated(session, schema, _GLOBAL_TEST_THRESHOLD + 1)
            mock_fire_global.assert_not_called()

            await sample_service.create_sample(
                session, _make_payload(schema_id=schema, validation_passed=True)
            )
            await asyncio.sleep(0)
            mock_fire_global.assert_called_once_with(
                "http://tdg.test.local:8152/api/triggers/fire-global",
                _GLOBAL_TEST_THRESHOLD + 2,
            )

    async def test_threshold_refires_after_training_consumes(self, session: AsyncSession) -> None:
        """After training marks samples used (count drops below threshold), a fresh
        accumulation back up to the threshold fires a second time."""
        schema = f"ks_Refire_{uuid4().hex[:8]}"

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=_TEST_THRESHOLD),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            # First accumulation via create_sample: fires once at the crossing.
            for _ in range(_TEST_THRESHOLD):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
            await asyncio.sleep(0)
            assert mock_fire.call_count == 1

            # Training consumes ALL samples for this schema -> count drops to ~0.
            ids = (
                (
                    await session.execute(
                        select(InferenceSampleDB.id).where(InferenceSampleDB.schema_id == schema)
                    )
                )
                .scalars()
                .all()
            )
            await sample_service.mark_used(
                session,
                MarkUsedRequest(
                    sample_ids=[UUID(str(i)) for i in ids],
                    training_batch_id="batch-consume",
                ),
            )

            # Re-accumulate to the threshold: the marker re-arms (observes a count
            # strictly below the threshold first) and fires a SECOND time.
            for _ in range(_TEST_THRESHOLD):
                await sample_service.create_sample(
                    session, _make_payload(schema_id=schema, validation_passed=True)
                )
            await asyncio.sleep(0)
            assert mock_fire.call_count == 2


# ---------------------------------------------------------------------------
# Cross-job dedup: same content, different job_ids → both stored
# ---------------------------------------------------------------------------


class TestCrossJobDedup:
    async def test_same_content_different_job_ids_stored_separately(
        self, session: AsyncSession
    ) -> None:
        """Two samples with identical content but different job_ids should both be stored."""
        fixed_content = f'{{"cross_job": "{uuid4().hex}"}}'
        messages = [{"role": "user", "content": "same input"}]

        s1 = await sample_service.create_sample(
            session,
            _make_payload(
                job_id="job-A",
                messages=messages,
                response_content=fixed_content,
            ),
        )
        s2 = await sample_service.create_sample(
            session,
            _make_payload(
                job_id="job-B",
                messages=messages,
                response_content=fixed_content,
            ),
        )

        # They must be different samples
        assert s1.id != s2.id
        assert s1.job_id == "job-A"
        assert s2.job_id == "job-B"

        # Both persisted in DB
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.response_content == fixed_content)
        db_result = await session.execute(stmt)
        rows = db_result.scalars().all()
        assert len(rows) == 2

    async def test_same_content_same_job_still_deduplicates(self, session: AsyncSession) -> None:
        """Two samples with identical content AND job_id should still dedup."""
        fixed_content = f'{{"same_job_dedup": "{uuid4().hex}"}}'
        messages = [{"role": "user", "content": "same input"}]

        s1 = await sample_service.create_sample(
            session,
            _make_payload(
                job_id="job-shared",
                messages=messages,
                response_content=fixed_content,
            ),
        )
        s2 = await sample_service.create_sample(
            session,
            _make_payload(
                job_id="job-shared",
                messages=messages,
                response_content=fixed_content,
            ),
        )

        # Same ID returned (dedup)
        assert s1.id == s2.id

        # Only one row in DB
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.response_content == fixed_content)
        db_result = await session.execute(stmt)
        rows = db_result.scalars().all()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Query by job_id + schema_id
# ---------------------------------------------------------------------------


class TestQueryByJobAndSchema:
    async def test_filter_by_job_id_and_schema_id(self, session: AsyncSession) -> None:
        """Querying by job_id + schema_id should return only matching samples."""
        await sample_service.create_sample(
            session, _make_payload(job_id="job-X", schema_id="ks_Alpha")
        )
        target = await sample_service.create_sample(
            session, _make_payload(job_id="job-Y", schema_id="ks_Alpha")
        )
        await sample_service.create_sample(
            session, _make_payload(job_id="job-Y", schema_id="ks_Beta")
        )

        filters = SampleFilter(job_id="job-Y", schema_id="ks_Alpha")
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == target.id
        assert results[0].job_id == "job-Y"
        assert results[0].schema_id == "ks_Alpha"

    async def test_filter_by_job_id_only(self, session: AsyncSession) -> None:
        """Querying by job_id alone returns all samples for that job."""
        await sample_service.create_sample(session, _make_payload(job_id="job-Z", schema_id="ks_A"))
        await sample_service.create_sample(session, _make_payload(job_id="job-Z", schema_id="ks_B"))
        await sample_service.create_sample(session, _make_payload(job_id="job-other"))

        filters = SampleFilter(job_id="job-Z")
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 2
        assert all(r.job_id == "job-Z" for r in results)


# ---------------------------------------------------------------------------
# Query by inference_id
# ---------------------------------------------------------------------------


class TestQueryByInferenceId:
    async def test_filter_by_inference_id_exact_match(self, session: AsyncSession) -> None:
        """Querying by inference_id should return the exact sample."""
        target = await sample_service.create_sample(
            session, _make_payload(inference_id="inf-exact-match-001")
        )
        await sample_service.create_sample(session, _make_payload(inference_id="inf-other-002"))

        filters = SampleFilter(inference_id="inf-exact-match-001")
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == target.id
        assert results[0].inference_id == "inf-exact-match-001"

    async def test_filter_by_inference_id_no_match(self, session: AsyncSession) -> None:
        """Querying by non-existent inference_id returns empty."""
        await sample_service.create_sample(session, _make_payload())

        filters = SampleFilter(inference_id="inf-does-not-exist")
        results = await sample_service.list_samples(session, filters)
        assert results == []

    async def test_inference_id_persisted_in_db(self, session: AsyncSession) -> None:
        """inference_id should be persisted in the DB row."""
        result = await sample_service.create_sample(
            session, _make_payload(inference_id="inf-persist-check")
        )

        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(result.id))
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.inference_id == "inf-persist-check"


# ---------------------------------------------------------------------------
# Query by failover_chain_id
# ---------------------------------------------------------------------------


class TestQueryByFailoverChainId:
    async def test_filter_by_failover_chain_id(self, session: AsyncSession) -> None:
        """Querying by failover_chain_id returns matching samples."""
        target = await sample_service.create_sample(
            session, _make_payload(failover_chain_id="chain-specific-001")
        )
        await sample_service.create_sample(
            session, _make_payload(failover_chain_id="chain-other-002")
        )

        filters = SampleFilter(failover_chain_id="chain-specific-001")
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == target.id
        assert results[0].failover_chain_id == "chain-specific-001"


# ---------------------------------------------------------------------------
# Tri-state validation_passed (None = schema-not-applicable / validation skipped)
# ---------------------------------------------------------------------------


class TestValidationPassedTriState:
    async def test_create_sample_with_none_persists_as_null(self, session: AsyncSession) -> None:
        """validation_passed=None must round-trip through Pydantic -> DB -> domain as None."""
        payload = _make_payload(validation_passed=None)
        result = await sample_service.create_sample(session, payload)

        assert result.validation_passed is None

        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == str(result.id))
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.validation_passed is None

    async def test_none_excluded_from_only_validated_count(self, session: AsyncSession) -> None:
        """get_count(only_validated=True) must not count schema-skipped rows."""
        schema = f"ks_TriState_{uuid4().hex[:8]}"
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=True)
        )
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=False)
        )
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=None)
        )

        validated = await sample_service.get_count(session, schema, only_validated=True)
        assert validated.count == 1

        total = await sample_service.get_count(session, schema)
        assert total.count == 3

    async def test_none_excluded_from_stats_aggregates(self, session: AsyncSession) -> None:
        """validated_samples + failed_samples must NOT include schema-skipped rows."""
        schema = f"ks_TriStateStats_{uuid4().hex[:8]}"
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=True)
        )
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=False)
        )
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=None)
        )
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=None)
        )

        stats = await sample_service.get_stats(session, schema_id=schema)
        assert len(stats) == 1
        s = stats[0]
        assert s.total_samples == 4
        assert s.validated_samples == 1
        assert s.failed_samples == 1
        # The two None rows are neither validated nor failed.
        assert s.validated_samples + s.failed_samples == 2

    async def test_none_does_not_trigger_per_schema_webhook(self, session: AsyncSession) -> None:
        """Threshold webhook fires only on validation_passed=True inserts, not None."""
        schema = f"ks_NoneNoFire_{uuid4().hex[:8]}"

        with (
            patch(
                "services.sample_service.get_settings",
                return_value=_threshold_settings(threshold=1),
            ),
            patch(
                "services.sample_service._fire_training_trigger", new_callable=AsyncMock
            ) as mock_fire,
        ):
            await sample_service.create_sample(
                session, _make_payload(schema_id=schema, validation_passed=None)
            )
            await asyncio.sleep(0)
            mock_fire.assert_not_called()

    async def test_list_filter_validation_passed_true_excludes_none(
        self, session: AsyncSession
    ) -> None:
        """SampleFilter(validation_passed=True) must not return None rows."""
        schema = f"ks_FilterExclNone_{uuid4().hex[:8]}"
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=True)
        )
        await sample_service.create_sample(
            session, _make_payload(schema_id=schema, validation_passed=None)
        )

        results = await sample_service.list_samples(
            session, SampleFilter(schema_id=schema, validation_passed=True)
        )
        assert len(results) == 1
        assert results[0].validation_passed is True

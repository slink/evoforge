# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""SQLite-backed archive for individuals, fitness cache, prefix cache, and lineage."""

from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from evoforge.core.types import Fitness, Individual

# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class IndividualRow(Base):
    __tablename__ = "individuals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    genome: Mapped[str] = mapped_column(String)
    ir_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    ir_serialized: Mapped[str | None] = mapped_column(String, nullable=True)
    generation: Mapped[int] = mapped_column()
    fitness_primary: Mapped[float | None] = mapped_column(nullable=True)
    fitness_json: Mapped[str | None] = mapped_column(String, nullable=True)
    behavior_descriptor: Mapped[str | None] = mapped_column(String, nullable=True)
    mutation_source: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[float] = mapped_column()


class EvaluationRow(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ir_hash: Mapped[str] = mapped_column(String, index=True)
    backend_version: Mapped[str] = mapped_column(String)
    config_hash: Mapped[str] = mapped_column(String)
    fitness_primary: Mapped[float] = mapped_column()
    fitness_json: Mapped[str] = mapped_column(String)
    diagnostics_json: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[float] = mapped_column()

    __table_args__ = (UniqueConstraint("ir_hash", "backend_version", "config_hash"),)


class PrefixCacheRow(Base):
    __tablename__ = "prefix_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prefix_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    state_json: Mapped[str] = mapped_column(String)
    created_at: Mapped[float] = mapped_column()


class LineageRow(Base):
    __tablename__ = "lineage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_hash: Mapped[str] = mapped_column(String, index=True)
    child_hash: Mapped[str] = mapped_column(String, index=True)
    operator_name: Mapped[str] = mapped_column(String)
    generation: Mapped[int] = mapped_column()
    created_at: Mapped[float] = mapped_column()


class CheckpointRow(Base):
    __tablename__ = "checkpoints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    generation: Mapped[int] = mapped_column(index=True)
    state_json: Mapped[str] = mapped_column(String)
    created_at: Mapped[float] = mapped_column()


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def _fitness_to_json(fitness: Fitness) -> str:
    return json.dumps(
        {
            "primary": fitness.primary,
            "auxiliary": fitness.auxiliary,
            "constraints": fitness.constraints,
            "feasible": fitness.feasible,
        }
    )


def _json_to_fitness(raw: str) -> Fitness:
    d: dict[str, Any] = json.loads(raw)
    return Fitness(
        primary=d["primary"],
        auxiliary=d["auxiliary"],
        constraints=d["constraints"],
        feasible=d["feasible"],
    )


class Archive:
    """Async SQLite archive for evolutionary search state."""

    def __init__(self, url: str) -> None:
        self._engine = create_async_engine(url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def close(self) -> None:
        """Dispose the async engine and release all connections."""
        await self._engine.dispose()

    async def create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # ---- individuals -----------------------------------------------------

    async def store(self, individual: Individual) -> None:
        """Store an individual.  Skip silently if ir_hash already exists."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            async with session.begin():
                existing = (
                    await session.execute(
                        select(IndividualRow).where(IndividualRow.ir_hash == individual.ir_hash)
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    return

                ir_serialized: str | None = None
                if individual.ir is not None and hasattr(individual.ir, "serialize"):
                    ir_serialized = individual.ir.serialize()

                fitness_primary: float | None = None
                fitness_json: str | None = None
                if individual.fitness is not None:
                    fitness_primary = individual.fitness.primary
                    fitness_json = _fitness_to_json(individual.fitness)

                bd_json: str | None = None
                if individual.behavior_descriptor is not None:
                    bd_json = json.dumps(individual.behavior_descriptor)

                row = IndividualRow(
                    id=individual.id,
                    genome=individual.genome,
                    ir_hash=individual.ir_hash,
                    ir_serialized=ir_serialized,
                    generation=individual.generation,
                    fitness_primary=fitness_primary,
                    fitness_json=fitness_json,
                    behavior_descriptor=bd_json,
                    mutation_source=individual.mutation_source,
                    created_at=time.time(),
                )
                session.add(row)

    def _row_to_individual(self, row: IndividualRow) -> Individual:
        """Convert an IndividualRow to an Individual."""
        fitness: Fitness | None = None
        if row.fitness_json is not None:
            fitness = _json_to_fitness(row.fitness_json)

        bd: tuple[Any, ...] | None = None
        if row.behavior_descriptor is not None:
            bd = tuple(json.loads(row.behavior_descriptor))

        return Individual(
            genome=row.genome,
            ir=None,  # IR is not round-trippable without a backend
            ir_hash=row.ir_hash,
            generation=row.generation,
            fitness=fitness,
            id=row.id,
            behavior_descriptor=bd,
            mutation_source=row.mutation_source,
        )

    async def lookup(self, ir_hash: str) -> Individual | None:
        """Retrieve an individual by ir_hash, or None on miss."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(IndividualRow).where(IndividualRow.ir_hash == ir_hash)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._row_to_individual(row)

    async def lookup_many(self, ir_hashes: list[str]) -> dict[str, Individual]:
        """Retrieve multiple individuals by ir_hash in a single query."""
        from sqlalchemy import select

        if not ir_hashes:
            return {}

        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(IndividualRow).where(IndividualRow.ir_hash.in_(ir_hashes))
                    )
                )
                .scalars()
                .all()
            )
            return {row.ir_hash: self._row_to_individual(row) for row in rows}

    # ---- fitness cache ---------------------------------------------------

    async def store_fitness(
        self,
        ir_hash: str,
        backend_version: str,
        config_hash: str,
        fitness: Fitness,
        diagnostics_json: str,
    ) -> None:
        """Store a fitness evaluation result. Silently skips duplicates."""
        from sqlalchemy.exc import IntegrityError

        async with self._session_factory() as session:
            try:
                async with session.begin():
                    row = EvaluationRow(
                        ir_hash=ir_hash,
                        backend_version=backend_version,
                        config_hash=config_hash,
                        fitness_primary=fitness.primary,
                        fitness_json=_fitness_to_json(fitness),
                        diagnostics_json=diagnostics_json,
                        created_at=time.time(),
                    )
                    session.add(row)
            except IntegrityError:
                pass  # duplicate (ir_hash, backend_version, config_hash)

    async def lookup_fitness(
        self, ir_hash: str, backend_version: str, config_hash: str
    ) -> Fitness | None:
        """Look up a cached fitness by (ir_hash, backend_version, config_hash)."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(EvaluationRow).where(
                        EvaluationRow.ir_hash == ir_hash,
                        EvaluationRow.backend_version == backend_version,
                        EvaluationRow.config_hash == config_hash,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return _json_to_fitness(row.fitness_json)

    # ---- prefix cache ----------------------------------------------------

    async def get_prefix(self, prefix_hash: str) -> str | None:
        """Retrieve a cached prefix state, or None on miss."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(PrefixCacheRow).where(PrefixCacheRow.prefix_hash == prefix_hash)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return row.state_json

    async def put_prefix(self, prefix_hash: str, state_json: str) -> None:
        """Store a prefix cache entry."""
        async with self._session_factory() as session:
            async with session.begin():
                row = PrefixCacheRow(
                    prefix_hash=prefix_hash,
                    state_json=state_json,
                    created_at=time.time(),
                )
                session.add(row)

    # ---- lineage ---------------------------------------------------------

    async def store_lineage(
        self,
        parent_hash: str,
        child_hash: str,
        operator_name: str,
        generation: int,
    ) -> None:
        """Record a parent -> child lineage edge."""
        async with self._session_factory() as session:
            async with session.begin():
                row = LineageRow(
                    parent_hash=parent_hash,
                    child_hash=child_hash,
                    operator_name=operator_name,
                    generation=generation,
                    created_at=time.time(),
                )
                session.add(row)

    async def get_lineage(self, ir_hash: str) -> list[dict[str, Any]]:
        """Get all lineage records where child_hash matches."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            rows = (
                (await session.execute(select(LineageRow).where(LineageRow.child_hash == ir_hash)))
                .scalars()
                .all()
            )
            return [
                {
                    "parent_hash": row.parent_hash,
                    "child_hash": row.child_hash,
                    "operator_name": row.operator_name,
                    "generation": row.generation,
                }
                for row in rows
            ]

    # ---- checkpoints -------------------------------------------------

    async def store_checkpoint(
        self, generation: int, state: dict[str, Any], *, keep: int = 3
    ) -> None:
        """Store a checkpoint and prune all but the *keep* most recent."""
        from sqlalchemy import delete, select

        async with self._session_factory() as session:
            async with session.begin():
                row = CheckpointRow(
                    generation=generation,
                    state_json=json.dumps(state),
                    created_at=time.time(),
                )
                session.add(row)
                await session.flush()

                # Prune old checkpoints
                keep_ids = (
                    select(CheckpointRow.id)
                    .order_by(CheckpointRow.generation.desc())
                    .limit(keep)
                    .scalar_subquery()
                )
                await session.execute(
                    delete(CheckpointRow).where(CheckpointRow.id.not_in(keep_ids))
                )

    async def load_latest_checkpoint(self) -> dict[str, Any] | None:
        """Load the most recent checkpoint, or None if no checkpoints exist."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(CheckpointRow).order_by(CheckpointRow.generation.desc()).limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            result: dict[str, Any] = json.loads(row.state_json)
            return result

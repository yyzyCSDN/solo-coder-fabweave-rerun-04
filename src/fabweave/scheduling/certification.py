"""Certification scheduling for recipe/chamber qualifications.

A certification job must be placed so that, all at once:

* it never overlaps planned chamber maintenance;
* the chamber keeps its cooldown gap to every neighbouring activity;
* one qualified engineer covers the whole run from a single on-duty shift;
* the run finishes while the resulting certification still has usable life.

Scheduling is *atomic*: a batch is committed only when every job is placed.
If anything cannot be placed, the complete conflict list is returned and no
resource is held.  The solver is deterministic: identical inputs always yield
identical placements, so retries are stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from fabweave.core.timebox import TimeBox


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chamber:
    """A chamber that can host certification runs."""

    chamber_id: str
    maintenance: tuple[TimeBox, ...] = ()

    def __post_init__(self) -> None:
        if not self.chamber_id:
            raise ValueError("chamber id is required")
        object.__setattr__(self, "maintenance", tuple(self.maintenance))


@dataclass(frozen=True, slots=True)
class Engineer:
    """An engineer who can supervise certifications.

    ``chamber_ids`` is the set of chambers the engineer is trained on; an
    empty set means every chamber.  ``recipe_revisions`` is the set of recipe
    revisions the engineer may certify.
    """

    engineer_id: str
    shifts: tuple[TimeBox, ...]
    recipe_revisions: frozenset[int]
    chamber_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.engineer_id:
            raise ValueError("engineer id is required")
        if not self.shifts:
            raise ValueError("engineer needs at least one shift")
        if not self.recipe_revisions:
            raise ValueError("engineer needs at least one qualified recipe revision")
        object.__setattr__(self, "shifts", tuple(self.shifts))
        object.__setattr__(self, "recipe_revisions", frozenset(self.recipe_revisions))
        object.__setattr__(self, "chamber_ids", frozenset(self.chamber_ids))

    def can_run(self, chamber_id: str, recipe_revision: int) -> bool:
        return recipe_revision in self.recipe_revisions and (
            not self.chamber_ids or chamber_id in self.chamber_ids
        )


@dataclass(frozen=True, slots=True)
class CertificationJob:
    """A recipe certification run on one chamber.

    Times are integer ticks (the project-wide convention) and intervals are
    half-open: a run ``[start, end)`` may start exactly when maintenance ends
    or when a cooldown gap has elapsed.
    """

    job_id: str
    chamber_id: str
    recipe_revision: int
    duration: int
    earliest_start: int
    validity_expires_at: int
    required_valid_for: int = 0
    cooldown_after: int = 0

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job id is required")
        if self.recipe_revision <= 0:
            raise ValueError("recipe revision must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.required_valid_for < 0 or self.cooldown_after < 0:
            raise ValueError("cooldown and required validity must be non-negative")

    @property
    def latest_finish(self) -> int:
        """Last tick at which the run may finish and still be usable."""

        return self.validity_expires_at - self.required_valid_for


@dataclass(frozen=True, slots=True)
class ChamberActivity:
    """Pre-existing chamber occupancy (production run, prior qualification, ...)."""

    chamber_id: str
    window: TimeBox
    cooldown_after: int = 0
    reference: str = ""

    def __post_init__(self) -> None:
        if self.cooldown_after < 0:
            raise ValueError("cooldown must be non-negative")


@dataclass(frozen=True, slots=True)
class EngineerCommitment:
    """Pre-existing engineer occupancy that is not a chamber activity."""

    engineer_id: str
    window: TimeBox
    reference: str = ""


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Placement:
    job_id: str
    chamber_id: str
    engineer_id: str
    window: TimeBox
    cooldown_after: int = 0
    validity_expires_at: int = 0

    @property
    def certified_at(self) -> int:
        return self.window.end


@dataclass(frozen=True, slots=True)
class Conflict:
    """One reason a job cannot be placed.

    Resource windows point at the blocking object (maintenance window, shift,
    busy window).  ``related_job_id`` is set when the blocker is another
    certification job in the same request, which marks batch contention.
    """

    code: str
    job_id: str
    chamber_id: str | None = None
    engineer_id: str | None = None
    window: TimeBox | None = None
    related_job_id: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class SchedulingResult:
    request_id: str | None
    placed: tuple[Placement, ...] = ()
    conflicts: tuple[Conflict, ...] = ()

    @property
    def scheduled(self) -> bool:
        return not self.conflicts

    def conflicts_for(self, job_id: str) -> tuple[Conflict, ...]:
        return tuple(conflict for conflict in self.conflicts if conflict.job_id == job_id)


# ---------------------------------------------------------------------------
# Solver state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ChamberBusy:
    window: TimeBox
    cooldown_after: int
    reference: str


def _conflict_key(conflict: Conflict) -> tuple:
    window = (conflict.window.start, conflict.window.end) if conflict.window else ()
    return (
        conflict.job_id,
        conflict.code,
        conflict.chamber_id or "",
        conflict.engineer_id or "",
        conflict.related_job_id or "",
        tuple(window),
        conflict.note,
    )


class _State:
    """Mutable occupied-time view; the solver pushes and pops placements."""

    def __init__(
        self,
        chambers: Mapping[str, Chamber],
        engineers: Mapping[str, Engineer],
        activities: list[ChamberActivity],
        commitments: list[EngineerCommitment],
    ) -> None:
        self._chambers = chambers
        self._engineers = engineers
        self.chamber_busy: dict[str, list[_ChamberBusy]] = {
            chamber_id: [] for chamber_id in chambers
        }
        self.engineer_busy: dict[str, list[tuple[TimeBox, str]]] = {
            engineer_id: [] for engineer_id in engineers
        }
        for activity in activities:
            if activity.chamber_id not in chambers:
                raise ValueError(f"unknown chamber in activity: {activity.chamber_id}")
            self.chamber_busy[activity.chamber_id].append(
                _ChamberBusy(activity.window, activity.cooldown_after, activity.reference)
            )
        for commitment in commitments:
            if commitment.engineer_id not in engineers:
                raise ValueError(f"unknown engineer in commitment: {commitment.engineer_id}")
            self.engineer_busy[commitment.engineer_id].append(
                (commitment.window, commitment.reference)
            )

    def push(self, placement: Placement) -> None:
        self.chamber_busy[placement.chamber_id].append(
            _ChamberBusy(placement.window, placement.cooldown_after, placement.job_id)
        )
        self.engineer_busy[placement.engineer_id].append(
            (placement.window, placement.job_id)
        )

    def pop(self, placement: Placement) -> None:
        chamber_items = self.chamber_busy[placement.chamber_id]
        chamber_items.pop()
        engineer_items = self.engineer_busy[placement.engineer_id]
        engineer_items.pop()

    # -- feasibility -------------------------------------------------------

    def candidate_starts(self, job: CertificationJob, eligible: list[Engineer]) -> list[int]:
        """Deterministic candidate starts covering every possible optimum."""

        lo = job.earliest_start
        hi = job.latest_finish - job.duration
        if hi < lo:
            return []
        envelope = TimeBox(lo, job.latest_finish)
        starts = {lo}
        chamber = self._chambers[job.chamber_id]
        for window in chamber.maintenance:
            if window.overlaps(envelope) and lo <= window.end <= hi:
                starts.add(window.end)
        for busy in self.chamber_busy[job.chamber_id]:
            released_at = busy.window.end + busy.cooldown_after
            if busy.window.overlaps(envelope) and lo <= released_at <= hi:
                starts.add(released_at)
        for engineer in eligible:
            for shift in engineer.shifts:
                if shift.overlaps(envelope) and lo <= shift.start <= hi:
                    starts.add(shift.start)
            for window, _reference in self.engineer_busy[engineer.engineer_id]:
                if window.overlaps(envelope) and lo <= window.end <= hi:
                    starts.add(window.end)
        return sorted(starts)

    def candidate_conflicts(
        self, job: CertificationJob, engineers: list[Engineer], start: int
    ) -> list[Conflict]:
        """All constraint violations at one candidate start (empty => feasible)."""

        end = start + job.duration
        window = TimeBox(start, end)
        conflicts: list[Conflict] = []

        chamber = self._chambers[job.chamber_id]
        for maintenance in chamber.maintenance:
            if window.overlaps(maintenance):
                conflicts.append(
                    Conflict(
                        "maintenance_clash",
                        job.job_id,
                        chamber_id=job.chamber_id,
                        window=maintenance,
                        note="planned maintenance",
                    )
                )
        for busy in self.chamber_busy[job.chamber_id]:
            if window.overlaps(busy.window):
                conflicts.append(
                    Conflict(
                        "chamber_busy",
                        job.job_id,
                        chamber_id=job.chamber_id,
                        window=busy.window,
                        related_job_id=busy.reference or None,
                    )
                )
                continue
            if busy.window.end <= start:
                # The candidate run follows this activity: its cooldown
                # must have elapsed before the run starts.
                gap = start - busy.window.end
                if gap < busy.cooldown_after:
                    conflicts.append(
                        Conflict(
                            "cooldown_unmet",
                            job.job_id,
                            chamber_id=job.chamber_id,
                            window=busy.window,
                            related_job_id=busy.reference or None,
                            note=f"need {busy.cooldown_after}, have {gap}",
                        )
                    )
            elif end <= busy.window.start:
                # A later activity follows the candidate run: the run's own
                # cooldown must fit into the gap.
                gap = busy.window.start - end
                if gap < job.cooldown_after:
                    conflicts.append(
                        Conflict(
                            "cooldown_unmet",
                            job.job_id,
                            chamber_id=job.chamber_id,
                            window=busy.window,
                            related_job_id=busy.reference or None,
                            note=f"need {job.cooldown_after} after run, have {gap}",
                        )
                    )

        for engineer in engineers:
            covering = [shift for shift in engineer.shifts if shift.start <= start and end <= shift.end]
            if not covering:
                containing = next(
                    (shift for shift in engineer.shifts if shift.contains(start)),
                    None,
                )
                if containing is not None:
                    conflicts.append(
                        Conflict(
                            "engineer_duty_gap",
                            job.job_id,
                            engineer_id=engineer.engineer_id,
                            window=containing,
                            note="shift ends before the run finishes",
                        )
                    )
                else:
                    conflicts.append(
                        Conflict(
                            "outside_engineer_duty",
                            job.job_id,
                            engineer_id=engineer.engineer_id,
                        )
                    )
                continue
            for busy_window, reference in self.engineer_busy[engineer.engineer_id]:
                if window.overlaps(busy_window):
                    conflicts.append(
                        Conflict(
                            "engineer_busy",
                            job.job_id,
                            engineer_id=engineer.engineer_id,
                            window=busy_window,
                            related_job_id=reference or None,
                        )
                    )
        return conflicts

    def feasible_placements(
        self, job: CertificationJob, engineers: list[Engineer]
    ) -> list[tuple[int, Conflict, str]]:
        """Feasible ``(start, engineer_id)`` pairs in deterministic order."""

        feasible: list[tuple[int, Conflict, str]] = []
        for start in self.candidate_starts(job, engineers):
            for engineer in engineers:
                if not self.candidate_conflicts(job, [engineer], start):
                    feasible.append((start, None, engineer.engineer_id))
        feasible.sort(key=lambda item: (item[0], item[2]))
        return feasible


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Holds chamber/engineer master data and committed placements.

    A successful request appends placements atomically; a failed request
    leaves the book unchanged, so partial allocations can never be observed.
    Request ids are idempotent: repeating the same id returns the stored
    outcome.
    """

    def __init__(
        self,
        chambers: list[Chamber],
        engineers: list[Engineer],
        prior_activities: list[ChamberActivity] | None = None,
        prior_commitments: list[EngineerCommitment] | None = None,
    ) -> None:
        self.chambers: dict[str, Chamber] = {c.chamber_id: c for c in chambers}
        if len(self.chambers) != len(chambers):
            raise ValueError("duplicate chamber id")
        self.engineers: dict[str, Engineer] = {e.engineer_id: e for e in engineers}
        if len(self.engineers) != len(engineers):
            raise ValueError("duplicate engineer id")
        self._prior_activities = list(prior_activities or ())
        self._prior_commitments = list(prior_commitments or ())
        self._placements: dict[str, Placement] = {}
        self._requests: dict[str, SchedulingResult] = {}

    def placements(self) -> tuple[Placement, ...]:
        return tuple(sorted(self._placements.values(), key=lambda p: (p.window.start, p.job_id)))

    def schedule(
        self,
        jobs: list[CertificationJob],
        request_id: str | None = None,
        node_budget: int = 200_000,
    ) -> SchedulingResult:
        if request_id is not None and request_id in self._requests:
            return self._requests[request_id]
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("duplicate job id in request")
        result = self._solve(jobs, request_id=request_id, node_budget=node_budget)
        if request_id is not None:
            self._requests[request_id] = result
        if result.scheduled:
            for placement in result.placed:
                self._placements[placement.job_id] = placement
        return result

    def _new_state(self) -> _State:
        activities = [
            *self._prior_activities,
            *(
                ChamberActivity(p.chamber_id, p.window, p.cooldown_after, p.job_id)
                for p in self._placements.values()
            ),
        ]
        commitments = [
            *self._prior_commitments,
            *(
                EngineerCommitment(p.engineer_id, p.window, p.job_id)
                for p in self._placements.values()
            ),
        ]
        return _State(self.chambers, self.engineers, activities, commitments)

    def _solve(
        self,
        jobs: list[CertificationJob],
        request_id: str | None,
        node_budget: int,
    ) -> SchedulingResult:
        state = self._new_state()
        job_by_id = {job.job_id: job for job in jobs}
        conflicts: dict[tuple, Conflict] = {}

        def add(conflict: Conflict) -> None:
            conflicts[_conflict_key(conflict)] = conflict

        eligible_engineers: dict[str, list[Engineer]] = {}
        for job in jobs:
            if job.chamber_id not in self.chambers:
                add(Conflict("chamber_unknown", job.job_id, chamber_id=job.chamber_id))
                eligible_engineers[job.job_id] = []
                continue
            eligible = [
                engineer
                for engineer in self.engineers.values()
                if engineer.can_run(job.chamber_id, job.recipe_revision)
            ]
            eligible.sort(key=lambda engineer: engineer.engineer_id)
            eligible_engineers[job.job_id] = eligible
            if not eligible:
                for engineer in sorted(self.engineers.values(), key=lambda e: e.engineer_id):
                    add(
                        Conflict(
                            "engineer_unqualified",
                            job.job_id,
                            chamber_id=job.chamber_id,
                            engineer_id=engineer.engineer_id,
                        )
                    )

        # Static pass against the committed baseline.  A job that is infeasible
        # before any sibling is placed can never succeed in this request; its
        # conflict list is final.  Feasible jobs still go to the joint search.
        searchable: list[CertificationJob] = []
        for job in jobs:
            eligible = eligible_engineers[job.job_id]
            if not eligible:
                continue
            envelope_conflict = self._envelope_conflict(job)
            if envelope_conflict is not None:
                add(envelope_conflict)
                continue
            placements = state.feasible_placements(job, eligible)
            if placements:
                searchable.append(job)
                continue
            for start in state.candidate_starts(job, eligible):
                for conflict in state.candidate_conflicts(job, eligible, start):
                    add(conflict)

        if not searchable:
            ordered_conflicts = tuple(conflicts[key] for key in sorted(conflicts))
            return SchedulingResult(request_id, (), ordered_conflicts)

        placements, budget_exhausted, traces = self._search(
            searchable, eligible_engineers, state, node_budget
        )
        if placements is not None:
            if not conflicts:
                ordered = tuple(
                    sorted(placements, key=lambda placement: (placement.window.start, placement.job_id))
                )
                return SchedulingResult(request_id, ordered, ())
            # The searchable jobs fit jointly; only jobs already proven
            # infeasible on their own keep the batch from committing.
            ordered_conflicts = tuple(conflicts[key] for key in sorted(conflicts))
            return SchedulingResult(request_id, (), ordered_conflicts)

        batch_ids = set(job_by_id)
        for job_id, job_conflicts in traces.items():
            peer_related = False
            for conflict in job_conflicts.values():
                add(conflict)
                if conflict.related_job_id in batch_ids:
                    peer_related = True
            if peer_related:
                add(
                    Conflict(
                        "batch_contention",
                        job_id,
                        chamber_id=job_by_id[job_id].chamber_id,
                        note="feasible alone, but the joint request over-contends resources",
                    )
                )

        # Symmetrise peer contention: if job B was observed blocking job A,
        # mirror the evidence onto B so every contending job's conflict list
        # is complete.
        peer_pairs = {
            (conflict.related_job_id, conflict.job_id)
            for conflict in conflicts.values()
            if conflict.related_job_id in batch_ids
        }
        for blocking_job_id, blocked_job_id in sorted(peer_pairs):
            blocking = job_by_id[blocking_job_id]
            add(
                Conflict(
                    "batch_contention",
                    blocking_job_id,
                    chamber_id=blocking.chamber_id,
                    related_job_id=blocked_job_id,
                    note=f"contends with {blocked_job_id}",
                )
            )

        for job in searchable:
            if not any(conflict.job_id == job.job_id for conflict in conflicts.values()):
                code = "search_incomplete" if budget_exhausted else "batch_contention"
                note = (
                    f"solver node budget {node_budget} exhausted"
                    if budget_exhausted
                    else "feasible alone, but the joint request cannot be jointly placed"
                )
                add(
                    Conflict(
                        code,
                        job.job_id,
                        chamber_id=job.chamber_id,
                        note=note,
                    )
                )

        ordered_conflicts = tuple(conflicts[key] for key in sorted(conflicts))
        return SchedulingResult(request_id, (), ordered_conflicts)

    def _envelope_conflict(self, job: CertificationJob) -> Conflict | None:
        # Earliest start plus the run length must still finish by the
        # certification usability deadline.
        if job.earliest_start + job.duration > job.latest_finish:
            return Conflict(
                "validity_window_too_tight",
                job.job_id,
                chamber_id=job.chamber_id,
                note=(
                    f"duration {job.duration} cannot finish by {job.latest_finish} "
                    f"after earliest start {job.earliest_start}"
                ),
            )
        return None

    def _search(
        self,
        jobs: list[CertificationJob],
        eligible_engineers: Mapping[str, list[Engineer]],
        state: _State,
        node_budget: int,
    ) -> tuple[list[Placement] | None, bool, dict[str, dict[tuple, Conflict]]]:
        """Depth-first backtracking with most-constrained-job-first ordering."""

        traces: dict[str, dict[tuple, Conflict]] = {job.job_id: {} for job in jobs}
        nodes = 0
        budget_exhausted = False

        def record_failure(job: CertificationJob) -> None:
            starts = state.candidate_starts(job, eligible_engineers[job.job_id])
            recorded = traces[job.job_id]
            for start in starts:
                for conflict in state.candidate_conflicts(job, eligible_engineers[job.job_id], start):
                    recorded[_conflict_key(conflict)] = conflict

        def dfs(remaining: list[CertificationJob]) -> list[Placement] | None:
            nonlocal nodes, budget_exhausted
            if not remaining:
                return []
            nodes += 1
            if nodes > node_budget:
                budget_exhausted = True
                return None

            ranked = self._rank_jobs(remaining, eligible_engineers, state)
            job = ranked[0]
            rest = ranked[1:]
            options = state.feasible_placements(job, eligible_engineers[job.job_id])
            if not options:
                record_failure(job)
                return None
            for start, _marker, engineer_id in options:
                window = TimeBox(start, start + job.duration)
                placement = Placement(
                    job.job_id,
                    job.chamber_id,
                    engineer_id,
                    window,
                    job.cooldown_after,
                    job.validity_expires_at,
                )
                state.push(placement)
                deeper = dfs(rest)
                if deeper is not None:
                    return [placement, *deeper]
                state.pop(placement)
            record_failure(job)
            return None

        placements = dfs(jobs)
        return placements, budget_exhausted, traces

    def _rank_jobs(
        self,
        jobs: list[CertificationJob],
        eligible_engineers: Mapping[str, list[Engineer]],
        state: _State,
    ) -> list[CertificationJob]:
        def key(job: CertificationJob) -> tuple:
            options = state.feasible_placements(job, eligible_engineers[job.job_id])
            room = (
                options[-1][0] - options[0][0]
                if options
                else job.latest_finish - job.earliest_start
            )
            return (len(options), room, job.earliest_start, job.job_id)

        return sorted(jobs, key=key)


def schedule_certifications(
    jobs: list[CertificationJob],
    chambers: list[Chamber],
    engineers: list[Engineer],
    prior_activities: list[ChamberActivity] | None = None,
    prior_commitments: list[EngineerCommitment] | None = None,
    node_budget: int = 200_000,
) -> SchedulingResult:
    """Stateless one-shot certification scheduling.

    Returns every placement when the whole batch fits, otherwise the complete
    conflict list with no placements made.
    """

    scheduler = Scheduler(chambers, engineers, prior_activities, prior_commitments)
    return scheduler.schedule(jobs, node_budget=node_budget)

from __future__ import annotations

import pytest

from fabweave.core.timebox import TimeBox
from fabweave.scheduling import (
    CertificationJob,
    Chamber,
    ChamberActivity,
    Engineer,
    EngineerCommitment,
    Placement,
    Scheduler,
    schedule_certifications,
)


def chamber(chamber_id: str = "C1", maintenance=()) -> Chamber:
    return Chamber(chamber_id=chamber_id, maintenance=tuple(maintenance))


def engineer(
    engineer_id: str = "E1",
    shifts=(TimeBox(0, 1000),),
    revisions=frozenset({1}),
    chambers=frozenset(),
) -> Engineer:
    return Engineer(
        engineer_id=engineer_id,
        shifts=tuple(shifts),
        recipe_revisions=revisions,
        chamber_ids=chambers,
    )


def job(
    job_id: str = "J1",
    chamber_id: str = "C1",
    duration: int = 10,
    earliest_start: int = 0,
    validity_expires_at: int = 1000,
    required_valid_for: int = 0,
    cooldown_after: int = 0,
    recipe_revision: int = 1,
) -> CertificationJob:
    return CertificationJob(
        job_id=job_id,
        chamber_id=chamber_id,
        recipe_revision=recipe_revision,
        duration=duration,
        earliest_start=earliest_start,
        validity_expires_at=validity_expires_at,
        required_valid_for=required_valid_for,
        cooldown_after=cooldown_after,
    )


# ---------------------------------------------------------------------------
# Happy path and each hard constraint
# ---------------------------------------------------------------------------


def test_simple_scheduling_places_job_at_earliest_start():
    result = schedule_certifications([job()], [chamber()], [engineer()])
    assert result.scheduled
    (placed,) = result.placed
    assert placed.window == TimeBox(0, 10)
    assert placed.engineer_id == "E1"
    assert placed.certified_at == 10


def test_maintenance_is_avoided_not_overlapped():
    result = schedule_certifications(
        [job()],
        [chamber(maintenance=[TimeBox(0, 20)])],
        [engineer()],
    )
    assert result.scheduled
    assert result.placed[0].window == TimeBox(20, 30)


def test_job_may_start_exactly_when_maintenance_ends():
    # Half-open intervals: touching is allowed.
    result = schedule_certifications(
        [job(earliest_start=20)],
        [chamber(maintenance=[TimeBox(10, 20)])],
        [engineer()],
    )
    assert result.scheduled
    assert result.placed[0].window == TimeBox(20, 30)


def test_cooldown_after_prior_activity_is_respected():
    result = schedule_certifications(
        [job(earliest_start=0)],
        [chamber()],
        [engineer()],
        prior_activities=[ChamberActivity("C1", TimeBox(0, 10), cooldown_after=15)],
    )
    assert result.scheduled
    assert result.placed[0].window == TimeBox(25, 35)


def test_cooldown_gap_of_zero_means_touching_is_allowed():
    result = schedule_certifications(
        [job(earliest_start=10)],
        [chamber()],
        [engineer()],
        prior_activities=[ChamberActivity("C1", TimeBox(0, 10), cooldown_after=0)],
    )
    assert result.scheduled
    assert result.placed[0].window == TimeBox(10, 20)


def test_job_cooldown_must_separate_it_from_a_later_activity():
    later = ChamberActivity("C1", TimeBox(40, 50), cooldown_after=0)
    result = schedule_certifications(
        [job(cooldown_after=10)],
        [chamber()],
        [engineer()],
        prior_activities=[later],
    )
    # earliest 0 + duration 10 + cooldown 10 must be <= 40 => start 0 fits.
    assert result.scheduled
    assert result.placed[0].window == TimeBox(0, 10)

    tight = schedule_certifications(
        [job(earliest_start=25, cooldown_after=10)],
        [chamber()],
        [engineer(shifts=(TimeBox(25, 40),))],
        prior_activities=[later],
    )
    # start 25 -> end 35, cooldown ends 45 > 40 colliding with [40,50);
    # starting at 50 leaves no coverage in the short duty window.
    assert not tight.scheduled
    assert "cooldown_unmet" in {c.code for c in tight.conflicts}


def test_engineer_must_cover_whole_run_within_one_shift():
    result = schedule_certifications(
        [job(duration=20)],
        [chamber()],
        [engineer(shifts=(TimeBox(0, 25),))],
    )
    assert result.scheduled
    assert result.placed[0].window == TimeBox(0, 20)

    bad = schedule_certifications(
        [job(earliest_start=10, duration=20)],
        [chamber()],
        [engineer(shifts=(TimeBox(0, 25),))],
    )
    assert not bad.scheduled
    assert {c.code for c in bad.conflicts} == {"engineer_duty_gap"}


def test_engineer_shift_boundaries_are_half_open():
    # end <= shift.end: a run ending exactly at shift end is covered.
    result = schedule_certifications(
        [job(earliest_start=15, duration=10)],
        [chamber()],
        [engineer(shifts=(TimeBox(0, 25),))],
    )
    assert result.scheduled


def test_run_landing_entirely_between_shifts_is_outside_duty():
    result = schedule_certifications(
        [job(earliest_start=30, duration=10, validity_expires_at=45)],
        [chamber()],
        [engineer(shifts=(TimeBox(0, 20), TimeBox(50, 70)))],
    )
    assert not result.scheduled
    assert {c.code for c in result.conflicts} == {"outside_engineer_duty"}


def test_engineer_skill_restrictions_are_enforced():
    result = schedule_certifications(
        [job(recipe_revision=2)],
        [chamber()],
        [engineer(revisions=frozenset({1}))],
    )
    assert not result.scheduled
    assert {c.code for c in result.conflicts} == {"engineer_unqualified"}

    chamber_limited = schedule_certifications(
        [job(chamber_id="C2")],
        [chamber("C1"), chamber("C2")],
        [engineer(chambers=frozenset({"C1"}))],
    )
    assert not chamber_limited.scheduled
    assert {c.code for c in chamber_limited.conflicts} == {"engineer_unqualified"}


def test_required_validity_window_pushes_finish_deadline():
    # Finishes at 10; certification must still be valid for 90 => deadline 100.
    ok = schedule_certifications(
        [job(validity_expires_at=190, required_valid_for=90)],
        [chamber()],
        [engineer()],
    )
    assert ok.scheduled

    late = schedule_certifications(
        [job(validity_expires_at=95, required_valid_for=90)],
        [chamber()],
        [engineer()],
    )
    assert not late.scheduled
    assert {c.code for c in late.conflicts} == {"validity_window_too_tight"}


def test_validity_deadline_moves_the_job_before_a_later_obstacle():
    # Shift after 100 exists, but certification must finish by 100,
    # so the run cannot wait for it.
    result = schedule_certifications(
        [job(earliest_start=75, duration=20, validity_expires_at=100)],
        [chamber()],
        [engineer(shifts=(TimeBox(0, 90), TimeBox(100, 200)))],
    )
    assert not result.scheduled
    assert {c.code for c in result.conflicts} == {"engineer_duty_gap"}


def test_unknown_chamber_is_reported():
    result = schedule_certifications(
        [job(chamber_id="GHOST")], [chamber()], [engineer()]
    )
    assert not result.scheduled
    assert {c.code for c in result.conflicts} == {"chamber_unknown"}


# ---------------------------------------------------------------------------
# Multiple engineers and multiple jobs
# ---------------------------------------------------------------------------


def test_scheduler_picks_idle_engineer_when_another_is_busy():
    commitment = EngineerCommitment("E1", TimeBox(0, 20))
    result = schedule_certifications(
        [job(earliest_start=5)],
        [chamber()],
        [engineer("E1"), engineer("E2")],
        prior_commitments=[commitment],
    )
    assert result.scheduled
    assert result.placed[0].engineer_id == "E2"


def test_batch_places_multiple_jobs_in_sequence_with_cooldown():
    jobs = [
        job("J1", earliest_start=0, cooldown_after=5),
        job("J2", earliest_start=0, cooldown_after=5),
    ]
    result = schedule_certifications(jobs, [chamber()], [engineer()])
    assert result.scheduled
    windows = {p.job_id: p.window for p in result.placed}
    assert windows["J1"] == TimeBox(0, 10)
    assert windows["J2"] == TimeBox(15, 25)


def test_two_chambers_run_in_parallel_on_one_engineer_is_serialized_correctly():
    jobs = [
        job("J1", chamber_id="C1", earliest_start=0),
        job("J2", chamber_id="C2", earliest_start=0),
    ]
    result = schedule_certifications(jobs, [chamber("C1"), chamber("C2")], [engineer()])
    assert result.scheduled
    first, second = sorted(result.placed, key=lambda p: p.window.start)
    assert first.window == TimeBox(0, 10)
    assert second.window == TimeBox(10, 20)


def test_two_engineers_run_two_chambers_in_parallel():
    jobs = [
        job("J1", chamber_id="C1", earliest_start=0),
        job("J2", chamber_id="C2", earliest_start=0),
    ]
    result = schedule_certifications(
        jobs,
        [chamber("C1"), chamber("C2")],
        [engineer("E1"), engineer("E2")],
    )
    assert result.scheduled
    assert all(p.window == TimeBox(0, 10) for p in result.placed)
    assert {p.engineer_id for p in result.placed} == {"E1", "E2"}


def test_solver_backtracks_when_first_fit_choice_dead_ends():
    # E2 can run both chambers, E1 only C1. J3 is pinned to C2 starting at
    # 10 with a tight deadline (only E2 can cover it). J1 and J2 are
    # interchangeable across C1/C2. Picking E2-on-C2 at time 0 for the first
    # job dead-ends J3; the solver must backtrack and leave E2 free at 10.
    chambers = [chamber("C1"), chamber("C2")]
    engineers = [
        Engineer("E1", (TimeBox(0, 1000),), frozenset({1}), frozenset({"C1"})),
        Engineer("E2", (TimeBox(0, 1000),), frozenset({1}), frozenset({"C1", "C2"})),
    ]
    jobs = [
        job("J1", chamber_id="C1", earliest_start=0),
        job("J2", chamber_id="C2", earliest_start=0),
        job("J3", chamber_id="C2", earliest_start=10, validity_expires_at=25),
    ]
    result = schedule_certifications(jobs, chambers, engineers)
    assert result.scheduled
    by_job = {p.job_id: p for p in result.placed}
    assert by_job["J3"].engineer_id == "E2"
    assert by_job["J3"].window == TimeBox(10, 20)


def test_mixed_static_and_batch_conflicts_are_all_reported():
    # J1 impossible on its own (maintenance covers its whole validity window);
    # J2/J3 contend and would fail jointly anyway. Nothing may be held.
    jobs = [
        job("J1", earliest_start=0, validity_expires_at=30),
        job("J2", earliest_start=0, duration=30),
        job("J3", earliest_start=0, duration=30),
    ]
    result = schedule_certifications(
        jobs,
        [chamber(maintenance=[TimeBox(0, 30)])],
        [engineer(shifts=(TimeBox(0, 40),))],
    )
    assert not result.scheduled
    assert set(result.conflicts_for("J1"))
    assert any(c.code == "maintenance_clash" for c in result.conflicts_for("J1"))
    assert set(result.conflicts_for("J2"))
    assert set(result.conflicts_for("J3"))
    assert result.placed == ()


def test_batch_contention_is_detected_and_fully_reported():
    # Two jobs, one chamber, both length 30, only a 40-tick shift:
    # each fits alone, never jointly.
    jobs = [
        job("J1", earliest_start=0, duration=30),
        job("J2", earliest_start=0, duration=30),
    ]
    result = schedule_certifications(jobs, [chamber()], [engineer(shifts=(TimeBox(0, 40),))])
    assert not result.scheduled
    assert {c.job_id for c in result.conflicts} == {"J1", "J2"}
    codes = {c.code for c in result.conflicts}
    assert "batch_contention" in codes
    # Detailed blockers still point at the sibling jobs.
    assert {c.related_job_id for c in result.conflicts if c.related_job_id} == {"J1", "J2"}


# ---------------------------------------------------------------------------
# Atomicity and stability
# ---------------------------------------------------------------------------


def test_failed_batch_holds_no_resources():
    scheduler = Scheduler([chamber()], [engineer(shifts=(TimeBox(0, 40),))])
    impossible = [
        job("J1", earliest_start=0, duration=30),
        job("J2", earliest_start=0, duration=30),
    ]
    failed = scheduler.schedule(impossible)
    assert not failed.scheduled
    assert scheduler.placements() == ()
    # A follow-up feasible request proves nothing was occupied.
    followup = scheduler.schedule([job("J3", earliest_start=0)])
    assert followup.scheduled
    assert followup.placed[0].window == TimeBox(0, 10)


def test_successful_batch_commits_and_blocks_later_requests():
    scheduler = Scheduler([chamber()], [engineer()])
    first = scheduler.schedule([job("J1", earliest_start=0, cooldown_after=5)], request_id="R1")
    assert first.scheduled
    second = scheduler.schedule([job("J2", earliest_start=0)], request_id="R2")
    assert second.scheduled
    assert second.placed[0].window == TimeBox(15, 25)


def test_request_id_is_idempotent():
    scheduler = Scheduler([chamber()], [engineer()])
    first = scheduler.schedule([job("J1")], request_id="R1")
    second = scheduler.schedule([job("J1")], request_id="R1")
    assert first == second
    assert len(scheduler.placements()) == 1


def test_scheduling_is_deterministic_across_input_order_and_retries():
    jobs_a = [job("J1"), job("J2", earliest_start=5)]
    jobs_b = list(reversed(jobs_a))
    first = schedule_certifications(jobs_a, [chamber()], [engineer()])
    second = schedule_certifications(jobs_b, [chamber()], [engineer()])
    assert first.placed == second.placed

    scheduler = Scheduler([chamber()], [engineer()])
    run1 = scheduler.schedule(jobs_a)
    run2 = schedule_certifications(jobs_a, [chamber()], [engineer()])
    assert run1.placed == run2.placed


def test_conflict_listing_is_stable_and_sorted():
    result = schedule_certifications(
        [job("J1"), job("J2")],
        [chamber(maintenance=[TimeBox(0, 1000)])],
        [engineer(shifts=(TimeBox(2000, 3000),))],
    )
    assert not result.scheduled
    codes = {(c.job_id, c.code) for c in result.conflicts}
    assert ("J1", "maintenance_clash") in codes
    assert ("J2", "maintenance_clash") in codes
    assert ("J1", "outside_engineer_duty") in codes
    assert ("J2", "outside_engineer_duty") in codes
    # Deterministic order across repeated calls.
    again = schedule_certifications(
        [job("J1"), job("J2")],
        [chamber(maintenance=[TimeBox(0, 1000)])],
        [engineer(shifts=(TimeBox(2000, 3000),))],
    )
    assert result.conflicts == again.conflicts


def test_duplicate_conflicts_are_deduplicated():
    # One long maintenance window blocks every candidate; it must appear once.
    result = schedule_certifications(
        [job(earliest_start=0, duration=10, validity_expires_at=40)],
        [chamber(maintenance=[TimeBox(0, 35)])],
        [engineer()],
    )
    assert not result.scheduled
    maintenance = [c for c in result.conflicts if c.code == "maintenance_clash"]
    assert len(maintenance) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        Engineer("E", shifts=(), recipe_revisions=frozenset({1}))
    with pytest.raises(ValueError):
        Engineer("E", shifts=(TimeBox(0, 10),), recipe_revisions=frozenset())
    with pytest.raises(ValueError):
        CertificationJob(
            job_id="J",
            chamber_id="C",
            recipe_revision=1,
            duration=0,
            earliest_start=0,
            validity_expires_at=10,
        )
    with pytest.raises(ValueError):
        schedule_certifications(
            [job("J1"), job("J1")], [chamber()], [engineer()]
        )


def test_placement_is_a_placement_type():
    result = schedule_certifications([job()], [chamber()], [engineer()])
    assert isinstance(result.placed[0], Placement)

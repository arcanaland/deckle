from __future__ import annotations

import errno
import os

import pytest

from deckle.ids import canonical_78
from deckle.labels import (
    COMMENT_XATTR,
    Label,
    LabelError,
    format_label,
    parse_label,
    plan_staging,
    read_label,
)
from deckle.project import PROJECT_FILE, Project, ProjectError, init
from deckle.staging import write_staging_index


def parse(text: str, default: str | None = None):
    return parse_label(text, card_back_default=default)


# --- grammar ----------------------------------------------------------------------------

SUITS = {"w": "wands", "c": "cups", "s": "swords", "p": "pentacles"}
RANKS = {
    "1": "ace",
    "a": "ace",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "t": "ten",
    "p": "page",
    "n": "knight",
    "q": "queen",
    "k": "king",
}


@pytest.mark.parametrize("suit", SUITS)
@pytest.mark.parametrize("rank", RANKS)
def test_every_minor_shorthand(suit, rank):
    label = parse(f"{suit}{rank}")
    assert label.ref == f"minor_arcana.{SUITS[suit]}.{RANKS[rank]}"
    assert label.rotate_180 is False


@pytest.mark.parametrize(
    "text, ref", [("m0", "major_arcana.00"), ("m2", "major_arcana.02"), ("m21", "major_arcana.21")]
)
def test_major_shorthand_is_zero_padded(text, ref):
    assert parse(text).ref == ref


@pytest.mark.parametrize(
    "text",
    [
        "major_arcana.02",
        "major_arcana.02:crowned",
        "minor_arcana.swords.seven",
        "minor_arcana.coins.sevn",  # custom suit and rank: valid grammar, roster says no
    ],
)
def test_longhand_passes_through(text):
    assert parse(text).ref == text


def test_shorthand_with_a_variant():
    label = parse("m2:crowned")
    assert (label.ref, label.variant, label.rotate_180) == (
        "major_arcana.02:crowned",
        "crowned",
        False,
    )


def test_rotation_mark_after_a_variant():
    label = parse("m2:crowned~")
    assert (label.ref, label.rotate_180) == ("major_arcana.02:crowned", True)


def test_surrounding_whitespace_is_stripped():
    assert parse("  w7~\n").ref == "minor_arcana.wands.seven"


@pytest.mark.parametrize("text", ["m2~:crowned", "~m2", "m2~~", "w~7"])
def test_rotation_mark_anywhere_but_the_end_is_refused(text):
    with pytest.raises(LabelError, match="goes at the end"):
        parse(text)


def test_backs():
    assert parse("back:alt").card_back == "alt"
    assert parse("back:alt").target == "card_backs.alt"
    assert parse("back", "classic").card_back == "classic"
    label = parse("back~", "classic")
    assert (label.card_back, label.rotate_180, label.card) == ("classic", True, None)


def test_bare_back_with_no_default_is_refused():
    with pytest.raises(LabelError, match="give a design: back:<key>"):
        parse("back")


@pytest.mark.parametrize(
    "text, match",
    [
        ("m2:Crowned", "not a custom name"),
        ("m2:king", "reserved"),
        ("back:2nd", "not a custom name"),
        ("M14", "expected m<number>"),
        ("m14:", "empty key"),
        ("back:", "empty key"),
        ("m100", "expected m<number>"),
        ("w0", "expected m<number>"),
        ("x7", "expected m<number>"),
        ("major_arcana.2", "both digits"),
        ("", "empty label"),
    ],
)
def test_bad_labels(text, match):
    with pytest.raises(LabelError, match=match):
        parse(text)


@pytest.mark.parametrize("card", canonical_78(), ids=str)
def test_format_label_round_trips_every_card(card):
    for variant in (None, "alt"):
        for rotate in (False, True):
            label = Label(card=card, variant=variant, card_back=None, rotate_180=rotate)
            assert parse(format_label(label)) == label


@pytest.mark.parametrize(
    "ref, text",
    [
        ("major_arcana.00", "m0"),
        ("major_arcana.14:crowned", "m14:crowned"),
        ("minor_arcana.wands.ace", "w1"),
        ("minor_arcana.cups.ten", "c10"),
        ("minor_arcana.pentacles.page", "pp"),
        ("major_arcana.the_fool", "major_arcana.the_fool"),
        ("minor_arcana.coins.seven", "minor_arcana.coins.seven"),
        ("minor_arcana.swords.sevn", "minor_arcana.swords.sevn"),
    ],
)
def test_format_label_is_short_where_it_can_be(ref, text):
    label = parse(ref)
    assert format_label(label) == text
    assert parse(text) == label


def test_format_label_names_the_back_design_even_when_it_is_the_default():
    label = parse("back~", "plain")
    assert format_label(label) == "back:plain~"


# --- planning ---------------------------------------------------------------------------


def make_project(tmp_path) -> Project:
    return init(tmp_path / "proj", "Test Deck")


def stage(project: Project, name: str, label: str | None = None):
    path = project.staging_dir / name
    path.write_bytes(b"pixels")
    if label is not None:
        try:
            os.setxattr(path, COMMENT_XATTR, label.encode())
        except OSError as exc:
            if exc.errno == errno.ENOTSUP:
                pytest.skip(f"{path.parent} does not support user xattrs")
            raise
    return path


def touch_master(project: Project, relpath: str):
    path = project.masters_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pixels")
    return path


def test_read_label_without_the_attribute_is_none(tmp_path):
    project = make_project(tmp_path)
    assert read_label(stage(project, "a_r0c0.png")) is None
    assert read_label(stage(project, "b_r0c0.png", "   ")) is None
    assert read_label(stage(project, "c_r0c0.png", " w7 ")) == "w7"


def test_a_clean_plan(tmp_path):
    project = make_project(tmp_path)
    stage(project, "s1_r0c0.png", "m2")
    stage(project, "s1_r0c1.png", "m2:crowned~")
    stage(project, "s1_r1c0.png", "pk")
    stage(project, "s1_r1c1.png", "back:plain")
    (project.staging_dir / "notes.txt").write_text("ignored")

    plan = plan_staging(project, force=False)
    assert plan.problems == [] and plan.warnings == []
    assert (plan.files, plan.labelled, plan.unlabelled) == (4, 4, 0)
    assert [(r.source.name, r.label.target, r.label.rotate_180) for r in plan.rows] == [
        ("s1_r0c0.png", "major_arcana.02", False),
        ("s1_r0c1.png", "major_arcana.02:crowned", True),
        ("s1_r1c0.png", "minor_arcana.pentacles.king", False),
        ("s1_r1c1.png", "card_backs.plain", False),
    ]
    assert plan.rows[1].dest == project.masters_dir / "major_arcana/02.crowned.png"
    assert (
        plan.rows[1].describe(project.root)
        == "s1_r0c1.png -> major_arcana.02:crowned (rotated 180)"
    )
    assert plan.summary() == "4 files, 4 labelled, 0 unlabelled, 0 problem(s)"


def test_unlabelled_files_are_counted_not_problems(tmp_path):
    project = make_project(tmp_path)
    stage(project, "s_r0c0.png", "w7")
    stage(project, "s_r0c1.png")
    stage(project, "s_r1c0.png", "")
    plan = plan_staging(project, force=False)
    assert (plan.files, plan.labelled, plan.unlabelled) == (3, 1, 2)
    assert plan.problems == []
    assert [r.source.name for r in plan.rows] == ["s_r0c0.png"]


def test_an_empty_staging_is_an_empty_plan(tmp_path):
    project = make_project(tmp_path)
    (project.staging_dir).rmdir()
    plan = plan_staging(project, force=False)
    assert (plan.files, plan.rows, plan.problems) == (0, [], [])


def test_a_parse_error_names_the_file(tmp_path):
    project = make_project(tmp_path)
    stage(project, "s_r0c0.png", "M14")
    plan = plan_staging(project, force=False)
    assert plan.labelled == 1 and plan.rows == []
    assert len(plan.problems) == 1
    assert plan.problems[0].startswith("s_r0c0.png: ")
    assert "expected m<number>" in plan.problems[0]


@pytest.mark.parametrize("label", ["minor_arcana.swords.sevn", "major_arcana.23"])
def test_a_card_outside_the_roster_is_a_problem(tmp_path, label):
    project = make_project(tmp_path)
    stage(project, "s_r0c0.png", label)
    plan = plan_staging(project, force=False)
    assert plan.problems == [f"s_r0c0.png: {label} is not in the roster"]


def test_an_extended_major_in_the_roster_is_fine(tmp_path):
    project = make_project(tmp_path)
    project.doc["roster"]["majors"].append("23")
    project.save()
    stage(project, "s_r0c0.png", "m23")
    assert plan_staging(Project.load(project.root), force=False).problems == []


def test_two_bare_labels_on_one_card(tmp_path):
    project = make_project(tmp_path)
    stage(project, "a_r0c1.png", "m14~")
    stage(project, "b_r1c0.png", "major_arcana.14")
    (problem,) = plan_staging(project, force=False).problems
    assert "major_arcana.14" in problem
    assert "a_r0c1.png" in problem and "b_r1c0.png" in problem
    assert "give one of these a variant key" in problem


def test_a_duplicate_variant(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/02.png")
    stage(project, "a_r0c0.png", "m2:crowned")
    stage(project, "b_r0c0.png", "m2:crowned~")
    (problem,) = plan_staging(project, force=False).problems
    assert "major_arcana.02:crowned" in problem
    assert "a_r0c0.png" in problem and "b_r0c0.png" in problem
    assert "variant key" not in problem


def test_a_duplicate_back_design(tmp_path):
    project = make_project(tmp_path)
    project.doc["project"]["card_back_default"] = "plain"
    project.save()
    stage(project, "a_r0c0.png", "back")
    stage(project, "b_r0c0.png", "back:plain")
    (problem,) = plan_staging(Project.load(project.root), force=False).problems
    assert "card_backs.plain" in problem and "a_r0c0.png" in problem and "b_r0c0.png" in problem


def test_a_variant_without_a_default_is_a_problem(tmp_path):
    project = make_project(tmp_path)
    stage(project, "a_r0c0.png", "m2:crowned")
    (problem,) = plan_staging(project, force=False).problems
    assert "major_arcana.02" in problem and "no default master" in problem


def test_a_variant_passes_when_the_default_is_already_a_master(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/02.png")
    stage(project, "a_r0c0.png", "m2:crowned")
    assert plan_staging(project, force=False).problems == []


def test_a_variant_passes_when_the_default_is_in_the_same_plan(tmp_path):
    project = make_project(tmp_path)
    stage(project, "a_r0c0.png", "m2:crowned")
    stage(project, "b_r0c0.png", "m2")
    assert plan_staging(project, force=False).problems == []


def test_a_variant_only_card_the_plan_does_not_touch_is_not_its_problem(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/02.crowned.png")
    stage(project, "a_r0c0.png", "w7")
    assert plan_staging(project, force=False).problems == []


def test_replacing_a_recorded_master_is_a_problem(tmp_path):
    project = make_project(tmp_path)
    old = touch_master(project, "minor_arcana/wands/seven.png")
    project.record_card(
        "minor_arcana.wands.seven", {"master": "masters/minor_arcana/wands/seven.png"}
    )
    project.save()
    stage(project, "s_r0c0.png", "w7")
    plan = plan_staging(Project.load(project.root), force=False)
    (problem,) = plan.problems
    assert "minor_arcana.wands.seven already has a master" in problem and "--force" in problem
    assert plan.rows[0].replaces == old


def test_a_record_without_its_file_still_guards(tmp_path):
    project = make_project(tmp_path)
    project.record_card("minor_arcana.wands.seven", {"source_scan": "s.jpg"})
    project.save()
    stage(project, "s_r0c0.png", "w7")
    (problem,) = plan_staging(Project.load(project.root), force=False).problems
    assert "already has a master" in problem


def test_an_unrecorded_file_at_the_destination_guards(tmp_path):
    project = make_project(tmp_path)
    old = touch_master(project, "card_backs/plain.png")
    stage(project, "s_r0c0.png", "back:plain")
    plan = plan_staging(project, force=False)
    (problem,) = plan.problems
    assert "card_backs.plain already has a master (masters/card_backs/plain.png)" in problem
    assert plan.rows[0].replaces == old


def test_force_lifts_the_replace_problem(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "minor_arcana/wands/seven.png")
    stage(project, "s_r0c0.png", "w7")
    plan = plan_staging(project, force=True)
    assert plan.problems == []
    assert plan.rows[0].describe(project.root) == (
        "s_r0c0.png -> minor_arcana.wands.seven (replaces masters/minor_arcana/wands/seven.png)"
    )


def test_a_non_rectified_name_warns_but_plans(tmp_path):
    project = make_project(tmp_path)
    stage(project, "seven of wands.png", "w7")
    plan = plan_staging(project, force=False)
    assert plan.problems == []
    assert [r.label.ref for r in plan.rows] == ["minor_arcana.wands.seven"]
    (warning,) = plan.warnings
    assert warning.startswith("seven of wands.png: ") and "provenance" in warning


def test_the_filename_is_never_a_label(tmp_path):
    project = make_project(tmp_path)
    stage(project, "w7.png")
    plan = plan_staging(project, force=False)
    assert (plan.unlabelled, plan.rows, plan.warnings) == (1, [], [])


def test_a_stale_index_row_is_ignored(tmp_path):
    project = make_project(tmp_path)
    stage(project, "s_r0c0.png", "w7")
    write_staging_index(project, {"gone_r0c0.png": {"source_scan": "gone.jpg", "slot": "r0c0"}})
    plan = plan_staging(project, force=False)
    assert plan.files == 1 and plan.problems == []


def test_a_degraded_project_refuses(tmp_path):
    project = make_project(tmp_path)
    (project.root / PROJECT_FILE).unlink()
    with pytest.raises(ProjectError, match="cannot assign"):
        plan_staging(Project.load(project.root), force=False)

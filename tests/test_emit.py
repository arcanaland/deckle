"""Regression tests on what `emit` *constructs*.

These are not a validator and must not become one. Validating a `deck.toml` against §9.4 is
libarcana's job; deckle will link or shell out to it once libarcana covers 2.0, and a second
implementation of that rule table would drift from the first invisibly. What is asserted
here is the narrower thing deckle owes: **emit builds only what it has files for**, so the
three §9.4 errors a partial deck could trip never arise. If one of these tests ever needs a
rule table to express itself, the line has been crossed.

The two structural claims are `test_h_roots_are_exactly_their_height` (§5.3) and
`test_re_emit_is_byte_identical`, which is what makes a deck directory disposable.
"""

from __future__ import annotations

import shutil
import tomllib

import cv2
import numpy as np
import pytest

from deckle.emit import MARKER, EmitError, emit
from deckle.project import Project, init


def make_project(tmp_path, **kw):
    project = init(tmp_path / "proj", "Test Deck", deck_dir=str(tmp_path / "deck"), **kw)
    project.doc["deck"].update({"name": "Test Deck", "version": "1.0"})
    project.save()
    return project


def write_master(project, relpath: str, *, height: int = 2840, aspect: float = 0.583) -> None:
    """A master with the shape of a real one: portrait, native scale, lossless PNG."""
    path = project.masters_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    width = round(height * aspect)
    rng = np.random.default_rng(abs(hash(relpath)) % (2**32))
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def read_deck_toml(target):
    with (target / "deck.toml").open("rb") as fh:
        return tomllib.load(fh)


def tree(target):
    return {
        p.relative_to(target).as_posix(): p.read_bytes()
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }


# -- the pyramid ----------------------------------------------------------------------


def test_h_roots_are_exactly_their_height(tmp_path):
    """§5.3: an `h<height>/` root is one where `<height>` *is* the pixel height."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png", height=2843)
    write_master(project, "minor_arcana/cups/king.png", height=2841)
    result = emit(project)

    for height in (750, 1200, 2400):
        images = sorted((result.target / f"h{height}").rglob("*.png"))
        assert images, f"h{height} is empty"
        for path in images:
            assert cv2.imread(str(path)).shape[0] == height, path


def test_widths_may_differ_between_cards(tmp_path):
    """Nothing requires an `h` root to be uniform in width. Each master is measured
    independently, so a common width could only be reached by distorting one of them."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png", height=2843, aspect=0.5835)
    write_master(project, "major_arcana/09.png", height=2841, aspect=0.5825)
    emit(project)

    widths = {cv2.imread(str(p)).shape[1] for p in (tmp_path / "deck/h2400").rglob("*.png")}
    assert len(widths) == 2


def test_a_target_above_the_master_is_skipped_not_upscaled(tmp_path):
    """§5.7.3: downscaling a raster is well-behaved and upscaling is not."""
    project = make_project(tmp_path)
    project.doc["project"]["emit_heights"] = [750, 1200, 2400, 4000]
    project.save()
    write_master(project, "major_arcana/03.png", height=2843)

    result = emit(project)
    assert not (result.target / "h4000").exists()
    assert any("h4000" in w and "skipping" in w for w in result.warnings)
    assert (result.target / "h2400/major_arcana/03.png").is_file()


def test_variants_pass_through(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/06.two_women.png")
    emit(project)
    assert (tmp_path / "deck/h1200/major_arcana/06.two_women.png").is_file()


def test_card_backs_reach_the_top_level_and_every_h_root(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    # §5.5: a back MAY differ in aspect ratio from the fronts, so no front-aspect check
    # applies to it.
    write_master(project, "card_backs/classic.png", height=2840, aspect=0.72)
    emit(project)

    assert (tmp_path / "deck/card_backs/classic.png").is_file()
    for height in (750, 1200, 2400):
        path = tmp_path / f"deck/h{height}/card_backs/classic.png"
        assert cv2.imread(str(path)).shape[0] == height


# -- regenerability -------------------------------------------------------------------


def test_re_emit_is_byte_identical(tmp_path):
    """`rm -rf` the deck directory and re-emit. This is what makes it disposable, and it
    is the property every other decision in [[ADR-003]] rests on."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png", height=2843)
    write_master(project, "minor_arcana/cups/king.png", height=2841)
    write_master(project, "card_backs/classic.png")
    (project.names_dir / "en.toml").write_text('[suits]\ncups = "Honey Pots"\n', encoding="utf-8")

    target = emit(project).target
    first = tree(target)
    shutil.rmtree(target)
    emit(project)
    assert tree(target) == first


def test_emit_replaces_stale_output(tmp_path):
    """A card removed from masters/ must not survive in the deck directory."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    write_master(project, "major_arcana/09.png")
    target = emit(project).target
    assert (target / "h750/major_arcana/09.png").is_file()

    (project.masters_dir / "major_arcana/09.png").unlink()
    emit(project)
    assert not (target / "h750/major_arcana/09.png").exists()
    assert (target / "h750/major_arcana/03.png").is_file()


# -- the sanity guard -----------------------------------------------------------------


def test_emit_refuses_a_directory_it_did_not_write(tmp_path):
    """The failure mode being guarded is deleting real data under
    `~/.local/share/tarot/decks/`."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    target = tmp_path / "deck"
    target.mkdir()
    (target / "something-precious.txt").write_text("hand-written", encoding="utf-8")

    with pytest.raises(EmitError, match="carries no"):
        emit(project)
    assert (target / "something-precious.txt").read_text() == "hand-written"


def test_emit_accepts_an_empty_directory_and_marks_it(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    (tmp_path / "deck").mkdir()
    target = emit(project).target
    assert (target / MARKER).is_file()
    emit(project)  # now recognised as its own


def test_emit_refuses_a_symlinked_target(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    real = tmp_path / "elsewhere"
    real.mkdir()
    (tmp_path / "deck").symlink_to(real)
    with pytest.raises(EmitError, match="symlink"):
        emit(project)


def test_emit_refuses_an_empty_project(tmp_path):
    project = make_project(tmp_path)
    with pytest.raises(EmitError, match="nothing to emit"):
        emit(project)


def test_emit_refuses_a_degraded_project(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    (project.root / "deckle.toml").write_text("broken [", encoding="utf-8")
    with pytest.raises(EmitError):
        emit(Project.load(project.root))


# -- what emit constructs -------------------------------------------------------------


def test_ranks_lists_name_only_ranks_with_files(tmp_path):
    """§9.4 **E**: every rank named in a `ranks` list has files in that suit.

    A 15-rank cups roster with three cups scanned must not emit the other twelve.
    """
    project = make_project(tmp_path)
    project.doc["roster"]["suits"]["cups"] = [
        "ace",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "princess",
        "page",
        "knight",
        "queen",
        "king",
    ]
    project.save()
    for rank in ("ace", "princess", "king"):
        write_master(project, f"minor_arcana/cups/{rank}.png")

    doc = read_deck_toml(emit(project).target)
    assert doc["suits"]["cups"]["ranks"] == ["ace", "princess", "king"]


def test_an_unmodified_canonical_suit_gets_no_ranks_list(tmp_path):
    """§4.4: a `ranks` list *replaces* a canonical suit's sequence, so writing one for a
    suit the roster has not modified would assert that the deck's cups suit has only the
    cards scanned so far. A card with no files is a resolution failure (§5.7.6), not a
    violation, so silence is the truthful answer."""
    project = make_project(tmp_path)
    write_master(project, "minor_arcana/cups/king.png")
    doc = read_deck_toml(emit(project).target)
    assert "suits" not in doc


def test_a_custom_suit_is_described_and_filtered(tmp_path):
    project = make_project(tmp_path)
    project.doc["roster"]["suits"]["stars"] = ["ace", "two", "three"]
    project.save()
    write_master(project, "minor_arcana/stars/ace.png")
    doc = read_deck_toml(emit(project).target)
    assert doc["suits"] == {"stars": {"ranks": ["ace"]}}


def test_cards_table_names_only_emitted_cards(tmp_path):
    """§9.4 **E**: every card declared in `[cards]` is a card the deck has files for."""
    project = make_project(tmp_path)
    project.doc["card_meta"] = {
        "major_arcana.23": {"number": "XXIII"},
        "major_arcana.happy_squirrel": {"name": "The Happy Squirrel"},
    }
    project.save()
    write_master(project, "major_arcana/23.png")

    result = emit(project)
    doc = read_deck_toml(result.target)
    assert doc["cards"] == {"major_arcana.23": {"number": "XXIII"}}
    assert any("happy_squirrel" in w for w in result.warnings)


def test_card_back_default_is_omitted_when_the_design_is_absent(tmp_path):
    """§9.4 **E**: `[card_backs].default` names a design the deck has."""
    project = make_project(tmp_path)
    project.doc["project"]["card_back_default"] = "classic"
    project.save()
    write_master(project, "major_arcana/03.png")

    result = emit(project)
    assert "card_backs" not in read_deck_toml(result.target)
    assert any("card_back_default" in w for w in result.warnings)

    write_master(project, "card_backs/classic.png")
    assert read_deck_toml(emit(project).target)["card_backs"] == {"default": "classic"}


def test_provenance_never_reaches_the_deck(tmp_path):
    """`[cards]` in the project file is provenance; `[cards]` in `deck.toml` is display
    metadata. Conflating them would put a scan filename in front of a reader."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    project.record_card("major_arcana.03", {"source_scan": "20260801204347_001.jpg"})
    project.save()
    assert "20260801204347_001" not in (emit(project).target / "deck.toml").read_text()


# -- the deck table -------------------------------------------------------------------


def test_schema_version_and_aspect_are_the_projects(tmp_path):
    project = make_project(tmp_path)
    project.doc["project"]["aspect"] = 0.583
    project.doc["deck"]["schema_version"] = "1.1"  # a stale value must not survive
    project.save()
    write_master(project, "major_arcana/03.png")

    deck = read_deck_toml(emit(project).target)["deck"]
    assert deck["schema_version"] == "2.0"
    assert deck["aspect_ratio"] == 0.583


def test_dates_are_written_as_strings(tmp_path):
    """§4.1: `created_date` and `updated_date` are strings. A bare date in the project
    file parses as a TOML date, which §9.4 makes an error in the output."""
    project = make_project(tmp_path)
    (project.root / "deckle.toml").write_text(
        '[project]\ndeck_dir = "{}"\n\n[deck]\nname = "D"\nversion = "1"\n'
        "created_date = 2026-08-04\n".format(tmp_path / "deck"),
        encoding="utf-8",
    )
    project = Project.load(project.root)
    write_master(project, "major_arcana/03.png")
    assert read_deck_toml(emit(project).target)["deck"]["created_date"] == "2026-08-04"


def test_links_must_be_absolute_http(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    for bad in (
        [{"rel": "homepage", "url": "/relative"}],
        [{"rel": "homepage", "url": "ftp://example.com/"}],
        [{"rel": "Home Page", "url": "https://example.com/"}],
        [{"url": "https://example.com/"}],
    ):
        project.doc["deck"]["links"] = bad
        with pytest.raises(EmitError):
            emit(project)

    project.doc["deck"]["links"] = [
        {"rel": "homepage", "url": "https://example.com/", "title": "Home"}
    ]
    doc = read_deck_toml(emit(project).target)
    assert doc["deck"]["links"] == [
        {"rel": "homepage", "url": "https://example.com/", "title": "Home"}
    ]


def test_path_fields_stay_inside_the_deck_root(tmp_path):
    """§10.1 / §2.3."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    for bad in ("/etc/passwd", "../outside.png", "a/../../b.png"):
        project.doc["deck"]["icon"] = bad
        with pytest.raises(EmitError):
            emit(project)


def test_name_and_version_are_required(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    del project.doc["deck"]["version"]
    with pytest.raises(EmitError, match="version"):
        emit(project)


def test_unknown_deck_keys_are_carried_not_dropped(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    project.doc["deck"]["some_future_key"] = "authorship"
    assert read_deck_toml(emit(project).target)["deck"]["some_future_key"] == "authorship"


def test_name_files_are_rendered_from_the_project(tmp_path):
    project = make_project(tmp_path)
    write_master(project, "minor_arcana/cups/king.png")
    (project.names_dir / "en.toml").write_text('[suits]\ncups = "Honey Pots"\n', encoding="utf-8")
    target = emit(project).target
    assert (target / "names/en.toml").read_text() == '[suits]\ncups = "Honey Pots"\n'


def test_excluded_cards_are_carried(tmp_path):
    project = make_project(tmp_path)
    project.doc["roster"]["excluded"] = ["minor_arcana.pentacles.page"]
    project.save()
    write_master(project, "major_arcana/03.png")
    doc = read_deck_toml(emit(project).target)
    assert doc["excluded_cards"]["cards"] == ["minor_arcana.pentacles.page"]

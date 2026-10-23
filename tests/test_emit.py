"""What `emit` constructs.

These test the generator, not the deck: emit builds only what it has files for. They are
not a validator, and a test that needs a rule table to express itself belongs elsewhere.
"""

from __future__ import annotations

import shutil
import tomllib
import zlib

import cv2
import numpy as np
import pytest

from deckle.emit import MARKER, EmitError, emit
from deckle.project import DECK_IN_FILE, Project, init


def make_project(tmp_path, **kw):
    """A project with its authored metadata in the project file and no deck.in.toml."""
    project = init(tmp_path / "proj", "Test Deck", deck_dir=str(tmp_path / "deck"), **kw)
    (project.root / DECK_IN_FILE).unlink()
    project.doc["deck"] = {"name": "Test Deck", "version": "1.0"}
    project.save()
    return project


def make_authored_project(tmp_path, text: str):
    project = init(tmp_path / "proj", "Test Deck", deck_dir=str(tmp_path / "deck"))
    (project.root / DECK_IN_FILE).write_text(text, encoding="utf-8")
    return project


DECK = '[deck]\nname = "Authored"\nversion = "0.9"\n'


def write_master(project, relpath: str, *, height: int = 2840, aspect: float = 0.6) -> None:
    """A master with the shape of a real one: portrait, native scale, lossless PNG."""
    path = project.masters_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    width = round(height * aspect)
    rng = np.random.default_rng(zlib.crc32(relpath.encode()))
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
    write_master(project, "major_arcana/03.png", height=2843, aspect=0.6005)
    write_master(project, "major_arcana/09.png", height=2841, aspect=0.5995)
    emit(project)

    widths = {cv2.imread(str(p)).shape[1] for p in (tmp_path / "deck/h2400").rglob("*.png")}
    assert len(widths) == 2


def test_a_target_above_the_master_is_skipped_not_upscaled(tmp_path):
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
    # A back may differ in aspect ratio from the fronts.
    write_master(project, "card_backs/classic.png", height=2840, aspect=0.72)
    emit(project)

    assert (tmp_path / "deck/card_backs/classic.png").is_file()
    for height in (750, 1200, 2400):
        path = tmp_path / f"deck/h{height}/card_backs/classic.png"
        assert cv2.imread(str(path)).shape[0] == height


# -- regenerability -------------------------------------------------------------------


def test_re_emit_is_byte_identical(tmp_path):
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
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    target = tmp_path / "deck"
    target.mkdir()
    (target / "something-precious.txt").write_text("hand-written", encoding="utf-8")

    with pytest.raises(EmitError, match="does not have a .deckle-emit marker file"):
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
    """A 15-rank cups roster with three cups scanned must not emit the other twelve."""
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
    """A `ranks` list replaces a canonical suit's sequence, so one for an unchanged suit
    would claim the suit holds only the cards scanned so far."""
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
    metadata."""
    project = make_project(tmp_path)
    write_master(project, "major_arcana/03.png")
    project.record_card("major_arcana.03", {"source_scan": "scan_0001.jpg"})
    project.save()
    assert "scan_0001" not in (emit(project).target / "deck.toml").read_text()


# -- the deck table -------------------------------------------------------------------


def test_schema_version_and_aspect_are_the_projects(tmp_path):
    project = make_project(tmp_path)
    project.doc["project"]["aspect"] = 0.6
    project.doc["deck"]["schema_version"] = "1.1"  # a stale value must not survive
    project.save()
    write_master(project, "major_arcana/03.png")

    deck = read_deck_toml(emit(project).target)["deck"]
    assert deck["schema_version"] == "2.0"
    assert deck["aspect_ratio"] == 0.6


def test_dates_are_written_as_strings(tmp_path):
    project = make_project(tmp_path)
    (project.root / "deckle.toml").write_text(
        '[project]\ndeck_dir = "{}"\n\n[deck]\nname = "D"\nversion = "1"\n'
        "created_date = 1909-12-01\n".format(tmp_path / "deck"),
        encoding="utf-8",
    )
    project = Project.load(project.root)
    write_master(project, "major_arcana/03.png")
    assert read_deck_toml(emit(project).target)["deck"]["created_date"] == "1909-12-01"


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


# -- deck.in.toml ---------------------------------------------------------------------


def test_the_authored_file_is_the_source_of_the_deck_table(tmp_path):
    project = make_authored_project(
        tmp_path,
        "# comments are the author's\n"
        + DECK
        + 'related = [{ rel = "pattern", target = "land.arcana/pattern/thoth" }]\n'
        "published_date = 1969-01-01\n",
    )
    write_master(project, "major_arcana/03.png")
    deck = read_deck_toml(emit(project).target)["deck"]
    assert (deck["name"], deck["version"]) == ("Authored", "0.9")
    assert deck["related"] == [{"rel": "pattern", "target": "land.arcana/pattern/thoth"}]
    assert deck["published_date"] == "1969-01-01"
    assert deck["schema_version"] == "2.0"


def test_name_and_version_errors_name_the_authored_file(tmp_path):
    project = make_authored_project(tmp_path, '[deck]\nname = "Authored"\n')
    write_master(project, "major_arcana/03.png")
    with pytest.raises(EmitError, match=f"version is required; set it in .*{DECK_IN_FILE}"):
        emit(project)


@pytest.mark.parametrize("table", ["deck", "card_meta"])
def test_two_sources_are_refused(tmp_path, table):
    project = make_authored_project(tmp_path, DECK)
    project.doc[table] = {"name": "Other"} if table == "deck" else {"major_arcana.03": {"a": 1}}
    project.save()
    write_master(project, "major_arcana/03.png")
    with pytest.raises(EmitError, match=rf"deckle.toml has \[{table}\] and {DECK_IN_FILE} exists"):
        emit(project)


def test_empty_tables_in_the_project_file_are_not_a_second_source(tmp_path):
    project = make_authored_project(tmp_path, DECK)
    project.doc["deck"] = {}
    project.doc["card_meta"] = {}
    project.save()
    write_master(project, "major_arcana/03.png")
    assert read_deck_toml(emit(project).target)["deck"]["name"] == "Authored"


@pytest.mark.parametrize(
    "value", ["schema_version = 2", 'schema_version = "1.1"', "aspect_ratio = 0.5"]
)
def test_computed_deck_keys_cannot_be_overridden(tmp_path, value):
    project = make_authored_project(tmp_path, DECK + value + "\n")
    write_master(project, "major_arcana/03.png")
    with pytest.raises(EmitError, match="but emit computes"):
        emit(project)


def test_computed_deck_keys_may_be_restated(tmp_path):
    project = make_authored_project(tmp_path, DECK + 'schema_version = "2.0"\n')
    project.doc["project"]["aspect"] = 0.6
    project.save()
    (project.root / DECK_IN_FILE).write_text(DECK + 'schema_version = "2.0"\naspect_ratio = 0.6\n')
    write_master(project, "major_arcana/03.png")
    assert read_deck_toml(emit(project).target)["deck"]["aspect_ratio"] == 0.6


@pytest.mark.parametrize(
    "text", ['[card_back."default"]\nreversible = true\n', 'card_backs = "default"\n']
)
def test_a_top_level_shape_emit_does_not_know_is_refused(tmp_path, text):
    project = make_authored_project(tmp_path, text + DECK)
    write_master(project, "major_arcana/03.png")
    with pytest.raises(EmitError, match=DECK_IN_FILE):
        emit(project)


def test_authored_cards_are_projected_onto_masters(tmp_path):
    project = make_authored_project(
        tmp_path,
        DECK + '[cards."major_arcana.03"]\nnumber = "III"\n'
        '[cards."major_arcana.06:two_women"]\nname = "The Lovers"\n'
        '[cards."major_arcana.happy_squirrel"]\nname = "The Happy Squirrel"\n',
    )
    write_master(project, "major_arcana/03.png")
    write_master(project, "major_arcana/06.two_women.png")

    result = emit(project)
    assert read_deck_toml(result.target)["cards"] == {
        "major_arcana.03": {"number": "III"},
        "major_arcana.06:two_women": {"name": "The Lovers"},
    }
    assert any("happy_squirrel" in w and "no master" in w for w in result.warnings)


def test_authored_image_paths_stay_inside_the_deck_root(tmp_path):
    project = make_authored_project(
        tmp_path, DECK + '[cards."major_arcana.03"]\nimage = "../outside.png"\n'
    )
    write_master(project, "major_arcana/03.png")
    with pytest.raises(EmitError, match="must not contain a `..` segment"):
        emit(project)


def test_authored_links_are_validated(tmp_path):
    project = make_authored_project(
        tmp_path, DECK + 'links = [{ rel = "homepage", url = "/relative" }]\n'
    )
    write_master(project, "major_arcana/03.png")
    with pytest.raises(EmitError, match="absolute"):
        emit(project)


def test_authored_card_backs_are_projected_and_guarded(tmp_path):
    project = make_authored_project(
        tmp_path,
        DECK + '[card_backs]\ndefault = "stars"\n'
        "[card_backs.designs.classic]\nreversible = true\n"
        "[card_backs.designs.stars]\nreversible = false\n",
    )
    write_master(project, "major_arcana/03.png")
    write_master(project, "card_backs/classic.png")

    result = emit(project)
    assert read_deck_toml(result.target)["card_backs"] == {
        "designs": {"classic": {"reversible": True}}
    }
    assert any("[card_backs].default = 'stars'" in w for w in result.warnings)
    assert any("[card_backs.designs.stars]" in w for w in result.warnings)


def test_an_authored_default_wins_over_the_projects(tmp_path):
    project = make_authored_project(tmp_path, DECK + '[card_backs]\ndefault = "stars"\n')
    project.doc["project"]["card_back_default"] = "classic"
    project.save()
    write_master(project, "major_arcana/03.png")
    write_master(project, "card_backs/classic.png")
    write_master(project, "card_backs/stars.png")
    assert read_deck_toml(emit(project).target)["card_backs"] == {"default": "stars"}


def test_authored_suits_merge_into_the_computed_ranks(tmp_path):
    project = make_authored_project(tmp_path, DECK + '[suits.stars]\nname = "Stars"\n')
    project.doc["roster"]["suits"]["stars"] = ["ace", "two", "three"]
    project.save()
    write_master(project, "minor_arcana/stars/ace.png")
    doc = read_deck_toml(emit(project).target)
    assert doc["suits"] == {"stars": {"name": "Stars", "ranks": ["ace"]}}


def test_tables_emit_computes_nothing_for_are_carried(tmp_path):
    project = make_authored_project(
        tmp_path,
        DECK + '[ranks.king]\nname = "Roy"\n'
        '[minor_arcana]\nname_template = "{rank} de {suit}"\n'
        '[strings]\nlicense = "CC0-1.0"\n'
        '[app."com.example"]\nseen = true\n',
    )
    write_master(project, "major_arcana/03.png")
    doc = read_deck_toml(emit(project).target)
    assert doc["ranks"] == {"king": {"name": "Roy"}}
    assert doc["minor_arcana"] == {"name_template": "{rank} de {suit}"}
    assert doc["strings"] == {"license": "CC0-1.0"}
    assert doc["app"] == {"com.example": {"seen": True}}


def test_an_authored_exclusion_of_a_scanned_card_is_dropped(tmp_path):
    project = make_authored_project(
        tmp_path,
        DECK + "[excluded_cards]\n"
        'cards = ["major_arcana.03", "minor_arcana.cups.page"]\nreason = "Not printed."\n',
    )
    write_master(project, "major_arcana/03.png")
    result = emit(project)
    assert read_deck_toml(result.target)["excluded_cards"] == {
        "cards": ["minor_arcana.cups.page"],
        "reason": "Not printed.",
    }
    assert any("major_arcana.03" in w for w in result.warnings)


def test_an_unwritable_authored_value_is_an_emit_error(tmp_path):
    project = make_authored_project(tmp_path, DECK + '[cards."major_arcana.03"]\nat = 07:32:00\n')
    write_master(project, "major_arcana/03.png")
    with pytest.raises(EmitError, match="cannot write deck.toml"):
        emit(project)

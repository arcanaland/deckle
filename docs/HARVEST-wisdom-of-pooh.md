# Harvest record: `wisdom-of-pooh-tarot`, 2026-08-04

Verbatim capture of `~/.local/share/tarot/decks/wisdom-of-pooh-tarot/` **before** any
deckle code ran against it. TASK-004 step 1; ADR-003 §Consequences.

The five cards there were hand-named and hand-copied. Their pixels are recoverable from
the source scans — **their identity assignment is not**, and existed nowhere but those
five filenames until this file. Nothing under `h2800/` may be deleted until this record
is committed.

## The five cards, and where their pixels came from

Identity was assigned by hand. Provenance below was **recovered by pixel-matching**, not
inferred from timestamps: each `h2800/` image was reduced to a 64×110 normalised
greyscale signature and correlated against every master `deckle rectify` produces from the
candidate scans. Every match is r = 1.0000 against a next-best of ≤ 0.71, so none of the
five is ambiguous.

| Canonical ID | Harvested file | `w×h` | Source scan | Slot | Edge strategy |
| --- | --- | --- | --- | --- | --- |
| `major_arcana.03` | `h2800/major_arcana/03.png` | 1660×2843 | `20260801204347_001.jpg` | `r0c0` | `innermost` (pre-pad) |
| `major_arcana.09` | `h2800/major_arcana/09.png` | 1660×2841 | `20260804193927_001.jpg` | `r1c0` | default |
| `major_arcana.16` | `h2800/major_arcana/16.png` | 1659×2841 | `20260804193927_001.jpg` | `r1c1` | default |
| `minor_arcana.cups.king` | `h2800/minor_arcana/cups/king.png` | 1659×2842 | `20260801204347_001.jpg` | `r0c1` | `innermost` (pre-pad) |
| `minor_arcana.pentacles.king` | `h2800/minor_arcana/pentacles/king.png` | 1660×2841 | `20260804193927_001.jpg` | `r0c0` | default |

`20260801204347_001.jpg` is the repo-root reference scan, not one in
`/mnt/truenas/home/media/tarot/working/scans/`. Its two cards are the pair
`tests/test_reference_scan.py` carries calipers for — *The Empress* (70.36 × 120.32mm) and
*King of Honey Pots* (70.33 × 120.14mm) — which independently confirms `r0c0` =
`major_arcana.03` and `r0c1` = `minor_arcana.cups.king`.

`20260804193927_001.jpg` yields **four** masters. Slot `r0c1` was scanned on 08-04 and
never assigned an identity, so no fifth ID is recoverable from it.

SHA-256 of the harvested files, as found:

```
acd8ef6161979349078e7fb06bd2af9d2b656ac95075fb7282cf090a6b16bf74  h2800/major_arcana/03.png
8adae68e67f86b33bc60a5ff40c9bbf504f51a72f7f64476cdfabb99670db35a  h2800/major_arcana/09.png
4455486e2b4ef5b9691540f3746fcf718f721f8a5498b44e3e626012edfa48ee  h2800/major_arcana/16.png
8e7c55546873229b0b848e038955026d97a1c1d71b380ab78ae907963a3ff180  h2800/minor_arcana/cups/king.png
3e16bbaf400ee4eec82132b0e1826118685a1dbac1553e8dd153cbcf78d8b222  h2800/minor_arcana/pentacles/king.png
db83c56f772fc9fcf15c458e138c3e4a7fb52aee76bdaf2ecb147a307d96d7e2  deck.toml
35faf87d8422e26e6df1b947ac4538f5d9de659b3fe298097a81b827f28415b6  names/en.toml
```

## `deck.toml`, verbatim

```toml
[deck]
id = "wisdom-of-pooh-tarot"
schema_version = "1.1"
name = "Wisdom of Pooh Tarot"
version = "0.2"
author = "Serefina & Angel Mesa"
license = "Personal Use Only"
description = "This deck is a personal scan of a copyrighted tarot deck. Not for redistribution."
attribution = "Original artwork by Kat L. Amsel, published by Rue & Vervain."
# 70 mm x 121 mm
aspect_ratio = 0.5785
website = "https://www.rueandvervain.com/wisdom-of-pooh-tarot"
tags = ["copyrighted", "extra-cards", "suit-renamed"]
```

## `names/en.toml`, verbatim

```toml
[suits]
wands = ""
cups = "Honey Pots"
swords = ""
pentacles = ""
```

## How each field was carried across

Per TASK-004 §Harvesting Wisdom of Pooh. Everything below now lives in the project's
`deckle.toml` and is rendered by `deckle emit`.

| Field | Disposition |
| --- | --- |
| `id` | **Dropped.** Removed in 2.0 (Appendix B); the library handle is the directory name. |
| `schema_version` | `"1.1"` → `"2.0"`. |
| `name`, `version`, `author`, `description`, `attribution`, `tags` | Carried verbatim. |
| `website` | **Dropped as a field**, remodelled as `links = [{ rel = "homepage", url = "…" }]`. Removed in 2.0. |
| `license` | **Omitted.** `"Personal Use Only"` is not an SPDX expression and the package grants nothing. |
| — | `rights_status = "https://rightsstatements.org/vocab/InC/1.0/"`, `redistribution = "none"`, `derivation = "none"` added: §7.4/§7.5's way of saying what `license` was being misused to say. |
| `packager` | **Added** — `Adam Fidel`. §9.4 warns on a deck naming artwork it did not produce with no `packager`, and this deck names someone else's. |
| `aspect_ratio` | `0.5785` → `0.583`, the project's configured value, measured across eight cards. The inherited value came from a `70 × 121mm` comment; measurement says `70.0 × 120.0mm`. Both sit inside §9.4's 10% warning band, so this is correctness rather than conformance. |
| `identifier` | **Not written.** §3.3 needs a realm Adam controls and TASK-004 §Open questions forbids inventing one. A missing `identifier` is a §9.4 *warning* only. |
| `names/en.toml` empty strings | `wands`, `swords`, `pentacles` **dropped**. §6.3 uses a resolved string verbatim, so `""` renders as an empty name, whereas an *absent* key correctly falls back to the title-cased key. Only `cups = "Honey Pots"` is kept. |

### Unresolved, carried across verbatim

`deck.toml` credits `author = "Serefina & Angel Mesa"` while `attribution` credits "Kat L.
Amsel, published by Rue & Vervain". These contradict. TASK-004 says not to resolve it, so
both are carried across unchanged and the question is Adam's.

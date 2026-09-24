# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

Materials from Denver Water for the Explore DDD 2026 Design Storm, plus a 3D map
of their collection system. Start with `README.md`, then
`reference/Explore DDD 2026 Denver Water Design Storm Presentation.pdf` for the
challenge itself. `guide.md` explains the domain and Jake's models from zero;
`glossary.md` defines every water and statistics term.

## Ground rules

- **Denver Water's originals are read-only.** `data/`, `scripts/`, `figures/`, and
  `reference/` hold files exactly as sent. Work on copies.
- **The data terms travel with the data.** Both Denver Water notices are in
  `README.md` and `data/TERMS.md`. Keep them with any derived dataset or shared
  output. Redistribution is restricted; read them before publishing anything built
  on this data.
- **Readings are provisional.** A fresh API pull can differ from the committed CSVs;
  USGS revises values after publication. Never present a number as settled when the
  source calls it provisional.
- **Never invent numbers.** Compute them from the files here, or say you cannot.
  When stating general water-treatment knowledge rather than something in these
  materials, say which it is.
- **Known data artifact:** `data/MichiganCreek.csv` shows SWE 9.0 on May 12 to 15,
  2026 between zero readings, an error in the NRCS feed. Jake replaced the station
  in his September update; the models use `data/HoosierPass.csv`.

## The 3D map

`design-storm-water-system-3d.html` is hand-maintained; there is no build step for
the page itself. It reads generated JSON from `water-system-3d/` and geodata from
`strontia-brief/`, so those directories travel with it. Serve it rather than opening
the file directly, because it fetches JSON:

```
python3 serve.py
```

`water-system-3d/README.md` says which JSON files are generated, by which script,
and which of those scripts need the network.


# Agent Instructions
Always create the below checklist for every prompt:

## Checklist Manifesto
Always use your checklist or todo list tool to track items. Do not leave it to chance that you will remember later.
Immediately before implementing any prompts set up the following tasks as a checklist.
- Prod Check
- Preparatory Unit Test Coverage
- Make it easy to change (which may be hard) (refactoring)
- Make the easy change
- Security Review
- Scout Rule
- Single Loop Learning
- Double Loop Learning
- Canary

## Prod Check
When applicable, evaluate the existing health of the production system.
There is no running service here: this repo is static data plus notebooks. "Health" means confirming the toolchain
is actually usable before writing analysis code -- check which Python packages (pandas, scikit-learn, matplotlib,
pytest) are already installed rather than assuming, and note any that are missing instead of silently requiring them.

## Preparatory Unit Test Coverage
Ensure the area that will be changed has approrpriate characterization tests making it safe to refactor.
Ensure characterization tests pass before starting any refactoring.
For greenfield analysis code (a new `teams/<name>/` folder, nothing pre-existing to characterize), this instead means:
write the pure data-loading/transform logic as small testable functions first, with unit tests against tiny synthetic
fixtures, *before* pointing any script at the real multi-year CSVs in `data/`.

## Make it easy to change (which may be hard)
Refactor to common computer science grounded design patterns.
The resulting code should be easy to read, limited in file length, appropriately decoupled, and cohesive.

## Make the easy change
Complete the prompt considering YAGNI and DRY concepts in software development. 

## Security Review
Evaluate for common OWASP pitfalls.
Run automated audits like pip audit, npm audit and correct package issues.
Evaluate for harder to detect problems with the system such as IDOR vulnerabilities.

## Scout Rule
Always leave the code better than you found it. Perform one of the following in priority order each time a prompt leads you to this area of the code.
- Evaluate Code Coverage and add more complete tests
- File length gate, reduce the file length of the files when over 500 lines by refactoring
- Mutation testing, use a analysis tool to perform mutant hunting on the modified files. For example Cosmic Ray in Python or Striker in Angular.

## Single Loop Learning
Learn from the tasks you complete:
Always end all of our chats with a list of skills that you used.
Always create new skills in your skills folder that you wish you had before starting the prompt. Actually write the file now.
This repo's skills folder is `.github/skills/<skill-name>/SKILL.md` (same format as the global skill catalog: a
`name`/`description` frontmatter block plus the playbook). Check there before starting a similar task again.

## Double Loop Learning
Learn from the process improvement opportunities:
Always evaluate the the process used here using a lens of Lean Software Development, Agile, Systems Thinking, Safety, Security, and Continuous Improvement. 
Always make the changes to the AGENTS.md with these changes. Update this very list you are reading now.

2026-09-24 (shallow-learning parameter catalog task): the "Never invent numbers" ground rule and the checklist's
"Make the easy change" step were doing the same job from two different documents with no cross-link. Any task that
produces documentation with statistics should run the numbers through a script and write them to a results file
*before* drafting prose, then quote only from that file -- codified in the new `.github/skills/` entry so it isn't
re-derived from scratch next time.

2026-09-24 (challenge writeup task, same day): the skill file written earlier that day was picked up automatically
and reused successfully for a follow-on prompt in the same session -- confirms the Single Loop Learning step is
worth the extra file, not just ceremony. Also: `reference/*.pdf` should be read with `pdfplumber` (installed, verified
working) rather than paraphrased from README.md's quotes of it, even when those quotes look complete -- the deck had
one extra context line (slide 5's "2026 drought") that materially strengthened a downstream data-grounded claim and
would have been missed without opening the source file directly.

2026-09-24 (transit-time tracing task, same day, first RACN-commit-as-you-go session): committing after each small
step (rather than batching everything at the end) caught a git-cwd assumption bug immediately -- a `cd` inside an
earlier terminal call for PDF extraction silently changed the working directory for every later command in that
terminal, so a script invocation failed with a "file not found" that had nothing to do with the code. Running
scripts/tests right after each edit, in the same terminal, surfaces this class of drift fast; batching all edits
before running anything would have hidden it until much later. Also: when a task's own gap analysis says "cannot
literally do X because of missing data," look for an honest substitute that says what it *is* rather than pretending
to be X -- a synchronized-cursor animation across two panels, clearly captioned as not a literal traced water
parcel, satisfied "single unified animation" without overclaiming physics the data can't support.

2026-09-24 (SVM/lag-sweep/web-viewer task, same day, third gap-closing prompt in a row): a plain gap list pasted back
verbatim by the user turned out to be a reliable, low-ambiguity work order -- each bullet mapped to either "genuinely
blocked" (Strontia sonde, confirmed a third time, still absent) or "small isolated extension," exactly as the
original writeup predicted. When a catalog document itself says a gap is "a small, well-isolated extension of the
same module," trust that estimate and implement it directly rather than re-scoping. Also: actually serving and
opening a new `.html` deliverable with the browser tools (not just reading the source) caught nothing broken this
time, but is now the standing bar for any UI-producing task in this repo -- an unverified static page is not
"implemented," it's "written."

2026-09-24 (writeup-splitting task, same day, fourth prompt in a row): a single markdown file that grew across three
prior prompts (scenario writeup, transit-time tracing, SVM/viewer) had accumulated enough content that "split into
one file per scenario plus a parent" was the right call the moment it was asked, not premature -- recognizing that
threshold before being asked would have been a better default. Also: `create_file` refuses to overwrite an existing
path; when a file needs a full-content rewrite rather than a targeted edit, delete it first (`Remove-Item`) then
`create_file`, instead of fighting `replace_string_in_file` with a whole-file oldString match.

2026-09-24 (Strontia sonde discovery, sixth prompt in a row, same day): a data source repeatedly documented across
three prior prompts as "confirmed absent" (with escalating confidence language -- "confirmed twice", then "confirmed
a third time") turned out to be present in `data/` all along by the time this prompt ran a plain `list_dir` on it
instead of trusting the accumulated notes. Systems-thinking read: each repetition of the claim in a writeup, a memory
file, and a skill file *felt* like independent corroboration but was actually the same unverified fact copied
forward -- confidence compounded without any new evidence. Safety/quality read: this class of error is silent and
directional (a false "missing" claim only ever loses opportunities, never fabricates a wrong number), so it won't
surface on its own the way an invented-number error might get caught by a sanity check -- it has to be caught by
re-verifying inputs, not by reviewing outputs. Continuous-improvement action taken: reworded the skill file's
guidance from "note the gap" to "verify with `list_dir` every time, regardless of what prior notes say", and
corrected the standing memory note rather than just adding a newer one on top of it. Lean read: the fix once the
data was found was cheap (one new loader module, ~40 lines) compared to three prior sessions' worth of documentation
that had to be revisited and corrected -- the earlier, cheaper action (a 5-second directory listing) would have
been worth taking every single time this gap was mentioned, not just once at the start.

2026-09-24 (same session): when a file (`visualize.py`) crosses the repo's 500-line Scout Rule gate mid-task because
new work was added to it, split by topic/scenario immediately rather than deferring to "next time this file comes
up" -- a same-day split (into `visualize.py` + `visualize_transit.py` + `visualize_sonde.py`) cost one extra
edit-and-retest cycle here; deferring it would have meant re-reading and re-splitting a still-larger file later with
less memory of which function belonged to which scenario.

2026-09-24 ("what are we missing" prompt, seventh in a row, same day): asked to find gaps and better-fitting models,
grepping the repo for named sklearn classes not yet imported/used (`LogisticRegression|IsolationForest|GradientBoosting`)
was a faster and more reliable gap-finding method than reasoning about the domain from scratch -- it found a
genuinely dead import and two ideas parameters.md had documented as possibilities in an earlier session but never
implemented. Quality/safety read: adding gradient boosting because a reference document (guide.md) said it won
previously, then finding it performed *worst* of all models tried here, was worth reporting as a real negative
result with a stated reason (no hyperparameter tuning, unlike the reference) rather than quietly dropping the
comparison or re-running until a flattering result appeared -- the same "never invent numbers" discipline applies
to not un-inventing an inconvenient one. Also caught in this pass: the smoke test's expected-figures list had gone
stale (figures 13-16 existed and were generated by `main()` but were never added to the test's assertion list from
an earlier session) -- a reminder that adding a new output to a script and adding the corresponding test assertion
are two separate steps, and only one of them reliably happened together in this session's earlier passes.

## Canary
Always end all of our chats with "# 🪁" Emoji. It should render as a markdown header so the Emoji will be large.
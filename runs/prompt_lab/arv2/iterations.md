# Active reading v2: plan-prompt iterations

Lab: `lab.py` (plan builder `arv2_plan.py`, rounds in `rounds/<tag>/out.json`), stats `stats.py`, embeddings
`eval_emb.py` (Qwen3-Embedding-4B; within-doc Vendi of the first 8 views, view-source cosine, cross-doc cosine,
batch Vendi = one random view per doc, batch of min(256, ndocs) docs, 20 batches), reader packets `dump.py`.
Docs: the 16 pilot docs + random audit-sample docs (seed 0). Generator Qwen3.5-9B instruct (interactive server),
plan thinking on (16384 max), execution thinking off, T 1.0 / top_p 0.95 / top_k 20 / presence 1.5.

## Round summary

| round | docs x views | within Vendi (8) | view-src cos | cross-doc cos (real) | batch Vendi (real), batch | md title raw | src refs | metadata pat. | ex. leaks | median words (p10-p90) | <150w / >900w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r0 (user draft, pilot) | 16 x 8 | 3.55 | 0.69 | 0.095 (0.107) | 14.9 (14.8), 16 | 83% | 12% | 9% | 6 | 491 (234-807) | 2% / 8% |
| r1 | 48 x 8 | 3.43 | 0.71 | 0.105 (0.120) | 38.7 (38.3), 48 | 4% | 3% | 8% | 2 real (operating costs) | 470 (106-1026) | 16% / 16% |
| r2 | 48 x 10 | 3.43 | 0.69 | 0.110 (0.120) | 38.3 (38.3), 48 | 2% | 6% (12% by wider reject regex) | 3% | 0 | 416 (97-880) | 15% / 8% |
| r3 | 48 x 12-48 (mean 29) | 3.43 | 0.68 | 0.119 (0.120) | 37.1 (38.3), 48 | 6% | 17% (reject regex) | 3% | 0 | 385 (94-901) | 18% / 10% |
| r4 (harness smoke) | 48 x 12-48 (mean 29) | 3.51 | 0.67 | 0.117 (0.120) | 37.2 (38.4), 48 | 0% | 13.9% (rejected) | 3% | 0 | 383 (90-936) | 19% / 11% |
| **r5 (final, 400-doc harness pilot)** | 400 x 8-48 (mean 29) | 3.42 | 0.65 | 0.123 (0.123) | **126.3 (140.9), 256** | 0% | 8.2% rejected (13.6% after the stricter final regex) | 6% (mostly source dates) | 3 prompts (now filtered) | 385 (93-955) | 18% / 12% |

Reference arms on the r1 48 docs: within Vendi direct_nothink 1.35, direct_think 1.42, mix 2.43, ideas 4.52;
cross-doc cos 0.109 / 0.118 / 0.111 / 0.163; batch Vendi(48) 39.0 / 38.3 / 38.5 / 34.7.

Reference arms on the r5 400 docs (256-doc batches): batch Vendi real 140.9, direct_nothink 145.8, direct_think 139.4,
mix 140.0, ideas 118.1; within Vendi 1.40 / 1.44 / 2.49 / 4.52; view-source cos 0.86 / 0.86 / 0.77 / 0.56;
cross-doc cos 0.104 / 0.117 / 0.110 / 0.161.

## r0: user draft (plan_prompt.txt)
See the task brief: within-doc content overlap, example leakage, invented experiences/metadata, "the source" in
plans, "Write a ..." monoculture, markdown titles, narrow length band, fidelity drift.

## r1 (plan_prompt_r1.txt, examples_r1.txt)
Changes vs r0:
- Step 1 "divide the content": list content units in reasoning; one primary focus per prompt; at most 2 broad.
- Per-call randomization (arv2_plan.py): per-slot assigned length (very short 40-120 ... very long 900-1500,
  long/very long only if the source has >=250/600 words) and quality (edited / everyday / rough 30/40/30);
  a random menu of 20 web forms (of 60) and 10 organizing principles (of 20).
- One fictional example per call, drawn from a pool of 6 (library, cache docs, sourdough blog, footbridge history,
  tomato blight guide, bike-lane council summary), each with 3 prompts of different lengths/phrasings; rule that
  example facts/wording/caveats never appear.
- Explicit fidelity rules: no invented experiences/discoveries, no invented metadata (dates, bylines, names), no
  limitations the source does not state, no answering worksheets.
- Refer to material by what it is, never "the source"; plain text; vary phrasing (not all "Write a").
- Execution wrapper suffix: plain web text, no Markdown/HTML, never mention a source document / instructions.
- Deterministic markdown strip (`clean_markdown`): heading #s, bold/italic markers, horizontal rules, `* ` bullets.

Readers (4: across-doc + 3 within/fidelity, 12 docs each):
- Across-doc 3/5: surface variety good (no shared chrome), but every doc gets the same ladder (one blurb, one
  "Comprehensive" guide, one forum post, one FAQ, one memo/email, one "draft"); internal/offline genres (memos,
  tickets) bring invented metadata ("FROM: ... Date: [Current Date]", "October 26, 2023"); "landscape" openers.
- Within-doc distinctness 2-3/5: prompts re-list most facts, outputs drift back to full retellings; short or
  single-thread sources cannot support 8 distinct views.
- Fidelity 2/5 (~40-55% of views with >=1 false/distorted claim): many errors are planner-induced paraphrases
  ("doubles the likelihood" for 31%, "organized until 1959" for "not organized until 1959"); executor adds
  dates/numbers, strengthens hedges.
- Naturalness 3/5: house style ("testament", "Ultimately"), ~3% refer to "the text/source material".
- Rough writing 2/5: caricature (lowercase run-ons, "2 am", stripped punctuation, self-narrated confusion) or
  near-verbatim source copies; long rough views degenerate.
- Example leak: "did not measure operating costs" (library example caveat) in 2 plans.

## r2 (plan_prompt_r2.txt, examples_r2.txt, arv2_plan_r2.py)
Changes vs r1:
- "Point to the material, do not restate it": name facts with the source's own terms/short quotes; never
  restate numbers/dates/names/causes in own words, never assert what the source lacks; no requests to explain /
  define / date / quote what the source does not.
- Only the first prompt may be broad; focused prompts name the focus + at most 3-4 details (no fact lists).
- Per-slot assignment now = length + quality + 3 suggested forms drawn from a weighted list of ~46 PUBLIC web page
  types (no memos/internal emails); long/very long need >=450/900 source words; rough only up to medium length.
- Rough writing described concretely; caricature banned (slang theatrics, "2 am", lowercase gimmicks, "repeat X").
- Execution wrapper: rule list (plain text, never mention a source/"the text"/prompt, add no facts/dates/names/
  sign-offs/placeholders, own words, requested length).
- Example pool: library caveat "did not measure operating costs" replaced by a concrete fact (it leaked twice).
- Reject filter (arv2_plan.reject_reason): placeholders "[Your ...]", source references, script switches (CJK not
  in source), 10-gram repetition >20%, 8-gram copy rate from the source >50%.
- n = 10 prompts per plan call.

Readers (across + 3 within):
- Across 3/5: better (real news datelines, letters, forum threads with handles, ad copy); no memos; "landscape"
  mostly gone. Still a per-doc genre ladder (encyclopedia/FAQ/handout/bulletin in nearly every doc),
  "Comprehensive ..." in slot 1, genre labels leaking into the page ("Educational Overview: ..."), ~1/3 of views a
  neutral encyclopedic voice.
- Within distinctness 2-2.5/5: the writer pads long targets by retelling the whole source; short sources
  (~250 words) still get 10 views of up to 500-900 words -> nearly all pairs overlap.
- Fidelity 2-2.5/5 (~30-45% of views with >=1 false claim; ~5-15% planner-induced: merged facts, wrong
  direction "less alive" vs "more alive", amps called voltage, attributing claims to cited bodies).
- "the text notes / the source material" went UP (planner wrote "the text" in 28% of prompts; 12% of outputs
  rejected by the regex). House style "Ultimately", "testament".
- Rough writing 2/5: still mostly polished or gimmicky; source/prompt leakage in rough notes.

## r3 (plan_prompt_r3.txt, examples_r3.txt, arv2_plan_r3.py)
Changes vs r2:
- Views per source scale with its length: n_views = clamp(ceil(words / 30), 12, 48), planned in calls of <= 12;
  each later call sees the earlier prompts (title + first 30 words) and must pick other content/forms, no broad one.
- Lengths scaled to the source: a length bucket is allowed only if its typical size (80/185/375/700/1200 words)
  <= 1.2 x source words. Simulated on the 3,072-doc audit sample: ~12.5k kept synthetic tokens/doc = 1.10x the 3C
  need (11.43k/doc).
- One assigned form + one alternative per slot (instead of 3 suggestions); rough quality only with rough-plausible
  forms (forum posts, comments, personal blogs, notes...).
- Planner: ignore navigation/ads/chrome; do not merge facts the source keeps separate or attribute statements to
  others; attribute promotional/opinion claims; final self-check of numbers/units/dates/directions/attributions;
  titles may not use "comprehensive/overview/draft/deep dive/introduction"; rough writing described via an author
  and situation.
- Execution wrapper: "background material ... your readers will never see it, so the page must not refer to it
  ('the text', 'the source', 'the article says')" at the top; "cover what the instructions assign; do not recap".

Metrics: within 3.43 (first 8 views), cross-doc cos 0.119 = real, batch Vendi(48) 37.1 vs real 38.3.
Plan cost: 7-10k tokens per plan call regardless of n (24k/doc with up to 4 calls) -> move to 24 prompts per call.
Source-reference rejections rose to 17% (P(ref) 0.22 when the prompt itself says "the text", 0.12 otherwise; r1's
short end-of-message wrapper had 0.03): quoting the banned phrases in the wrapper primes them.

Readers (across + 3 within, 5 docs each with all views):
- Across 3/5 (up from ~2.5): openings now vary like real pages (casual first-person blogs, diaries, forum handles,
  podcasts, Q/A); but ~45% neutral encyclopedic voice; classroom genres over-represented (110/1390 titles);
  commercial/local-business pages missing; some offline types ("Lab Notebook", "Email chain").
- Within distinctness 2-2.5/5: ~35-45% of views still essentially full retellings (writer pads, esp. long views
  and short sources). Later plan calls find real new niches for long docs, but call 4 of 48-view docs repeats
  call 1 or writes finished drafts that get copied.
- Fidelity 2.5/5: ~20-30% of views with a false fact (down from 40-55%); planner errors persist (asking for
  "exact figures" from empty tables -> invented numbers; misattributed studies); writer completes truncated
  sentences, adds materials ("bronze"), reverses orders.
- Rough writing 1.5-2/5: asked-for roughness comes back as clean prose with a casual opener.
- No example leakage in any reader's grep.

## r4 (plan_prompt_r4.txt, examples_r4.txt, arv2_plan_r4.py) -- run through the real harness
Generated by `simplescale submit ... --handler simplescale.harnesses.multisample:active_reading_v2` on the same 48
docs (run dir runs/smoke48-arv2-r4; the shared interactive server was preempted). Changes vs r3:
- 24 prompts per plan call (plan cost ~7-10k tokens per call regardless of n).
- Planner: never paste source sentences / write the page (a prompt is an instruction, not a draft); quote only a
  few words; no "exact figures" unless printed; each page gets a concrete author/venue with a point of view; each
  prompt states what to leave out; ignore empty tables and cut-off endings; concrete rough-writing features.
- Forms: classroom genres down-weighted; commercial/local pages, magazine features, advice columns, essays added.
- Execution wrapper: rules first, author/readers sentence last, no quoted banned phrases.
- Plan sections that are mostly pasted source text (8-gram copy > 35%) or > 300 words are dropped.

Metrics: within 3.51 (best), cross-doc 0.117 < real 0.120, batch Vendi(48) 37.2 vs real 38.4. 564 tokens/view,
16.4k tokens of views per doc (kept 13.2k = 7.2x the source tokens with 19% rejected), 32.8k generated tokens/doc
including plans.

Exec-only wrapper A/B on 400 r4 prompts (source_reference rejections): r4 wrapper 14.0%; r1-style "Source
document" wrapper 18.5%; same + prompt sanitizing 18.0%; "You are writing a web page. Everything you know about
the topic is in these notes" + sanitizing + author sentence 8.8% (copy rejections 3%). A deterministic repair of
sentence-initial "The text notes that X" -> "X" (neutral reporting verbs only, hedges such as "suggests" kept and
rejected) recovers about half of the remaining hits.

Readers (across + 2 within):
- Across 3/5: register varies more (lowercase posts, rants, curt forum fixes, mailing-list posts); still ~48%
  neutral encyclopedic voice and a per-doc menu; commercial forms now forced onto unsuitable topics ("Product page on
  Lysenko's education"); style labels/"Simulation" in titles; few placeholders ("DATE: [Current Date]").
- Within: full retellings ~25-35% (down from 35-45%; long sources ~15%, sources < 400 words still mostly retell);
  "leave out X" obeyed in short views, ignored at >= 250 words.
- Fidelity: ~20-25% of accepted views with a false fact; new failure: scraped-away numbers ("THE NUMBER of ...")
  that the planner points at and the writer fills with invented figures; testimonial/"hypothetical interview" forms.
- Rough writing 2.5-3/5 (improved); planner format failure on one doc (24 sections written as page text).

## r5 = FINAL (plan_prompt_r5.txt, examples_r5.txt, arv2_plan.py r5) -- 400-doc pilot through the harness
Run dir runs/smoke400-arv2-r5 (manifest dclm.arv2smoke400r5.arv2: the 16 pilot docs, the 32 lab docs, then 352
fresh audit-sample docs). Changes vs r4:
- Views: clamp(ceil(words / 25), 8, 48) (min 8 instead of 12: fewer retellings for short sources).
- Planner: never point at figures that scraping removed ("THE NUMBER of ..."); no testimonials / customer stories /
  (hypothetical) interviews; a forced form is worse than an ordinary one; no style labels or "simulation" in titles.
- Forms: product/local-business/org pages down-weighted, customer review removed.
- Execution: "You are writing a web page. Everything you know about the topic is in these notes" framing; plan
  prompts sanitized ("the text" -> "it"); views repaired ("The text notes that X" -> "X") before the reject checks.
- Plan sections without instruction words (the planner writing page text) are dropped.

Pilot volume (Qwen3 tokenizer): 29.0 views/doc, 568 tokens/view, 16.5k view tokens/doc, 32.4k generated
tokens/doc incl. plans; kept/source tokens 9.17 with 11.2% rejected in the pilot.

Post-pilot (filter-only, re-scored on the pilot outputs; no regeneration):
- 3 of 11,605 plan prompts echoed the QuickCache example -> EXAMPLE_MARKERS: a prompt or view containing an
  example-specific name its source lacks is dropped (bad_prompt / reject "example_leak").
- Readers found "the notes say / as the notes record" (the notes framing moves rather than removes source
  references: a broad regex finds 21-24% of views mentioning the text/notes/source under every wrapper variant).
  Repair extended (clause verbs say/state/note/point out/add/confirm may drop "that"; object verbs
  mention/describe/list/show/... only before "that"; ", as the notes point out" and "described in the notes,"
  removed) and the reject regex extended to "the notes say/mention/...", "according to the notes", "information
  provided", "this source". Re-scored pilot: 83.5% kept (13.6% source_reference, 2.5% copy, 0.3% placeholder).
- To keep the 3C margin with the extra rejections: 23 words per view (est. ~19.7B kept tokens vs 17.19B needed).

### r5 embedding results (400 docs; batch = 256 docs, one random accepted view each, 20 batches)
arv2: within Vendi 3.42 (first 8 views), view-source cos 0.65, cross-doc cos 0.1228 (real 0.1228), batch Vendi
126.3 (real 140.9, direct_nothink 145.8, direct_think 139.4, mix 140.0, ideas 118.1). Token-weighted view choice:
arv2 130.8, mix 138.6, direct 145.5.

Diagnosis (analyze_emb.py / analyze2-4.py on rounds/r5/emb_all.npz):
- Length: batch Vendi of arv2 views <150 words 108, 150-300 114, 300-600 131. Re-weighting the existing views
  (drop <250 words, down-weight forum/comment/social) reaches at most ~132.5.
- Later plan calls are as diverse as the first (B=96: 63.8 vs 63.4).
- arv2 has one dominant direction (top-PC variance share 4.8% vs 2.4% for real docs) correlated with length and
  register (casual forum/blog vs reference), and real docs project to one side of it.
- The 256-batch Vendi mostly measures how much of its source's topic a view keeps: within arv2, views in the
  lowest to highest view-source-cosine quartile give 100.9 / 122.9 / 131.4 / 134.5; mix's lowest quartile (view-src
  0.60) gives 123.9, the same as arv2 at matched similarity (0.62 -> 122.9). Across arms: direct 0.86 -> 146,
  mix 0.77 -> 140, arv2 0.65 -> 126, ideas 0.56 -> 118. Focused views (the point of arv2: within-doc diversity)
  carry less of the document's topic, so the stop criterion "batch Vendi >= ~140" conflicts with within-doc
  diversity under this metric; at matched topic retention arv2 is not template-collapsed (unlike ideas' assistant
  scaffolding). Decision: stop iterating (5 rounds) and generate with r5.

### r5 readers (across 150 fresh docs; within 9 fresh docs, all views)
- Across 3/5: no dominant opening pattern ("This guide/page..." ~7%, question hooks ~6%, "Welcome to" 3%);
  ~45% neutral expository voice; genres: explainer ~38%, forum/Q&A/comments ~14%, blog/opinion/essay ~15%,
  how-to ~8%, news ~7%, letters/newsletters ~7%, quiz/FAQ/list/notes ~8%, commercial ~3%; lengths 23-1717 words.
  Main tell: references to "the notes"/"the text" (addressed afterwards by the extended repair + reject), few
  genre labels leaking, little real page chrome.
- Within: ~35-38% of accepted views essentially full retellings (short sources worst); fidelity: 28-43% of
  accepted views with >= 1 false/distorted fact when every view is checked line by line (mostly writer errors:
  relation reversals, part/property confusions, hardened hedges, world-knowledge padding; ~15-30% of errors
  planner-induced); rough writing still mostly polished (2/5); no example copying in outputs (1-2 planner traces,
  now filtered by EXAMPLE_MARKERS).

## Final configuration (production run runs/pool025-qwen35-9b-arv2)
plan_prompt_r5.txt + examples_r5.txt + arv2_plan.py (r5, 23 words/view) -> simplescale/src/simplescale/harnesses/
arv2.py (generated by build_harness_assets.py) used by multisample.active_reading_v2; manifest
runs/manifests/dclm.pool025.arv2.manifest.jsonl (1,503,775 pool docs; words_per_view 23, min 8, max 48, 24 prompts
per plan call, plan thinking on 16384, execution thinking off, T 1.0 / top_p 0.95 / top_k 20 / presence 1.5).
Markdown post-processing happens in the harness (clean_markdown: heading #s, bold/italic markers, horizontal rules,
"* " bullets -> "- "); rejected views keep their text with finish_reason "rejected:<reason>" so export drops them.
Estimated cost: ~31 views/doc, ~35k generated tokens/doc (plans ~40%), ~52B tokens total, ~2,500-3,000
H100-hours at 5-6k tokens/s/GPU; expected kept synthetic tokens ~19.5B (3C needs 17.19B).

## Production generation (runs/pool025-qwen35-9b-arv2), 2026-11-21 04:28-10:42 UTC
- 1,503,775 pool docs in 46,993 chunks of 32; 30.1 views/doc; 33.3k generated tokens/doc (plans + views), ~50.1B
  generated tokens total; 82.9% of views kept (13.8% source_reference, 2.9% copy, 0.2% placeholder, <0.1% other,
  0.02% length); kept views median 323 words (p10 90, p90 884, mean 410). (Sample of 900 shards / 28,800 docs.)
- ~3,640 H100-hours of worker time in total (estimate before launch: 2,500-3,000), including: the first 384-worker
  array dying on sglang's 300 s scheduler watchdog while hundreds of workers JIT-compiled flashinfer's top-k/top-p
  sampling kernel at once (fixed by resubmitting workers with --server-arg="--watchdog-timeout 1800"); a manager
  requeue at 05:42 (new lease epoch: all in-flight chunks became stale and were recomputed; moved the manager to
  explore); and the tail. Workers: 384 scale + 96 explore + up to 128 lowest, then +256 scale at 07:50; steady state
  ~4.9k tokens/s per worker (~104 running requests, capped by the 128 mamba slots).

## Export and training results
Export assets/data/ms_arv2_pool025_qwen3_tokens: 18,654,338,307 tokens (3C needs 17,190,257,792), 37,489,483 views of 1,503,699 sources.
Reference-document Vendi (analysis/multisample_vendi.py docs, arv2_vendi_ref.json): across 125.9 +- 0.9 (real 143.9, direct_nothink 147.0, direct_think 140.9, mix 141.9, ideas 117.8); within 3.38/8; across cos 0.124 (real 0.119); view-source cos 0.68.

| run | steps | dclm_bpb | paloma_bpb | arc_easy | arc_challenge | hellaswag | piqa | winogrande | sciq | mmlu | avg7 | mmlu_bpb | gsm8k_p32 | math_bpb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r4s_arv2 | 16392 | 0.8943 | 1.1774 | 0.657 | 0.374 | 0.433 | 0.668 | 0.544 | 0.758 | 0.308 | 0.535 | 1.131 | 0.283 | 0.904 |
| r4s_ideas | 16392 | 0.8978 | 1.1746 | 0.657 | 0.385 | 0.412 | 0.665 | 0.528 | 0.749 | 0.322 | 0.531 | 1.106 | 0.302 | 0.834 |
| r4s_mix | 16392 | 0.8955 | 1.1841 | 0.672 | 0.381 | 0.422 | 0.671 | 0.530 | 0.757 | 0.319 | 0.536 | 1.102 | 0.321 | 0.838 |
| r4s_direct_nothink | 16392 | 0.8963 | 1.1862 | 0.675 | 0.386 | 0.419 | 0.669 | 0.507 | 0.753 | 0.319 | 0.532 | 1.125 | 0.309 | 0.855 |
| r4s_direct_think | 16392 | 0.8953 | 1.1815 | 0.667 | 0.392 | 0.414 | 0.669 | 0.545 | 0.757 | 0.325 | 0.538 | 1.110 | 0.285 | 0.862 |
| real_p8 | 16392 | 0.8884 | 1.1681 | 0.641 | 0.364 | 0.415 | 0.671 | 0.522 | 0.745 | 0.309 | 0.524 | 1.111 | 0.286 | 0.882 |
| r4s_arv2_3x | 24588 | 0.8919 | 1.1795 | 0.669 | 0.398 | 0.444 | 0.674 | 0.534 | 0.773 | 0.322 | 0.545 | 1.112 | 0.279 | 0.914 |
| r4s_mix_3x | 24588 | 0.8944 | 1.1860 | 0.694 | 0.395 | 0.437 | 0.672 | 0.537 | 0.782 | 0.329 | 0.550 | 1.088 | 0.351 | 0.823 |
| r4s_direct_nothink_3x | 24588 | 0.8949 | 1.1889 | 0.686 | 0.387 | 0.434 | 0.674 | 0.526 | 0.764 | 0.331 | 0.543 | 1.104 | 0.309 | 0.842 |
| real_p12 | 24588 | 0.8809 | 1.1632 | 0.645 | 0.368 | 0.427 | 0.679 | 0.525 | 0.764 | 0.312 | 0.531 | 1.097 | 0.293 | 0.871 |

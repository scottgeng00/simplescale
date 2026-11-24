# mix16 / mix16plan pilot (2026-11-24)

Handler `simplescale.harnesses.multisample:prompt_pool` (assets + helpers in `harnesses/mix16.py`, regenerated from
`pilot/pool16.json` + `pilot/planner_prompt.txt` by `build_mix16.py`). Run through the real harness
(`simplescale submit`, Qwen3.5-9B, TP1, 2 explore workers per arm) on 256 pool documents (pool manifest rows
`int(i * 1503775 / 256)`, i < 256; manifests `runs/manifests/dclm.pilot256.{mix16,mix16plan}.manifest.jsonl`; run dirs
`runs/pilot256-qwen35-9b-{arm}`, code-checked copies `runs/pilot256-qwen35-9b-{arm}-checked2`). Stats:
`mix16_stats.py` -> `pilot_stats_{arm}.json`; Vendi: `pilot_vendi.py` -> `pilot_vendi.json`.

Settings: views thinking off, DEFAULT_PARAMS (T 1, top_p .95, top_k 20, presence 1.5, 4096 new tokens); code /
code_problem thinking on, 12288 new tokens, reasoning stripped; planner thinking off, T 0.7, top_p .95, top_k 20,
presence 0 (no penalty on a JSON score list), 1024 new tokens.

## Volume
- 8 views/doc for every document in both arms (2048/2048). Kept (export-eligible) views/doc: mix16 7.14, mix16plan 7.23.
- Kept Qwen3 tokens/doc: mix16 5,426, mix16plan 5,423 (mix arm, 8 views: 6,853/doc = 10.31B over the pool).
  **Projected 8-view export: ~8.16B tokens per arm vs 17.19B needed for 3C (2.1x short).** 16 views/doc -> ~16.3B
  (still ~5% short); 24 views/doc -> ~24.5B.
- Generated tokens/doc (views, reasoning included): mix16 12,971 (code reasoning: ~7k tokens per code view),
  mix16plan 6,815 (+~200 planner). Production estimate at ~4.9k tok/s/GPU: ~1,100 / ~600 H100-h per 8-view round.

## Per type (mix16 | mix16plan): views, kept %, kept tokens/view, main reject reason
| type | mix16 views | kept | tok | mix16plan views | kept | tok |
|---|---|---|---|---|---|---|
| article | 145 | 99.3% | 1025 | 180 | 100% | 1026 |
| discussion | 108 | 99.1% | 917 | 141 | 100% | 918 |
| faq | 123 | 100% | 996 | 143 | 99.3% | 910 |
| math | 130 | 93.1% (length 7%) | 962 | 19 | 94.7% | 1297 |
| narrative | 117 | 99.1% | 510 | 109 | 100% | 530 |
| table | 121 | 100% | 403 | 150 | 100% | 406 |
| tutorial | 119 | 99.2% | 1012 | 107 | 100% | 985 |
| wiki | 127 | 100% | 1016 | 179 | 99.4% | 1024 |
| diverse_qa | 126 | 56.3% (srcref 43%) | 639 | 182 | 49.5% (srcref 51%) | 633 |
| knowledge_list | 122 | 95.1% | 431 | 183 | 95.6% | 441 |
| explanation | 138 | 97.1% | 657 | 170 | 97.6% | 660 |
| forum_thread | 130 | 61.5% (srcref 38%) | 970 | 168 | 62.5% (srcref 36%) | 895 |
| entity_focus | 138 | 83.3% (srcref 17%) | 407 | 118 | 87.3% | 441 |
| persona | 132 | 97.7% | 680 | 175 | 94.3% | 706 |
| code | 139 | 69.1% (srcref 22%, exec 9%) | 911 | 9 | 77.8% | 964 |
| code_problem | 133 | 69.2% (srcref 25%, exec 4%) | 705 | 15 | 80.0% | 735 |

Source-reference rejections are real references, not regex false positives (checked by phrase): diverse_qa answers
"the text states / implies / does not specify", "based on the text"; forum posts "the document mentions / says",
"the prompt asks"; code comments "from the source material / source text / provided text" (a few code_problem hits
are legitimate "a provided text log" input descriptions, ~5% of code rejections). Re-scoring the pilot with the final
repair rules (adds "Yes, the text states that X" -> "Yes, X"): diverse_qa 42%, forum_thread 38%, code_problem 23%,
code 21%, entity_focus 15%, knowledge_list 5%, persona 4%, explanation 3%.

## Code execution (mix16 pilot, views that passed the reference filter)
Python 3.12 (system, no third-party packages) and gcc 13 (-std=c99) on CPU nodes; sandbox = temp dir, minimal env (no
proxy/credentials), RLIMIT CPU 11 s / AS 2 GB / FSIZE 16 MB / NOFILE 64, 10 s wall timeout, process-group kill,
LD_PRELOAD shim failing connect()/sendto(), static reject of subprocess/socket/os.remove/system()-style calls.
- Python: code 32/40 pass (7 crash: TypeError/KeyError/NameError, 1 unsafe); code_problem 29/31 (1 unfenced, 1 wrong lang).
- C: code 36/40 pass (4 compile errors); code_problem 26/28 (1 compile error, 1 crash).
- **Java: no javac on the cluster (only a Java 8 JRE)**: 65 Java views (~1/3 of code views) are unverified and
  currently kept (installing a user-space JDK was not permitted in this session).
- code_problem example I/O: extracted for ~95% of views, but stdout == stated output for only 31/54 (57%) Python+C;
  about half the mismatches are extraction artifacts (stated output wrapped in prose, "FOR INPUT: ..."), the rest
  wrong hand traces. Too unreliable to reject on: **code_problem only has to run**; match recorded in `code_check`.

## Planner (mix16plan)
256/256 replies parsed on the first try (no retries, no fallbacks). Score distribution (0/1/2) and chosen counts:
knowledge_list 4/6/246 (183), diverse_qa 3/20/233 (182), article 3/58/195 (180), wiki 4/19/233 (179), persona 0/19/237 (175),
explanation 2/31/223 (170), forum_thread 5/49/202 (168), table 19/84/153 (150), faq 17/92/147 (143), discussion 6/114/136 (141),
entity_focus 49/112/95 (118), narrative 28/151/77 (109), tutorial 81/58/117 (107), math 215/23/18 (19), code_problem 231/9/16 (15),
code 230/14/12 (9). The planner mostly removes math/code (0 for ~85-90% of docs) and keeps the prose types.

## Vendi (Qwen3-Embedding-4B, same 256 docs, kept views)
| | within (views/doc) | across-256 (mean of 10 draws) | view-source cos |
|---|---|---|---|
| real | - | 145.4 | - |
| mix (f200w run) | 2.45 (7.92) | 141.4 +- 1.1 | 0.780 |
| mix16 | 2.70 (7.14) | 143.0 +- 0.4 | 0.749 |
| mix16plan | 2.20 (7.23) | 142.8 +- 0.9 | 0.810 |
mix16 is more diverse within a document than mix despite fewer kept views; mix16plan less (prose-heavy picks,
closer to the source).

## Production
Round 1 (8 views/doc, seed 0) launched for both arms: `runs/pool025-qwen35-9b-{mix16,mix16plan}` on
`runs/manifests/dclm.pool025.{mix16,mix16plan}.manifest.jsonl` (same 1,503,775 docs as mix/arv2).

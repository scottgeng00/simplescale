# basegen smoke test + reader review (2026-11-21)

Generator: Qwen3.5-9B-Base, raw prompt `## Document 1\n\n{doc}\n\n## Document 2\n\n`, T 1.0, top_p 0.95,
presence_penalty 0.5, max_new_tokens 8192, stop `\n## Document` + EOS (handler
`simplescale.harnesses.multisample:base_continue`). Smoke: first 400 pool documents x 6 views
(runs/smoke-bg400-qwen35-9b-base-basegen; pairs in smoke_pairs.jsonl).

## Smoke 1 numbers (original stops/filters)
- ends: header 50.0%, eos 28.8%, length (8192 cap) 21.2%; source truncated (4096 tokens) 5.5%
- original filters (loop/salad/short) dropped 11.7% (loop 5.9, short 3.3, salad 2.5); 33% of the capped samples
- 2,758 Qwen3 tokens per kept view; capped samples were ~47% of all kept tokens

## Round 1 readers (7 readers)
Readers 1-5: one random kept view of 36 docs each (180 pairs). Totals:
- natural: yes 51, mostly 97, no 32 (18% not natural)
- relation to Document 1: same topic 41, related 56, same genre/other topic 30, unrelated 41, near-copy 6,
  continuation of doc 1 5, chat leak 1
- degeneracy: none 86/180 (48%); the problems were concentrated in the 31 capped samples (nearly all degraded:
  late drift, stitched "pages", reference-list spam, loops, word salad) plus leaked prompt structure
  ("# Document 3" with one `#`, bare "Document 3", "Based on the two Documents...", `<think>`, user/assistant turns)
- fabrication: none 31, some 86, heavy 63 (fake authors/DOIs/citations, worst in long academic-style outputs)
- NSFW/toxic: none 165, mild 14, yes 1
- verdicts: 5/5 "acceptable only with filtering / problematic as-is"
Reader 6 (18 capped kept + 18 dropped): capped: 9 fluent, 6 degenerate in the last 30-50%, 3 bad throughout;
filter precision 17/18 (one salad false positive: long archaic sentences).
Reader 7 (openings, 400 generations): 46% same topic, 30% same genre/template, 25% unrelated; ~4% restart by copying
the source; academic/institutional documents 12-15% vs ~2.5% of sources; markdown headers in 17% vs 0% of
sources; 197/200 view pairs clearly different; ~2% non-English.

Quotes: "Taken together, the stories in Documents 1 and 2 paints a clear picture" (prompt leak);
"Figure 341: pillow / Figure 342: pillow / Figure 343: pillows" (capped list loop);
"Inside the large pot on Ape Atonu, recently snail baked crocodile skin skin has been made." (late word salad);
"user / Hello! Please edit and merge information from Document 1 into Document 2" (chat leak).

## Fixes
1. Decoding A/B: + min_p 0.05 (same 400 docs; runs/smoke-bg400-minp005-...) was much WORSE: loops 18%, copies of
   the source 16%, kept 62% vs 85% -> rejected; original decoding kept.
2. Extra stops: `\n# Document`, `\n### Document`, `\n**Document`, `\nDocument 3`; stop tokens <|im_start|>/<|im_end|>.
3. Post-processing: cut at any leaked `Doc[ument] N` line, `<think>`, `<|...|>`; capped samples keep their first
   half (cut at a paragraph/sentence boundary; roughly the 4096-token regime of the original sweep) and need
   >= 150 words; new drop reasons: copy (> 50% of 8-grams from the source), script (> 30% non-Latin letters, Latin
   source), repeat (a >= 8-word sentence 4+ times), gibberish (all-caps words > 25% or vowel-less words > 3%;
   thresholds above the max / p99.9 of 3.9k real DCLM docs); CJK sentence punctuation counted by the salad check.
Re-applied offline to smoke 1: kept 86.4% (copy 3.5, short 3.9, loop 2.5, salad 1.9, repeat 1.2, gibberish 0.5,
script 0.3); 1,795 Qwen3 tokens per generated view (2,078 per kept view).

## Round 2 readers (3 readers x 36 kept samples after the fixes; 12 trimmed capped samples each)
- would drop: 23/108 (21%): trimmed capped 13/36 (36%), others 10/72 (14%)
- natural: yes 33, mostly 53, no 22; no near-copies left; no loops in 2 of 3 readers; 1 chat-style leak
  ("Final Answer", \boxed) and 1 "# Doc 2" leak (now also cut)
- remaining issues: incoherent topic drift (~30%), heavy fabrication 46/108 (28/36 of the trimmed capped ones,
  mostly fake academic papers/monographs), occasional OCR junk / code soup
- verdicts: "acceptable"; "acceptable at 2:1 after a cheap extra pass, trimmed subset problematic";
  "acceptable for non-trimmed, problematic for trimmed"

Decision: launch with these filters (8 views/doc). The trimmed capped samples are the weakest bucket (~38% of kept
tokens) but the user asked to keep long documents; each sample records `end` and `raw_chars`, so this bucket can be
cut or excluded later without regenerating.

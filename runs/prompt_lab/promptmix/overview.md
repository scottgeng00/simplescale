# What others have tried: prompts for synthetic pretraining data (rephrasing / rewriting web text)

45 verbatim prompts from 13 works, plus each work's key findings. Full prompts in Part 2.
Files: ~/scottg/pretrain/simplescale/runs/prompt_lab/promptmix/{catalog.jsonl, catalog.md, papers/, finephrase/}

## Part 1: The works, and what they found

### Fixed prompt sets (the "static mix" family)

**WRAP: Rephrasing the Web (Maini et al. 2024)**, arXiv 2401.16380
- 4 styles: easy (toddler), medium (Wikipedia), hard (scholar), Q/A. Mistral-7B-Instruct, applied to C4 paragraphs.
- Medium + QA is best, and mixing with real data is necessary.
- The template that later work copies. Its medium prompt is the "wiki" rephrase in Nemotron-CC and in our own mix.

**Nemotron-CC (NVIDIA 2024)**, arXiv 2412.02595
- 5 prompts: diverse_qa_pairs (yes/no, open-ended, MCQ, comparison, reading-comprehension, problem-solving), distill, extract_knowledge, knowledge_list, and the WRAP wiki rephrase (applied to low-quality documents).
- High-quality documents get the 4 knowledge prompts; low-quality ones get the wiki rephrase.
- BeyondWeb notes this mix is more diverse than QA-only WRAP.

**FinePhrase (HuggingFaceFW 2026)**, huggingface.co/spaces/HuggingFaceFW/finephrase
- 333 training runs over 90 configurations: 1.7B student trained on 21B tokens, generators mostly Gemma-3-1B.
- Re-ran WRAP-era, Nemotron, REWIRE and BeyondWeb prompts, plus 9 new "format" prompts. Our mix arm is 7 of these 9 plus WRAP wiki.
- Results:
  - Only table, math, faq and tutorial clearly beat DCLM. These restructure the content; prompts that only paraphrase don't.
  - Each prompt has its own benchmark profile. Table gives the biggest ARC gain (+7.5pp). Math is the only prompt that moves GSM8K (+1.5pp).
  - Every synthetic prompt loses to DCLM on HellaSwag and PIQA. Rephrasing trades commonsense ability for knowledge.
  - Mixing prompts, generator families, or both did not compound: "performance averages rather than compounds." But with enough prompts mixed, the source data was no longer needed, which no single prompt achieved.
  - Generator size barely matters (1B is enough for most prompts). 60-80% synthetic is best. Typos in the prompt don't hurt.
  - Open questions they list include repeated data: "can you repeat data more often if repetitions are rephrased?" That is our setting.

**BeyondWeb (DatologyAI 2025)**, arXiv 2508.10975
- Published only summarize and continue. The main recipe is not public.
- Plain summarization roughly matches Cosmopedia; information density is one mechanism.
- Continuation partly closes the data wall.
- **Diversity at scale (RQ5):** single-style data (QA-WRAP, Cosmopedia) plateaus. Their diverse recipe keeps improving even about 50x past Chinchilla at 1B, where the baselines start to overfit.
- Generator family barely matters, and returns diminish beyond 3B. Upsampling conversational style helps modestly.

**REWIRE / Recycling the Web (Meta 2025)**, arXiv 2506.04689
- One "guided rewrite" prompt: an expert-persona meta-reasoning plan followed by an improved version. Llama-3.3-70B, T=1.
- Targets documents that filters would otherwise discard, recycling them.
- FinePhrase found the cleaned-up and typo-laden versions perform the same.

**Pieler et al. 2024 (multilingual / quality levels)**, arXiv 2410.20796
- WRAP templates plus an "optimized QA" template and German/Italian/Spanish versions. T=0.7.

**Demystifying Synthetic Data (Kang et al., EMNLP 2025)**, arXiv 2510.01631
- Long Wikipedia-style rephrase, complete-coverage QA, and a 2-step textbook generator (outline, then chapter) for 4 audiences.
- About 30% rephrased synthetic is optimal. Textbook-style data at high ratios degrades.

**Kimi K2 (2025)**, arXiv 2507.20534 (prompts not published)
- Style- and perspective-diverse knowledge rephrasing. Long documents are rewritten chunk by chunk, with a fidelity check.
- SimpleQA results: raw text repeated 10 epochs scores 23.8; one rephrasing repeated 10 epochs scores 27.4; 10 different rephrasings for 1 epoch score 28.9. Direct evidence that diverse rephrasings beat repetition.
- Math is rewritten in a "learning-note" style following SwallowMath.

**Domain rewriters**
- MegaMath (arXiv 2504.02807): web-to-math refine prompt. In QA generation, an ELI5 framing plus "information completeness" works best.
- SwallowMath / SwallowCode (arXiv 2505.02881): math rewrite that strips boilerplate and adds steps; self-contained code rewrite.

### Per-document planned prompts (the "dynamic mix" family)

**MGA: Massive Genre-Audience reformulation (Hao et al. 2025)**, arXiv 2502.04235
- **Closest to a dynamic prompt mix.** Stage 1: for each document, generate 5 (genre, audience) pairs, each with a 2-3 sentence description. It explicitly asks for diverse audiences, including uninterested ones and people who dislike the content.
- Stage 2: one reformulation per pair. A small model is distilled to do this at scale (770B tokens from 195B, about 3.9x).
- Ablation: "limited consistency" beats both a strict polishing prompt (too close to the source) and a relaxed creative one (drifts and hallucinates). This is the fidelity vs diversity dial.

**EntiGraph: Synthetic Continued Pretraining (Yang et al. 2024)**, arXiv 2409.07431
- Extract entities, then write about every entity pair and sampled triples (rephrase the document around each entity, then analyze how they interact).
- Diversity comes from the combinatorial structure, not the prompt wording. Gains scale log-linearly where paraphrase (WRAP-style easy/medium/hard baselines) saturates quickly.
- Each generation is anchored to a different slice of the content. This is the fix our arv2 audit pointed to: arv2 writers always retell the whole source.

**Active Reading (Lin et al. 2025)**, arXiv 2508.09494
- The generator writes its own per-document learning strategies (task-agnostic, or task-specific such as "studying for trivia"), then applies each one to the full document.
- Beats paraphrase and synthetic QA on fact learning.
- Our arv2 is a heavily modified descendant: forms, lengths and quality are random slots, the planner writes web-document prompts, and the writer sees the source as "notes".

## Takeaways for our dynamic-prompt-mix idea
1. Restructuring formats (QA, table, tutorial, math, knowledge lists) beat paraphrase on knowledge benchmarks. Real web text keeps the commonsense benchmarks (HellaSwag/PIQA), so the real-data share matters.
2. Diversity's payoff shows up under repetition and overtraining (Kimi K2 rephrase-vs-epochs, BeyondWeb past 50x Chinchilla). Not in FinePhrase's single-pass 21B setting. Our data-constrained study is the regime where it should matter.
3. Per-document dynamic mixes exist in two forms:
   - choose the prompt by document (MGA's genre x audience, Active Reading's strategies);
   - choose the content slice by document (EntiGraph's entity combinations).

   Nobody published both together. That combination is the v3 opportunity: a planner that picks applicable prompt types from this catalog and gives each a different content slice.
4. Fidelity dial: MGA's "limited consistency" is the sweet spot. Too strict collapses to paraphrase; too loose hallucinates.
5. Prompt types missing from our current 8-prompt mix:
   - Nemotron knowledge_list / extract_knowledge and multi-type diverse QA;
   - FinePhrase commentary and explanation;
   - entity-focused rewrites;
   - an audience or difficulty axis (WRAP easy/hard, MGA audiences);
   - continuation;
   - code rewrites.
6. Not covered here: Cosmopedia's templates (on GitHub, fetch was blocked), plus Kimi K2, BeyondWeb's main recipe and MegaMath QA, which didn't publish prompts.

# Part 2: Full prompt catalog

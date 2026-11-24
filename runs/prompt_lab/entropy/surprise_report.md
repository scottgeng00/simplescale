# Where is the surprise? Human web text vs Qwen3.5-9B-Base generations, scored by the same model

**Setup.** 32 random pool documents (filtered DCLM). Prefix = first 32 tokens; human continuation = next ≤1024 tokens (mean ~810); the 9B base model continues the same prefix for the same length under three decoding settings. Every continuation is scored teacher-forced by the **same 9B base model**: per-token NLL of the realized token, entropy of the next-token distribution (lower bound from top-100 logprobs). Generations are compared with the human text on the same number of positions. One sample per setting per doc; 32 docs → differences of a few hundredths are noise.

Decoding settings: **T 1 (pure)** = temperature 1, top_p 1; **top_p 0.95**; **production basegen** = top_p 0.95 + presence_penalty 0.5 (the setting used for the `r4s_basegen` training data).

## TL;DR

- **Mean surprise:** production basegen is slightly *harder* than human text (2.20 vs 2.08 nats/token); top_p 0.95 alone is slightly easier (1.97); pure sampling drifts far off (2.82).
- **Shape:** human text is **spikier**: more near-certain tokens (18% under 0.1 nats) *and* much bigger spikes (p99 ≈ 10 nats) — its information is concentrated in rare surprising tokens (names, numbers, specific facts, scraping artifacts). Generated text spreads **moderate surprise evenly**: fewer give-away tokens, no extreme spikes. Every concentration measure (CV, Gini, share of surprisal in the top 10% of tokens) is highest for human text.
- **Clustering:** weak everywhere (lag-1 autocorrelation ≈ 0.05–0.09). Human high-surprise tokens come in slightly more adjacent pairs (multi-token names / numbers); production generations have the most scattered surprise.
- **Calibration:** the model is *overconfident on human text* (actual NLL exceeds its predicted entropy by 0.29 nats/token); top_p 0.95 samples are nearly self-consistent (+0.08); the presence penalty adds surprise (+0.22).
- **Drift:** human and top_p text get easier as context accumulates; production stays flat (the penalty keeps pushing to new tokens); pure sampling gets harder (2.58 → 3.14 from first to last quarter).
- **Interpretation:** fabricated 'facts' are drawn from the model's moderately likely options, so they lack the high-surprise spikes that real facts carry. This is a plausible token-level account of why basegen data improves web-text bpb (it covers the broad moderate-surprise mass of web text) but not knowledge benchmarks (it lacks the fact-carrying spikes).

## Distribution of per-token surprise (NLL, nats)

| | human | T 1 (pure) | top_p 0.95 | production basegen |
|---|---|---|---|---|
| mean | 2.08 | 2.82 | 1.97 | 2.20 |
| median | 1.28 | 1.99 | 1.36 | 1.57 |
| p90 | 5.37 | 6.72 | 4.86 | 5.38 |
| p99 | 9.96 | 11.59 | 7.89 | 8.38 |
| tokens < 0.1 nats (near-certain) | 18.2% | 12.0% | 17.5% | 15.1% |
| tokens > 5 nats (big surprise) | 12.1% | 19.3% | 10.2% | 12.5% |

Pooled histogram (share of tokens per NLL bin):

| NLL bin | human | T 1 (pure) | top_p 0.95 | production basegen |
|---|---|---|---|---|
| 0–0.1 | 18.3% | 12.2% | 18.5% | 15.7% |
| 0.1–0.5 | 15.4% | 11.5% | 14.9% | 14.1% |
| 0.5–1 | 12.1% | 10.5% | 11.8% | 11.4% |
| 1–2 | 16.9% | 16.4% | 17.4% | 17.0% |
| 2–3 | 11.9% | 12.6% | 12.6% | 12.9% |
| 3–5 | 13.7% | 17.5% | 15.1% | 16.9% |
| 5–8 | 8.5% | 12.8% | 8.4% | 10.1% |
| 8–12 | 2.9% | 5.4% | 1.1% | 2.0% |
| > 12 | 0.3% | 1.0% | 0.0% | 0.0% |

## Concentration (spikiness)

| | human | T 1 (pure) | top_p 0.95 | production basegen |
|---|---|---|---|---|
| coefficient of variation (std/mean) | 1.159 | 1.001 | 1.057 | 0.989 |
| Gini of per-token surprise | 0.579 | 0.525 | 0.546 | 0.529 |
| share of total surprise in top 10% tokens | 36.6% | 32.1% | 33.3% | 31.3% |

## Clustering

| | human | T 1 (pure) | top_p 0.95 | production basegen |
|---|---|---|---|---|
| lag-1 autocorrelation of surprise | 0.059 | 0.095 | 0.093 | 0.052 |
| burstiness of >5-nat tokens, P(hi,hi)/P(hi)² (1 = independent) | 1.32 | 1.18 | 1.24 | 1.08 |
| CV of 32-token window means | 0.251 | 0.258 | 0.298 | 0.239 |

## Calibration and drift

| | human | T 1 (pure) | top_p 0.95 | production basegen |
|---|---|---|---|---|
| model's predicted entropy (lower bound) | 1.80 | 2.17 | 1.89 | 1.98 |
| actual NLL − predicted entropy | +0.29 | +0.65 | +0.08 | +0.22 |
| mean NLL, first quarter → last quarter | 2.13 → 2.01 | 2.58 → 3.14 | 2.05 → 1.88 | 2.20 → 2.20 |

(The T 1 calibration gap is overstated: the entropy is a top-100 lower bound and T 1 has a long tail.)

## Example: the 8 most surprising tokens in one document

Human continuation begins: _6 the publishing house that in 1878 became Charles Scribner's Sons and in 1870 he began Scribner's Monthly, which in 1881 became the Century Magazine. His son, Charles Scribner, 1854-1…_


**human**

| NLL | context **[token]** |
|---|---|
| 10.3 | … was 2.94. ⏎  ⏎ In the**[ town]** |
| 9.4 | … in 1887. He was the**[ donor]** |
| 8.9 | …1 at the 2000 census.**[⏎⏎⏎]** |
| 8.0 | … of those age 65 or over. ⏎  ⏎  ⏎ **[External]** |
| 7.8 | …887. He was the donor of the**[ Princeton]** |
| 6.9 | … 65 or over. ⏎  ⏎  ⏎ External links ⏎  ⏎ **[Search]** |
| 6.9 | …, which in 1881 became the**[ Century]** |
| 6.6 | … Univ. Press building. ⏎ Scribner**[ is]** |

**production basegen** (same prefix)

| NLL | context **[token]** |
|---|---|
| 10.3 | … in the U.S. Writing for the town's**[ synagogue]** |
| 10.1 | … … Wikipedia ⏎  ⏎ Charles Scribner's Sons —**[ referring]** |
| 9.8 | … taken over by the firm of Scribner.**[ Significant]** |
| 9.7 | …en, and British surgeon Joseph Lister. For**[ Western]** |
| 9.2 | … book publisher in the U.S. Writing for the**[ town]** |
| 9.2 | … Progressive values ensured its wide acceptance. Scribner**[ celebrated]** |
| 9.0 | …ias and literary works aimed at immigrants. Introducing**[ copyrighted]** |
| 8.7 | …ner (d.1871) and**[ taken]** |

In this document the human spikes are partly scraping artifacts: the page stitches a Scribner biography onto a Wikipedia-style town page (census figures, "External links", "Search"), and named entities ("Princeton", "Century"). The generation's spikes are idiosyncratic word choices inside fluent prose ("synagogue", "celebrated", "copyrighted"): fabricated detail, not facts the model could not have known.

## Files
`/home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab/entropy/`: `run.py` (generation + scoring; `DUMP=` saves per-token arrays), `tokens_9b.json` (per-token data), `spikiness.py` → `spikiness_9b.json`, `result.json` (earlier 9B/4B/cross-scoring summary).
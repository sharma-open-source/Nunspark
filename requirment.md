The NunSpark is the package for run the high-end llm modals on the low-end device.

Idea:

The core idea is split the models into small small pieces and load only require weight to gpu get the outputs.

-------
4th Jun 2026: update

In reality this was not possible at all; tried with 70b modal on 16gb ram m4 system;
got around ~0.6-8 token/s which means for 500 token we need to wait around 15 minuts.

operation success but patient died :-)
------

5th Jun 2026: update

Before droping this expirment I'm going to try the another approach called speculative streaming with Disk-streaming method; So far speculative streaming has been expormented with actual GPU, here we are trying that with disk 

----

## Notes from analysis with AI:

1. The inverted-economics thesis is real, not hand-waving. As K grew, M rose (2.46→5.0) while resident tok/s fell (11.9→8.4). Those opposite slopes are the finding: the rest of the market keeps K small because deeper speculation costs compute they're bound by; in the disk-streaming regime that cost is hidden behind the weight read, so you ride M up to its ceiling. Nobody is tuning speculative decoding for the disk-bound regime — that's the open lane.

2. The multiplier lands the mid-tier into usable territory:

| Target (4-bit) | Naive stream | × M(~5) | × M(~3, conservative) |
|----------------|--------------|---------|-----------------------|
| 70B            | 0.08         | 0.40    | 0.24 tok/s            |
| 30B            | 0.17         | 0.85    | 0.51 tok/s            |
| 26B            | 0.20         | 1.00    | 0.60 tok/s            |

A 26–30B-4bit at ~0.6–1 tok/s on a 16 GB Mac — that's the "aha." Plain streaming (and llama.cpp's mmap) gives ~0.17. This is a 3–5× exact-output speedup (speculative decoding is mathematically lossless — identical distribution to running the target alone).

The honest caveats (so you don't over-believe it)

This was an 8B target. The real 70B + 0.6B-draft gap is larger, so expect M lower — likely the M3 column, not M5. I used a Qwen3 pair because it shares vocab and the 8B was cached; the real Llama-3 test needs a Llama-3.2-1B draft (128k vocab) against your 70B.

These prompts are "easy" (code, factual, summarization — predictable). Creative or out-of-distribution text accepts less. Real-world M is somewhere below these.

Vocab lock-in: draft and target must share a tokenizer. That constrains which (draft, target) pairs exist — fine for major families, awkward for exotic models. This actually narrows the "max compatibility" story: you win on the families that have a small sibling.

The 70B even at M=5 is still 0.4 tok/s — the win is concentrated in the 13–30B band, not the hero 70B number.

What this means for the experiment

Plain weight-streaming: still no moat vs llama.cpp mmap. Confirmed.

Speculative streaming with deep-K: genuine, measured, ~3–5× lossless, and specifically unexploited by current tools. This is the thing worth building, and it makes the 26–30B tier you asked about actually usable.

The truly confirmatory next step is the real pair — Llama-3.2-1B draft against your streamed 70B — but that requires wiring draft_model into NunSpark's streaming engine, i.e. it's the first build increment, not a throwaway probe. So the decision in front of you is: build that increment (deep-K speculative decode on top of the streaming engine) and measure the real pair

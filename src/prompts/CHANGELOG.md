# Prompt Changelog

A prompt-engineering logbook. Unlike git history and `locks.json` (which record
*that* a prompt changed, plus its version + hash), this file records the two
things those can't: **why** a prompt changed and **what it did to the eval
numbers**. Every `version` bump in `registry.py` should add an entry here.

Scores come from the Ragas harness (`tests/eval/ragas_eval.py`, results in
`tests/eval/results/latest.json`); always note the judge model, since scores are
only comparable across runs that used the same judge.

**Entry template** (newest version first within each prompt section):

```
### <prompt_id> v<N> — <YYYY-MM-DD> (<hash>)
- Why: <the problem in the previous version, or rationale for the change>
- Change: <what actually changed in the template>
- Eval delta (<judge model>): <metric> A -> B, <metric> A -> B (or "baseline" / "not yet measured")
- Baseline: <commit / results file the numbers came from>
```

---

## `query_classifier`

### v1 — 2026-06-15 (`e346c7694d29`)
- Why: Initial registry capture (verbatim from `search_workflow.classify_query`).
- Change: None — established the versioned baseline. No behavior change.
- Eval delta: baseline (this prompt isn't scored directly by Ragas; it routes
  simple vs. complex search).
- Baseline: n/a

## `query_expansion`

### v1 — 2026-06-15 (`bf390c3aaa4f`)
- Why: Initial registry capture (verbatim from `search_workflow.complex_search`).
- Change: None — established the versioned baseline. No behavior change.
- Eval delta: baseline (affects retrieval recall on complex queries; not scored
  in isolation).
- Baseline: n/a

## `result_synthesis`

### v2 — 2026-06-16 (`c7be25899bcd`)
- Why: The `llama3.1:8b` baseline showed perfect source retrieval but weak
  answer quality: verbose answers discussed distractor documents, sometimes
  contradicted themselves with "I couldn't find..." language, and missed facts
  near the end of truncated contexts.
- Change: Tightened synthesis instructions to answer directly, ignore
  irrelevant documents, avoid unsupported facts, and cite only supporting
  document numbers. Paired with workflow context selection that sends a smaller
  selected context set to synthesis.
- Eval delta (`llama3.1:8b`): faithfulness 0.572 -> 0.722,
  answer_relevancy 0.624 -> 0.896, context_precision 1.000 -> 1.000,
  context_recall 0.900 -> 0.772. Retrieval stayed at hit_at_1 1.000,
  hit_at_3 1.000, mrr 1.000.
- Baseline: `tests/eval/results/latest.json` before/after v2
  (`result_synthesis` v1 `33b2e391618f` -> v2 `c7be25899bcd`)

### v1 — 2026-06-15 (`33b2e391618f`)
- Why: Initial registry capture (verbatim from `search_workflow.synthesize_results`).
- Change: None — established the versioned baseline. No behavior change.
- Eval delta: baseline. Provisional scores with the `llama3.2:3b` judge:
  faithfulness 1.000, answer_relevancy 0.869, context_precision 1.000,
  context_recall 0.950. Known issue this prompt should fix in v2: hedged
  "I couldn't find..." answers even when the fact is in context.
- Baseline: `tests/eval/results/latest.json` (judge `llama3.2:3b`, provisional)

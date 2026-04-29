# BioLark GSC Error Analysis Prompt

## Purpose

This prompt is designed to be sent to a strong LLM (e.g., GPT-4o, Claude Opus, Gemini Ultra) repeatedly — one invocation per document sample — to categorize the root cause of each false positive (FP) and false negative (FN). Responses are structured so they can be aggregated across the full 228-sample corpus to estimate true error distributions.

---

## Background: Pipeline Description

The system is a three-stage HPO phenotype mining pipeline applied to biomedical text (abstracts and case reports). The pipeline produces a set of HPO codes per document. Evaluation is exact-match against gold-standard HPO annotations from the BioLark GSC benchmark.

### Stage 1 — Extraction

A retrieval-augmented LLM extracts raw phenotype spans from the text. The text is split into sentences. For each sentence:

1. The top-5 most similar HPO terms are retrieved via embedding similarity (MedEmbed-small).
2. The LLM is called with this system prompt:

> *"You are a rare disease expert with extensive medical knowledge. Carefully review every sentence of the clinical passage to identify terms related to genetic inheritance patterns, anatomical anomalies, clinical symptoms, diagnostic findings, lab test results, and specific conditions or syndromes. Return the extracted terms in a JSON object with a single key 'findings'…"*

3. And this user prompt per sentence:

> *"I have a clinical sentence: '[SENTENCE]'*
> *Here are some relevant HPO terms for context: [top-5 candidates]*
> *Extract all phenotype terms… MAKE SURE IT MATCHES EXACTLY AS IT APPEARS IN THE SENTENCE.*
> *Return as JSON: {"findings": [...]}"*

**Key policy**: Negated findings and family-history findings are both **included** (not filtered) at this stage.

**Key limitation**: No filter for non-human subjects (mouse studies), hypothesis/discussion sentences, or mechanism-only sentences.

---

### Stage 2 — Verification (MultiStageHPOVerifierV4)

Each extracted span goes through up to 7 sequential binary LLM calls. The top-20 HPO candidates are retrieved for each span.

**Step 1 — Direct phenotype check** (always runs):
> *"Is '[ENTITY]' a valid human phenotype? A valid phenotype must describe an abnormal characteristic or trait, not just a normal anatomical structure, physiological process, laboratory test, or medication. Respond YES or NO."*

**Step 2 — Lab test detection** (if Step 1 = NO):
> *"Does '[ENTITY]' contain a lab test name AND a numerical value/result? Respond YES or NO."*

**Step 3 — Lab value analysis** (if Step 2 = YES): Returns JSON with `is_abnormal`, `abnormality`, `direction`, `confidence`.

**Step 4 — Implied phenotype check** (if Step 1 = NO and not a lab value):
> *"Does '[ENTITY]' DIRECTLY AND UNAMBIGUOUSLY imply a specific phenotype? Be extremely conservative — only YES if the implication is clear and specific. Medications, procedures, and normal anatomy are NOT valid. Respond YES or NO."*

**Steps 5–7**: Extract and validate the implied phenotype via two more YES/NO LLM calls.

**Key limitation**: The system prompt says "not just a normal anatomical structure… or medication" but does **not** explicitly instruct the verifier to reject molecular mechanism terms (e.g., uniparental disomy, trisomy, contiguous gene deletion) or inheritance-etiology nodes as distinct from inheritance-pattern nodes.

---

### Stage 3 — Matching (RAGHPOMatcher, standard mode)

For each verified phenotype span, the top-20 HPO candidates are retrieved. The LLM is called with:

**System prompt**:
> *"You are a rare disease expert with extensive medical knowledge. Identify the most clinically appropriate and context-supported Human Phenotype Ontology (HPO) term for the given patient data. Prioritize specificity and clinical relevance. Return exactly one HPO ID (e.g., HP:0001250), no extra commentary."*

**User prompt**:
> *"Query: [ENTITY]*
> *Original Sentence: [SENTENCE]*
> *Context: [top-20 HPO candidates with labels]*"*

Before calling the LLM, an exact/fuzzy string match is attempted (>93% similarity). The LLM is only called if that fails.

**Key limitation**: When both a parent and a child (or two siblings) are present in the top-20 candidates, the LLM may choose the wrong granularity level.

---

## Your Task

You will be given a single document sample consisting of:
- The original text
- The system's predicted HPO codes (with labels)
- The gold-standard HPO codes (with labels)
- The breakdown into TPs, FPs, and FNs

For **each FP** and **each FN**, assign it to exactly one category from the lists below, and provide a one-sentence rationale referencing the text.

---

## FP Categories (choose one per false positive)

**FP-A: Etiology/mechanism term** — The HPO code belongs to a molecular mechanism, etiology, or chromosomal event subtree (e.g., uniparental disomy, trisomy, contiguous gene syndrome, de novo mutation). The text discusses this as a *cause*, not as a *patient phenotype*. The verifier accepted it because it is a real HPO term describing an abnormal state.
- *Counter-evidence to consider*: Are there TPs in this same document or similar documents where the gold standard does annotate mechanism/etiology HPO codes? If so, note it.

**FP-B: Inheritance mode term correctly rejected elsewhere** — The code is an inheritance pattern term (autosomal dominant, X-linked, etc.) that appears in the text but was not annotated in the gold standard for this document, even though such terms are TPs in other documents. This suggests annotation inconsistency rather than a system error.
- *Counter-evidence to consider*: Are there documents in the provided TP list where inheritance mode terms ARE annotated? If so, note it.

**FP-C: Extractor hallucination / retrieval contamination** — The predicted HPO concept has no plausible textual basis. The span either does not appear in the text or the predicted code does not match any reasonable reading of the extracted span. Likely caused by the retrieval-enhanced prompt surfacing an HPO candidate that the LLM erroneously grounded.
- *Counter-evidence to consider*: Is there a sentence in the text where a reader could plausibly (even if incorrectly) extract this concept?

**FP-D: Correct concept, wrong granularity** — The predicted HPO code is a sibling, parent, or closely related node to a gold-standard code. The system identified the right phenotype domain but landed on the wrong level of the ontology hierarchy (too broad, too narrow, or lateral).
- *Counter-evidence to consider*: Would the predicted code be a reasonable annotation in a different annotation schema? Does the text support either level of specificity?

**FP-E: Annotation noise / legitimate ambiguity** — The predicted code is clinically reasonable given the text, and a competent annotator could have included it. The FP may reflect annotation incompleteness rather than a system error.
- *Counter-evidence to consider*: Is there any explicit reason in the text to exclude this finding (e.g., explicitly negated, explicitly about a different individual)?

---

## FN Categories (choose one per false negative)

**FN-A: Named syndrome not decomposed** — The text names a syndrome or disease (e.g., "Angelman syndrome", "basal cell naevus syndrome") but does not list the specific phenotypic features. The gold standard annotates the cardinal features of that syndrome. The system cannot recover these without disease-to-phenotype expansion, which is disabled (`--decompose_compound=False`).
- *Counter-evidence to consider*: Does the text anywhere hint at or partially describe the missing phenotype?

**FN-B: Extractor missed explicit span** — The missing phenotype concept is explicitly and clearly stated in the text (not implied, not in a syndrome name), yet the system produced no corresponding prediction. This is an extraction recall failure.
- *Counter-evidence to consider*: Is the span buried in a complex sentence, negated, or in a discussion/hypothesis section that might reasonably confuse the extractor?

**FN-C: Matcher granularity mismatch** — The extractor likely found the right concept (there is a nearby FP that is a sibling/parent/child), but the matcher selected the wrong HPO node. The FN and a corresponding FP are ontologically close.
- *Counter-evidence to consider*: Is the gold-standard code actually more specific than what the text supports?

**FN-D: Obsolete or unmapped gold ID** — The gold-standard annotation uses an HPO ID that appears to be obsolete, retired, or lacks a clear label (e.g., listed as "HP:0006746" with no label). The system could not match to a code that isn't meaningfully in the current ontology.
- *Counter-evidence to consider*: Is there a current HPO code that is a reasonable successor to the obsolete ID, and did the system predict it?

**FN-E: Annotation noise / legitimate omission** — The gold-standard code refers to a concept that a reasonable reader would not extract from this text. It may be over-annotation, a curator inference, or a code included based on background knowledge not present in the text.
- *Counter-evidence to consider*: Can you find explicit textual support for the gold annotation?

---

## Output Format

Respond with a JSON object in this exact structure:

```json
{
  "doc_id": "<document ID>",
  "fp_analysis": [
    {
      "hp_id": "HP:XXXXXXX",
      "label": "<HPO label>",
      "category": "FP-A|FP-B|FP-C|FP-D|FP-E",
      "rationale": "<one sentence citing the text>",
      "counter_evidence": "<one sentence, or null if none>"
    }
  ],
  "fn_analysis": [
    {
      "hp_id": "HP:XXXXXXX",
      "label": "<HPO label>",
      "category": "FN-A|FN-B|FN-C|FN-D|FN-E",
      "rationale": "<one sentence citing the text>",
      "counter_evidence": "<one sentence, or null if none>"
    }
  ],
  "overall_notes": "<optional: one or two sentences on any cross-cutting observation for this document>"
}
```

Do not add any text outside the JSON object.

---

## Sample Input Format

```
DOC_ID: <id>

TEXT:
<full document text>

PREDICTED (system output):
<HP:XXXXXXX  Label>
<HP:XXXXXXX  Label>
...

GOLD STANDARD:
<HP:XXXXXXX  Label>
<HP:XXXXXXX  Label>
...

TPs (correct predictions):
<HP:XXXXXXX  Label>
...

FPs (predicted but not in gold):
<HP:XXXXXXX  Label>
...

FNs (in gold but not predicted):
<HP:XXXXXXX  Label>
...
```

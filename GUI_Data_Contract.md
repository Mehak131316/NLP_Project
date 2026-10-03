# GUI data contract: what the Assignment 1 notebook produces

Based on a real run of `Domain_Text_Analysis_Retrieval.ipynb` on **20 synthetic test contracts** (16 typed documents plus 4 synthetic scans). Row counts below come from that run and will change with your corpus. **Column names and file names will not.**

The sample files are packaged as `sample_results_SYNTHETIC.zip`. Build and test the GUI against them, but never report their numbers.

---

## 1. Read these three rules first

1. **Evaluation numbers are PROVISIONAL until you fill `annotations/relevance.csv`.** Every row of `evaluation_results.csv` carries a `Relevance Source` column (`PROVISIONAL` or `manual`). `pipeline_comparison.csv` does **not** carry this flag, so the GUI must read it from `evaluation_results.csv` and show a visible banner whenever any row is `PROVISIONAL`.
2. **Blank means "not reviewed", not "zero".** `ner_results.csv` has an empty `Correct?` column until you fill it. `Recall`, `F1` and `R@5` are NaN when a query has no relevant documents. Empty retrieval results come back as NaN in pandas unless you pass `keep_default_na=False` (see recipes).
3. **Some files are fixed-size, some depend on your corpus.** Fixed: `preprocessing_results` (4 rows), `stemming_lemmatization` (18), `pipeline_comparison` (9), `ngram_results` (5). Corpus-dependent: everything else.

---

## 2. Files at a glance

All files land in `./results/`. The 14 files the assignment requires are all present, plus three extras.

| File | Contents | Rows (sample run) | Row formula |
|---|---|---|---|
| `document_map.csv` | One row per document, with OCR provenance | 20 | n_docs |
| `document_statistics.csv` | Sentences, tokens, characters, vocabulary per document | 21 | n_docs + 1 (`TOTAL` row is last) |
| `preprocessing_results.csv` | Before and after preprocessing (Table A) | 4 | fixed |
| `tokenization_comparison.csv` | Gold tokenization vs 4 tokenizers (Table B) | 8 | len(`TOKEN_GOLD`) |
| `stemming_lemmatization.csv` | 3 stemmers and 3 lemmatizers (Table C) | 18 | fixed |
| `pos_tagging_results.csv` | Default vs custom POS on domain terms (Table D) | 28 | domain-term tokens in `POS_GOLD_TEST` |
| `ner_results.csv` | Entity inventory across the corpus (Table E) | 74 | unique (entity, type) pairs |
| `unigram_results.csv`, `bigram_results.csv`, `trigram_results.csv` | Full n-gram lists | 172, 332, 444 | unique n-grams |
| `ngram_results.csv` | Summary for 1 to 5-grams (Table F) | 5 | fixed |
| `bpe_results.csv` | BPE splits for sample words (Table G) | 28 | up to 8 common + 8 rare + 12 domain |
| `inverted_index.json` | Positional index, **Final pipeline only** | 172 terms | unique terms |
| `retrieval_results.csv` | Results for every query, all 3 pipelines | 45 | 3 x n_queries |
| `evaluation_results.csv` | Precision, recall, F1, P@K, R@K per query and pipeline | 45 | 3 x n_queries |
| `pipeline_comparison.csv` | Pipeline A vs B vs Final (Table H) | 9 | fixed |
| `ocr_report.csv` | OCR quality per scanned document | 4 | **only written if OCR was used** |
| `../annotations/relevance.csv` | Your hand-made relevance judgements | 15 | n_queries |

Pipeline names are constants everywhere: `Pipeline A`, `Pipeline B`, `Final Pipeline`.

---

## 3. Schemas

### `document_map.csv`
`doc_id, file, format, pages, ocr_pages, unread_scanned_pages, ocr_engine, ocr_mean_conf, ocr_repairs, ocr_unresolved_amounts`

```
D01  contract_01.pdf   pdf   1.0   0   0   (blank)          NaN    0   0
D17  scan_a_fullpage.pdf pdf  1.0   1   0   rapidocr 3.9.2   99.5   1   0
```
`pages` is a float with NaN for non-PDF, non-image files. `ocr_engine` and `ocr_mean_conf` are empty when no OCR happened. The `D##` id is assigned in filename order and is the key used by every other file.

### `document_statistics.csv`
`doc_id, sentences, tokens, characters, vocabulary`. The last row is `doc_id = TOTAL`: sums for the first three columns, and **corpus-wide unique** tokens (not a sum) for `vocabulary`. Exclude it from per-document charts.

### `preprocessing_results.csv`
`Measure, Before Processing, After Processing, Change %`, with rows `Documents`, `Tokens`, `Unique Tokens`, `Vocabulary Size`.
```
Tokens   4039   2068   -48.8
```

### `tokenization_comparison.csv`
`input, gold, nltk, spacy, custom, hybrid`. Each tokenizer cell is the token list joined by `" | "`. Split on that exact separator.
```
input:   The CTC is ₹12,00,000 per annum.
spacy:   The | CTC | is | ₹ | 12,00,000 | per | annum | .
hybrid:  The | CTC | is | ₹12,00,000 | per | annum | .
```
This is a real sample row. spaCy splits the rupee sign off the amount and the hybrid tokenizer keeps it whole. These are the fixed gold sentences, not your whole corpus.

### `stemming_lemmatization.csv`
`Word, Porter, Snowball, Lancaster, WordNet (no POS), WordNet (POS), spaCy`. Eighteen fixed words.

### `pos_tagging_results.csv`
`Word, Context, Gold, Default POS (spaCy), Custom POS (rules), Custom POS (ML), Default Correct, Rules Correct, ML Correct`. The three `*Correct` columns are booleans. `Context` is the sentence with tokens joined by spaces. Tags use the Penn Treebank set.

### `ner_results.csv`
`Entity, Predicted Type (custom), Predicted Type (default), Frequency, Correct?`. Sorted by `Frequency` descending. `Predicted Type (default)` is the literal string `(missed)` when spaCy's default model found nothing. `Correct?` is blank until you fill it. Custom labels include `MONEY, DATE, LAW, DURATION, DESIGNATION` plus spaCy's own.

The sample run already shows the kind of error you should review: `Employee` is labelled `GPE` by both pipelines.

### `unigram_results.csv`, `bigram_results.csv`, `trigram_results.csv`
`ngram, frequency, meaningful`. Sorted by `frequency` descending. `meaningful` is a boolean (does not start or end with a function word, frequency 2 or more). Tokens are lowercase lemmas from the Final pipeline, so `governing law` appears as `govern law`.

### `ngram_results.csv`
`N-Gram, Total, Unique, Meaningful (freq>=2), Top N-Grams`. Rows: `Unigram, Bigram, Trigram, 4-Gram, 5-Gram`. `Top N-Grams` is one string: `"shall (116); employee (85); ..."`. **Full 4-gram and 5-gram lists are not saved**, only this top 10. Use the parser in the recipes.

### `bpe_results.csv`
`S.No., Word, category, BPE Tokens, bpe_subwords, nltk, spacy, custom, GPT-2 BPE, gpt2_subwords`. `category` is `common`, `rare` or `domain`. Subwords and tokens are space-joined. **`GPT-2 BPE` and `gpt2_subwords` exist only if the GPT-2 tokenizer could be downloaded**, so they can be absent on an offline machine. Check `if "GPT-2 BPE" in df.columns`.

### `inverted_index.json`
```json
{ "non-compete": { "df": 8, "postings": { "D04": [76], "D05": [76], ... } },
  "notice":      { "df": 19, "postings": { "D01": [46, 57, 82], ... } } }
```
- Keys are **normalised terms** from the Final pipeline: lowercase, lemmatised, custom stop words removed, hyphenated terms and money amounts kept whole (`non-compete`, `₹12,00,000`).
- Positions index into the **filtered token stream**, not the original text. They cannot be used to highlight words in the document.
- Only the Final pipeline is saved. Pipelines A and B have no saved index.
- A typed word like `terminating` will not be found, because the index holds `terminate`. A GUI search box needs the notebook's query normaliser, which is not saved (see section 5).

### `retrieval_results.csv`
`Query ID, Query, Type, Pipeline, Retrieved Documents, # Results, Time (ms)`. `Retrieved Documents` is a space-separated list of doc ids **in ranked order** (best first, not sorted by id). Empty means zero results. `Type` is inferred from the query text. It is one of `Unigram, Bigram, Trigram, Boolean AND, Boolean OR, Boolean NOT`, or `<n>-gram phrase` for a plain query of four or more words (for example `4-gram phrase`). Match the last form with a pattern such as `str.endswith("-gram phrase")`, not an exact string. `# Results` always equals the length of the list.

### `evaluation_results.csv`
`Query ID, Query, Type, Pipeline, Relevant, Retrieved, Precision, Recall, F1, P@5, R@5, Time (ms), Relevance Source`. Same key as retrieval: (`Query ID`, `Pipeline`). **The `P@5` and `R@5` column names follow `CONFIG["TOP_K"]`**, so they become `P@3` if you change K. Find them with a regex, not a literal name.

### `pipeline_comparison.csv`
`Measure, Pipeline A, Pipeline B, Final Pipeline`, with rows `Token Count, Vocabulary Size, Meaningful N-Grams, Domain Terms Preserved, Negations Preserved, Preprocessing Time (s), Precision, Recall, F1`.
```
Domain Terms Preserved    10/14    14/14    14/14
F1                        0.994    1.0      1.0
```
**Every column is a string**, because two rows hold fractions like `14/14`. Cast the numeric rows yourself. `Precision`, `Recall` and `F1` are means over queries and inherit the PROVISIONAL status.

### `ocr_report.csv`
`doc_id, file, pages, ocr_pages, ocr_engine, ocr_mean_conf, ocr_repairs, ocr_unresolved_amounts`. **Not created when no page was OCR'd**, so handle a missing file. `ocr_mean_conf` is not a trustworthy quality signal: it stayed at 96 to 99 even when text was lost.

### `annotations/relevance.csv`
`query_id, query, relevant_docs`. `relevant_docs` is space-separated ids (`D01 D04 D07`). It is blank until you fill it.

---

## 4. Where each of the 15 GUI components gets its data

"Files" means it works today from the results folder. "Live" means it needs the notebook's functions running (section 6).

| # | Component | From files (precomputed) | Needs live code |
|---|---|---|---|
| 1 | Document upload / loading | `document_map.csv` lists the loaded documents | Extracting a new upload, including OCR |
| 2 | Document statistics | `document_statistics.csv` | Statistics for a new document |
| 3 | Tokenization | `tokenization_comparison.csv` (8 fixed sentences) | Tokenizing any text with `TOKENIZERS[name]` |
| 4 | Preprocessing | `preprocessing_results.csv`, `stemming_lemmatization.csv` | Running any text through `PIPELINES[name]` |
| 5 | POS tagging | `pos_tagging_results.csv` (default column) | `tag_nltk`, `tag_spacy` on any text |
| 6 | Custom POS tagging | `pos_tagging_results.csv` (rules and ML columns) | `tag_rules`, `TAGGERS[...]` |
| 7 | NER | `ner_results.csv` (whole corpus) | `nlp_ner(text)`, `nlp_custom(text)` |
| 8 | N-gram analysis | `ngram_results.csv` plus the 3 full lists | 4-gram and 5-gram full lists, per-document n-grams |
| 9 | BPE analysis | `bpe_results.csv` (28 words) | `bpe_tokens(text)` |
| 10 | Inverted index | `inverted_index.json` (browse a term's postings) | Index for Pipelines A and B |
| 11 | Query input | the 15 queries in `retrieval_results.csv` | Any free-text query |
| 12 | Query type selection | `Type` column | See note below |
| 13 | Document retrieval | `retrieval_results.csv` | `search(q, pipeline)` for free text |
| 14 | Pipeline comparison | `pipeline_comparison.csv` | none, fully precomputed |
| 15 | Performance / evaluation | `evaluation_results.csv`, `pipeline_comparison.csv` | none, fully precomputed |

**Note on component 12.** `query_type(q)` infers the type from the text; it is not an input. A "Query Type" dropdown must therefore build the query string: Keyword is plain text, Phrase wraps the text in double quotes, and Boolean AND / OR / NOT join two boxes with that operator. `tokenize_query` already understands quotes and the operators `AND`, `OR`, `NOT` (precedence NOT, then AND, then OR).

**Table I and Table J are not saved.** The notebook displays them but writes no file. Rebuild Table J with `evaluation_results.csv` grouped by `Pipeline` (recipe 4), and Table I from `retrieval_results.csv`.

---

## 5. What the notebook does NOT save (and what that blocks)

| Missing | Blocks | Fix |
|---|---|---|
| Cleaned text of each document | Showing a document, or any live analysis of an existing document without re-reading the file. OCR'd documents cost about 5 seconds per page to re-read. | Write `results/cleaned_text/D##.txt` |
| Per-document token streams for A, B, Final | Preprocessing view per document. Only the Final stream can be rebuilt from the index (recipe 3). | Save token lists as JSON |
| The trained models (custom NER, ML POS tagger, corpus BPE) | Every live analysis. The ML tagger retrains from the Treebank in about a minute at each start. | Persist them to disk |
| The query normaliser (`CORPUS_LEMMA`) and indexes for Pipelines A and B | Free-text search, and any A-versus-B search outside the 15 stored queries | Save the lemma map and both indexes |
| A provenance flag in `pipeline_comparison.csv` | Safe display of Table H | Read it from `evaluation_results.csv` for now |
| A run-info record (corpus size, config, timestamp) | Telling which run a results folder came from | Write `run_info.json` |

---

## 6. Live functions available in the notebook

These exist in the notebook's global namespace, verified by name. There is **no importable module yet**, so a standalone GUI cannot call them until they are exported.

| Call | Returns |
|---|---|
| `search(q, pipeline_name="Final Pipeline")` | `(ranked_doc_ids, elapsed_ms)`. Handles phrases, quotes, `AND`/`OR`/`NOT`. |
| `query_type(q)` | `"Unigram"`, `"Bigram"`, `"Trigram"`, `"Boolean AND"`, `"Boolean OR"`, `"Boolean NOT"`, or `"<n>-gram phrase"` (for example `"4-gram phrase"`) |
| `prf(retrieved_list, relevant_set)` | `(precision, recall, f1, p_at_k, r_at_k)`, with NaN recall, F1 and R@K when nothing is relevant |
| `TOKENIZERS[name](text)` | token list. Names: `nltk`, `spacy`, `custom`, `hybrid` |
| `PIPELINES[name](text)` | flat token list. Names: `Pipeline A`, `Pipeline B`, `Final Pipeline` |
| `PIPELINE_SENTS[name](text)` | one token list per sentence (used for n-grams) |
| `clean_text(text)`, `READERS[".pdf"](path)` | cleaned text; raw text from a file (OCRs scanned pages) |
| `tag_nltk(words)`, `tag_spacy(words)`, `tag_rules(words)`, `TAGGERS[name](words)` | list of Penn tags. `words` must already be a token list. |
| `nlp_ner(text).ents`, `nlp_custom(text).ents` | spaCy entities with `.text` and `.label_` |
| `bpe_tokens(text)`, `gpt2_tokens(text)` | subword list (`gpt2_tokens` is empty if GPT-2 was not loaded) |
| `ngram_counter(sent_docs, n)` | `Counter` of n-gram tuples |

---

## 7. Tested recipes

Each of these was run against the sample results.

```python
import pandas as pd, json, re
R = "results/"

# 1. Retrieval: empty results as [] (not NaN), ranked order preserved
retr = pd.read_csv(R + "retrieval_results.csv", keep_default_na=False)
retr["docs"] = retr["Retrieved Documents"].str.split()          # [] when nothing was found
# check: every row satisfies len(docs) == "# Results"

# 2. Parse the "Top N-Grams" string into [(ngram, count), ...]
def parse_top(s):
    return [(m.group(1), int(m.group(2))) for m in re.finditer(r"(.+?) \((\d+)\)(?:; |$)", s)]
ng = pd.read_csv(R + "ngram_results.csv")
top5 = parse_top(ng.loc[ng["N-Gram"] == "5-Gram", "Top N-Grams"].iloc[0])

# 3. Inverted index: load, look up a term, rebuild a document's token stream
idx = json.load(open(R + "inverted_index.json", encoding="utf-8"))
docs_with_term = list(idx["non-compete"]["postings"])           # ['D04', 'D05', ...]
def doc_tokens(index, doc_id):
    pos = {p: t for t, v in index.items() for p in v["postings"].get(doc_id, [])}
    return [pos[i] for i in sorted(pos)]

# 4. Provisional banner, and Table J (which the notebook does not save)
ev = pd.read_csv(R + "evaluation_results.csv")
provisional = (ev["Relevance Source"] == "PROVISIONAL").any()    # show a banner when True
k_cols = [c for c in ev.columns if re.fullmatch(r"[PR]@\d+", c)]  # P@K / R@K, whatever K is
table_j = (ev.groupby("Pipeline")[["Precision", "Recall", "F1", "Time (ms)"]]
             .mean().round(3).reindex(["Pipeline A", "Pipeline B", "Final Pipeline"]))

# 5. pipeline_comparison.csv is all strings: cast the numeric rows, keep the fractions
pc = pd.read_csv(R + "pipeline_comparison.csv").set_index("Measure")
f1 = pc.loc["F1"].astype(float)
domain_terms = pc.loc["Domain Terms Preserved"]                  # '14/14' style strings

# 6. Optional files: handle absence
import os
ocr = pd.read_csv(R + "ocr_report.csv") if os.path.exists(R + "ocr_report.csv") else None
bpe = pd.read_csv(R + "bpe_results.csv")
has_gpt2 = "GPT-2 BPE" in bpe.columns
```

---

## 8. Suggested scope for v1

**Tier 1, buildable today from files alone (no models, instant start):** components 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, plus 11 to 13 restricted to the 15 stored queries. Label these clearly as "precomputed on the loaded corpus", and show the provisional banner.

**Tier 2, needs the notebook's functions exported as a module and the models saved:** component 1 (upload and extract), live versions of 3 to 9 on arbitrary text, and free-text search (11 to 13). Without this tier the GUI cannot answer a viva question like "search for a term you did not prepare".

Tier 1 is a sound first milestone. Tier 2 is what makes it a working system rather than a results viewer.

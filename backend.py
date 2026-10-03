# Setup. Uncomment the install line on a fresh environment.

# Install required dependencies
#%pip install -U rapidocr onnxruntime pillow pymupdf python-docx
# Note: the package is called `rapidocr`. The older `rapidocr-onnxruntime` is a frozen legacy line.

import os, re, json, time, glob, math, unicodedata, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
import spacy
from spacy.language import Language
import fitz as pymupdf               # PyMuPDF is imported using 'fitz'
import docx                         # python-docx
from bs4 import BeautifulSoup
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

for name, mod in [("spacy", spacy), ("nltk", nltk), ("pandas", pd), ("pymupdf", pymupdf)]:
    print(f"{name:10s} {mod.__version__}")
import tokenizers, sklearn
print(f"{'tokenizers':10s} {tokenizers.__version__}\n{'sklearn':10s} {sklearn.__version__}")

# NLTK resources: these are the CURRENT resource names.
# "punkt" -> "punkt_tab" and "averaged_perceptron_tagger" -> "averaged_perceptron_tagger_eng" in NLTK 3.9+.
for res in ["punkt_tab", "stopwords", "wordnet", "omw-1.4",
            "averaged_perceptron_tagger_eng", "treebank", "universal_tagset"]:
    nltk.download(res, quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3_000_000
print("spaCy model:", nlp.meta["name"], nlp.meta["version"], "| pipes:", nlp.pipe_names)


import os

# Mount Google Drive


# Define the base path inside Google Drive
DRIVE_BASE_DIR = 'NLP_Assessment1-20261003T034029Z-1-001/NLP_Assessment1'

CONFIG = {
    "DATA_DIR": os.path.join(DRIVE_BASE_DIR, "data"),
    "RESULTS_DIR": os.path.join(DRIVE_BASE_DIR, "results"),
    "ANNOTATION_DIR": os.path.join(DRIVE_BASE_DIR, "annotations"),     # your hand-made gold files live here
    "BPE_VOCAB_SIZE": 2000,                # raise to 5000+ if your corpus is large
    "TOP_K": 5,                            # K for Precision@K / Recall@K
    "MIN_DOCS_WARNING": 15,
    # ---- OCR for scanned documents (RapidOCR) ----
    "OCR_ENABLED": True,                   # False = never OCR (scanned pages are left empty)
    "OCR_DPI": 200,                        # resolution PDF pages are rendered at before OCR (200 is a good default; 300 is slower)
    "OCR_MIN_LONG_SIDE": 2300,             # IMAGE files smaller than this (pixels, long side) are enlarged first; 2300 = A4 at ~200 dpi
    "OCR_MIN_CHARS_PER_PAGE": 25,          # a page with fewer text-layer characters AND an image is treated as a scan
    "OCR_WARN_CONF": 85,                   # warn when a document's mean OCR confidence (0-100) is below this
    "OCR_RESTORE_RUPEE": "safe",           # "off", "safe" (restore only unmistakable cases) or "context" (also amounts next to money words)
}

for d in ["DATA_DIR", "RESULTS_DIR", "ANNOTATION_DIR"]:
    os.makedirs(CONFIG[d], exist_ok=True)

def save(df, name):
    """Write a results table to the Google Drive results folder and return it for display."""
    path = os.path.join(CONFIG["RESULTS_DIR"], name)
    df.to_csv(path, index=False)
    print(f"  saved -> {path}")
    return df

# ---------------- OCR for scanned documents (RapidOCR) ----------------
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from PIL import Image, ImageSequence

# --- Repairing the rupee sign (see the Module 1 notes for the evidence behind these rules) ---
CURRENCY_BEFORE = re.compile(r"(?:₹|Rs\.?|INR|Re\.?|\$|USD|€|£)\s*$", re.IGNORECASE)
UNIT_AFTER = re.compile(r"\s*(?:shares?|units?|employees?|hours?|days?|months?|years?|sq|square|kg|km|copies|pages|words|%)\b",
                        re.IGNORECASE)
YEN_MISREAD = re.compile(r"[¥￥](?=\s?\d)")
SLASH_DASH_NUM = re.compile(r"(?<![\d,.])\d[\d,]*(?:\.\d+)?(?=/-)")                      # 5,00,000/-
LAKH_NUM = re.compile(r"(?<![\d,.])\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?(?!\d)(?!,\d)")      # 12,00,000  1,20,00,000
WESTERN_NUM = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)(?!,\d)")          # 15,000  (ambiguous)
MONEY_CUES = re.compile(r"\b(?:stipend|salary|ctc|fees?|penalty|penalties|bonus|compensation|remuneration|amount|"
                        r"sum|payable|pay|paid|rent|deposit|costs?|damages|rupees)\b", re.IGNORECASE)

def repair_ocr_text(text, mode=None):
    # Returns (repaired_text, number_of_rupee_signs_added_or_fixed, number_of_unresolved_candidates).
    mode = (mode or CONFIG["OCR_RESTORE_RUPEE"]).lower()
    if mode == "off":
        return text, 0, 0
    text, n_yen = YEN_MISREAD.subn("₹", text)
    has_marker = lambda pos: bool(CURRENCY_BEFORE.search(text[max(0, pos - 6):pos]))
    inserts, unresolved = set(), 0
    for m in SLASH_DASH_NUM.finditer(text):
        if not has_marker(m.start()):
            inserts.add(m.start())
    for m in LAKH_NUM.finditer(text):
        if not has_marker(m.start()) and not UNIT_AFTER.match(text, m.end()):
            inserts.add(m.start())
    for m in WESTERN_NUM.finditer(text):
        if m.start() in inserts or has_marker(m.start()) or UNIT_AFTER.match(text, m.end()):
            continue                     # already restored by an earlier rule, already marked, or not money
        if MONEY_CUES.search(text[max(0, m.start() - 40):m.start()]):
            if mode == "context":
                inserts.add(m.start())
            else:
                unresolved += 1
    for pos in sorted(inserts, reverse=True):
        text = text[:pos] + "₹" + text[pos:]
    return text, n_yen + len(inserts), unresolved

# --- RapidOCR engine: loaded once, lazily ---
_OCR = {"tried": False, "engine": None, "version": ""}

def get_ocr():
    if _OCR["tried"]:
        return _OCR["engine"]
    _OCR["tried"] = True
    if not CONFIG["OCR_ENABLED"]:
        print("OCR is switched off (CONFIG['OCR_ENABLED'] = False): scanned pages will be left empty.")
        return None
    try:
        from rapidocr import RapidOCR
        _OCR["engine"] = RapidOCR(params={"Global.log_level": "warning"})
        try:
            _OCR["version"] = _pkg_version("rapidocr")
        except PackageNotFoundError:
            pass
        print(f"OCR engine: RapidOCR {_OCR['version']} (loaded now because a scanned page was found)")
    except Exception as e:
        print(f"!! RapidOCR could not be loaded ({type(e).__name__}: {str(e)[:100]}).")
        print("   Scanned pages will be EMPTY. Fix:  pip install -U rapidocr onnxruntime")
    return _OCR["engine"]

def _ocr_lines(img, engine):
    # Run RapidOCR and rebuild reading order: group text boxes that share a line, then read top to bottom.
    res = engine(np.asarray(img.convert("RGB")))
    if res is None or res.txts is None or len(res.txts) == 0:
        return "", 0.0
    items = []
    for box, txt, sc in zip(res.boxes, res.txts, res.scores):
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        items.append((float(np.mean(ys)), min(xs), max(ys) - min(ys), txt, float(sc)))
    items.sort(key=lambda t: t[0])
    lines, cur, cur_y, cur_h = [], [], None, 0.0
    for y, x, h, txt, sc in items:
        if cur_y is None or abs(y - cur_y) <= 0.6 * max(cur_h, h):
            cur.append((x, txt)); cur_y = y if cur_y is None else (cur_y + y) / 2; cur_h = max(cur_h, h)
        else:
            lines.append(" ".join(t for _, t in sorted(cur))); cur, cur_y, cur_h = [(x, txt)], y, h
    if cur:
        lines.append(" ".join(t for _, t in sorted(cur)))
    return "\n".join(lines), 100 * float(np.mean([i[4] for i in items]))

def ocr_image(img):
    # OCR one PIL image. Returns (text, mean_confidence_0_to_100, rupee_repairs, unresolved_amounts).
    engine = get_ocr()
    if engine is None:
        return "", 0.0, 0, 0
    try:
        text, conf = _ocr_lines(img, engine)
    except Exception as e:
        print(f"  !! OCR failed on one page ({type(e).__name__}: {str(e)[:80]}); page left empty.")
        return "", 0.0, 0, 0
    text, repairs, unresolved = repair_ocr_text(text)
    return text, conf, repairs, unresolved

def page_to_image(page):
    pix = page.get_pixmap(dpi=CONFIG["OCR_DPI"], colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)

def enlarge_for_ocr(img):
    # Image files arrive at whatever resolution they have (PDF pages are re-rendered at OCR_DPI instead).
    # Small images lose or garble whole lines, so enlarge them to the minimum size; never shrink.
    long_side = max(img.size)
    if long_side >= CONFIG["OCR_MIN_LONG_SIDE"]:
        return img, 1.0
    scale = CONFIG["OCR_MIN_LONG_SIDE"] / long_side
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS), scale

OCR_LOG = {}     # path -> what OCR did for that file (feeds document_map.csv and the OCR report)

def _log(path, pages, ocr_pages, unread, confs, repairs, unresolved, first):
    OCR_LOG[path] = {"pages": pages, "ocr_pages": ocr_pages, "unread_scanned_pages": unread,
                     "ocr_engine": f"rapidocr {_OCR['version']}".strip() if ocr_pages else "",
                     "ocr_mean_conf": round(float(np.mean(confs)), 1) if confs else np.nan,
                     "ocr_repairs": repairs, "ocr_unresolved_amounts": unresolved, "first_ocr_page": first}

print(f"OCR configured: enabled={CONFIG['OCR_ENABLED']}, dpi={CONFIG['OCR_DPI']}, "
      f"rupee repair={CONFIG['OCR_RESTORE_RUPEE']!r} (RapidOCR is loaded only if a scanned page is found)")

SUPPORTED = (".pdf", ".txt", ".docx", ".html", ".htm", ".json", ".csv",
             ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def read_pdf(path):
    # Per PAGE: a page with almost no text layer but an embedded image is treated as a scan and OCR'd.
    # Mixed documents (some typed pages, some scanned) therefore work, and typed pages are never OCR'd.
    parts, n_ocr, n_unread, confs, repairs, unresolved, first = [], 0, 0, [], 0, 0, None
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        for pno, page in enumerate(doc):
            txt = page.get_text()
            if len(txt.strip()) < CONFIG["OCR_MIN_CHARS_PER_PAGE"] and page.get_images():
                if get_ocr() is None:
                    n_unread += 1
                else:
                    t0 = time.perf_counter()
                    txt, conf, r, u = ocr_image(page_to_image(page))
                    print(f"  OCR {os.path.basename(path)} page {pno + 1}/{n_pages}: "
                          f"{time.perf_counter() - t0:.1f}s, confidence {conf:.0f}")
                    n_ocr += 1; confs.append(conf); repairs += r; unresolved += u
                    first = pno if first is None else first
            parts.append(txt)
    _log(path, n_pages, n_ocr, n_unread, confs, repairs, unresolved, first)
    return "\n".join(parts)

def read_image(path):
    # Scanned images (PNG, JPG, TIFF, BMP). Multi-page TIFFs are read frame by frame.
    with Image.open(path) as im:
        frames = [f.convert("L").copy() for f in ImageSequence.Iterator(im)]
    parts, n_ocr, n_unread, confs, repairs, unresolved = [], 0, 0, [], 0, 0
    for k, f in enumerate(frames, start=1):
        if get_ocr() is None:
            n_unread += 1
        else:
            t0 = time.perf_counter()
            f, scale = enlarge_for_ocr(f)
            txt, conf, r, u = ocr_image(f)
            print(f"  OCR {os.path.basename(path)} frame {k}/{len(frames)}: "
                  f"{time.perf_counter() - t0:.1f}s, confidence {conf:.0f}"
                  + (f", enlarged x{scale:.2f} first" if scale > 1 else ""))
            parts.append(txt); n_ocr += 1; confs.append(conf); repairs += r; unresolved += u
    _log(path, len(frames), n_ocr, n_unread, confs, repairs, unresolved, 0 if n_ocr else None)
    return "\n".join(parts)

def read_docx(path):
    return "\n".join(p.text for p in docx.Document(path).paragraphs)

def read_html(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return BeautifulSoup(f.read(), "html.parser").get_text(separator="\n")

def read_json(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    def walk(x):
        if isinstance(x, str): return [x]
        if isinstance(x, dict): return [s for v in x.values() for s in walk(v)]
        if isinstance(x, list): return [s for v in x for s in walk(v)]
        return []
    return "\n".join(walk(data))

def read_csv(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    return "\n".join(" ".join(row) for row in df.values)

def read_txt(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

READERS = {".pdf": read_pdf, ".docx": read_docx, ".html": read_html, ".htm": read_html,
           ".json": read_json, ".csv": read_csv, ".txt": read_txt,
           ".png": read_image, ".jpg": read_image, ".jpeg": read_image,
           ".tif": read_image, ".tiff": read_image, ".bmp": read_image}

files = sorted(p for p in glob.glob(os.path.join(CONFIG["DATA_DIR"], "*"))
               if p.lower().endswith(SUPPORTED))
if not files:
    raise SystemExit(f"No documents found in {CONFIG['DATA_DIR']}. Add 15 to 30 contracts and re-run.")

raw_docs, doc_meta = {}, []
for i, path in enumerate(files, start=1):
    did = f"D{i:02d}"
    ext = os.path.splitext(path)[1].lower()
    try:
        text = READERS[ext](path)
    except Exception as e:
        print(f"  !! {os.path.basename(path)}: could not read ({type(e).__name__}: {e})")
        continue
    log = OCR_LOG.get(path, {})
    if len(text.strip()) < 200:
        why = ("scanned pages were left unread because no OCR backend is available"
               if log.get("unread_scanned_pages") else "very little extractable text")
        print(f"  !! {did} {os.path.basename(path)}: only {len(text.strip())} chars ({why}). Kept, but check it.")
    raw_docs[did] = text
    doc_meta.append({"doc_id": did, "file": os.path.basename(path), "format": ext[1:],
                     "pages": log.get("pages"), "ocr_pages": log.get("ocr_pages", 0),
                     "unread_scanned_pages": log.get("unread_scanned_pages", 0),
                     "ocr_engine": log.get("ocr_engine", ""), "ocr_mean_conf": log.get("ocr_mean_conf", np.nan),
                     "ocr_repairs": log.get("ocr_repairs", 0),
                     "ocr_unresolved_amounts": log.get("ocr_unresolved_amounts", 0)})

doc_map = pd.DataFrame(doc_meta)
save(doc_map, "document_map.csv")
if len(raw_docs) < CONFIG["MIN_DOCS_WARNING"]:
    print(f"\nWARNING: {len(raw_docs)} documents loaded; the assignment requires at least 15.")
doc_map

# Print the exact Google Drive directory path where contracts must be uploaded
print("Please upload your 15 to 30 contract files to this exact directory in Google Drive:")
print(CONFIG["DATA_DIR"])


import os

data_dir = CONFIG["DATA_DIR"]
if os.path.exists(data_dir):
    files_in_dir = os.listdir(data_dir)
    print(f"Found {len(files_in_dir)} item(s) in {data_dir}:")
    for item in sorted(files_in_dir):
        print(f" - {item}")
else:
    print(f"Directory does not exist: {data_dir}")

# OCR quality report. Only shows anything if at least one scanned page was found.
ocr_docs = doc_map[doc_map.ocr_pages > 0]
unread = doc_map[doc_map.unread_scanned_pages > 0]

if len(unread):
    print(f"!! {int(unread.unread_scanned_pages.sum())} scanned page(s) in {len(unread)} document(s) were NOT read "
          "because OCR is unavailable or switched off. Those documents are INCOMPLETE:")
    print(unread[["doc_id", "file", "pages", "unread_scanned_pages"]])

if ocr_docs.empty:
    print("No pages were OCR'd." if len(unread) else "No scanned pages found, so OCR was not needed.")
else:
    print(f"OCR was used on {int(ocr_docs.ocr_pages.sum())} page(s) across {len(ocr_docs)} document(s).\n")
    cols = ["doc_id", "file", "pages", "ocr_pages", "ocr_engine", "ocr_mean_conf", "ocr_repairs", "ocr_unresolved_amounts"]
    print(ocr_docs[cols])
    save(ocr_docs[cols], "ocr_report.csv")
    low = ocr_docs[ocr_docs.ocr_mean_conf < CONFIG["OCR_WARN_CONF"]]
    if len(low):
        print(f"!! Mean OCR confidence is below {CONFIG['OCR_WARN_CONF']} for: {', '.join(low.doc_id)}. "
              "Expect word errors; try a better scan or a higher CONFIG['OCR_DPI'].")
    if ocr_docs.ocr_repairs.sum():
        print(f"{int(ocr_docs.ocr_repairs.sum())} rupee sign(s) were restored or corrected automatically "
              f"(mode {CONFIG['OCR_RESTORE_RUPEE']!r}); RapidOCR often drops the sign at body-text size.")
    if ocr_docs.ocr_unresolved_amounts.sum():
        print(f"{int(ocr_docs.ocr_unresolved_amounts.sum())} amount(s) next to money words have NO currency marker and "
              "were left unchanged (ambiguous, e.g. '15,000'). Check them by hand, or set "
              "CONFIG['OCR_RESTORE_RUPEE'] = 'context' to restore them.")
    print("\nEvery amount found in OCR'd text. Compare each against the page image:")
    AMOUNT_LINE = re.compile(r"\d{1,3}(?:,\d{2,3})+|\bRs\.?\s?\d|INR\s?\d|₹")
    shown = 0
    for _, r in ocr_docs.iterrows():
        for line in raw_docs[r.doc_id].split("\n"):
            if AMOUNT_LINE.search(line) and shown < 12:
                print(f"  [{r.doc_id}] {line.strip()[:110]}"); shown += 1
    print("\nCaution: confidence stayed at 96 to 99 in testing even when a line was dropped or a rupee sign lost,")
    print("so a high score does not prove the text is right. Read the amounts above against the page image.")
    print("OCR errors flow into every later number in this notebook. State that in your report's limitations.")

KEEP_HYPHEN = {"non", "self", "co", "ex", "anti", "work", "part", "full", "year", "long", "short",
               "third", "cross", "on", "off", "in", "out", "pre", "post", "sub", "inter", "multi"}
ABBR_END = re.compile(r"\b(?:Pvt|Ltd|Co|Inc|No|Rs|Sr|Jr|Dr|Mr|Mrs|Ms|viz|Govt|Dept)\.$")

def _dehyphenate(m):
    # "termi-\nnation" -> "termination", but "non-\ncompete" -> "non-compete" (a real compound).
    return f"{m.group(1)}-{m.group(2)}" if m.group(1).lower() in KEEP_HYPHEN else f"{m.group(1)}{m.group(2)}"

def rejoin_wrapped_lines(lines):
    """Join hard-wrapped lines (common in PDFs, universal in OCR output) back into paragraphs.
    A line is joined to the next when it does not end a sentence AND either the next line starts
    lowercase or this physical line runs close to the document's typical line width."""
    lens = sorted(len(l) for l in lines if len(l) > 20)
    width = lens[int(0.9 * (len(lens) - 1))] if lens else 0
    out, last_len = [], 0
    for l in lines:
        if out and out[-1] and l:
            prev = out[-1]
            ended = bool(re.search(r"[.:;!?]$", prev)) and not ABBR_END.search(prev)
            if not ended and (l[0].islower() or last_len >= 0.8 * width):
                out[-1] = prev + " " + l; last_len = len(l)
                continue
        out.append(l); last_len = len(l)
    return out

def clean_text(text):
    """Conservative cleaning. Removes layout noise, keeps legal content."""
    text = unicodedata.normalize("NFKC", text)            # unify look-alike characters; keeps the rupee sign
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u2018\u2019]", "'", text)             # curly single quotes
    text = re.sub(r"[\u201c\u201d]", '"', text)             # curly double quotes
    text = re.sub(r"(\w+)-\n(\w)", _dehyphenate, text)     # re-join words hyphenated across a line break
    text = re.sub(r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", text)   # "Page 1 of 2"
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)            # bare page numbers
    text = re.sub(r"(?m)^\s*(?:\d+(?:\.\d+)*|[a-z]|[ivx]+)[.)]\s+", "", text)  # enumeration labels "1.", "(a)", "iv)"
    lines = [l.strip() for l in text.split("\n")]
    counts = Counter(l for l in lines if 0 < len(l) < 60)
    repeated = {l for l, c in counts.items() if c >= 3}    # running headers/footers repeated on many pages
    lines = [l for l in lines if l not in repeated]
    lines = rejoin_wrapped_lines(lines)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

docs = {did: clean_text(t) for did, t in raw_docs.items()}
corpus = "\n".join(docs.values())

sample = next(iter(docs))
print(f"Sample {sample}, first 400 characters after cleaning:\n")
print(docs[sample][:400])

# Document statistics. spaCy's parser gives sentence boundaries; tokens here are spaCy tokens
# excluding pure whitespace. These are the "Before Processing" numbers used in Table A.
spacy_docs = {did: nlp(t) for did, t in docs.items()}

rows = []
for did, sd in spacy_docs.items():
    toks = [t.text for t in sd if not t.is_space]
    rows.append({"doc_id": did,
                 "sentences": sum(1 for _ in sd.sents),
                 "tokens": len(toks),
                 "characters": len(docs[did]),
                 "vocabulary": len(set(toks))})
stats = pd.DataFrame(rows)

all_tokens = [t.text for sd in spacy_docs.values() for t in sd if not t.is_space]
summary = pd.DataFrame([{
    "doc_id": "TOTAL",
    "sentences": stats.sentences.sum(),
    "tokens": stats.tokens.sum(),
    "characters": stats.characters.sum(),
    "vocabulary": len(set(all_tokens)),
}])
stats_out = pd.concat([stats, summary], ignore_index=True)
save(stats_out, "document_statistics.csv")

print(f"\nNumber of documents:     {len(docs)}")
print(f"Number of sentences:     {stats.sentences.sum():,}")
print(f"Number of tokens:        {stats.tokens.sum():,}")
print(f"Number of characters:    {stats.characters.sum():,}")
print(f"Vocabulary size:         {len(set(all_tokens)):,}")
print(f"Average document length: {stats.tokens.mean():,.1f} tokens")
stats_out

from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer, sent_tokenize

regex_word = RegexpTokenizer(r"\w+")
BUILTIN = {
    "whitespace":       lambda t: t.split(),
    "nltk_word":        lambda t: word_tokenize(t),
    "nltk_wordpunct":   lambda t: wordpunct_tokenize(t),
    "nltk_regex_\\w+":  lambda t: regex_word.tokenize(t),
    "spacy":            lambda t: [tok.text for tok in nlp.tokenizer(t) if not tok.is_space],
}

def term_dictionary(tokens):
    return {t.lower() for t in tokens if re.search(r"\w", t)}

builtin_tokens, rows = {}, []
for name, fn in BUILTIN.items():
    toks = [tok for t in docs.values() for tok in fn(t)]
    builtin_tokens[name] = toks
    rows.append({"tokenizer": name, "tokens": len(toks),
                 "token_dictionary": len(set(toks)),
                 "term_dictionary": len(term_dictionary(toks)),
                 "reduction_%": round(100 * (1 - len(term_dictionary(toks)) / max(len(set(toks)), 1)), 1)})
ex1 = pd.DataFrame(rows)
ex1

# Exercise 2: inspect what the built-in tokenizers actually produce, to find what needs fixing.
def problems(tokens):
    return {
        "punctuation_only":     sum(1 for t in tokens if not re.search(r"\w", t)),
        "bare_currency_symbol": sum(1 for t in tokens if t in {"₹", "Rs", "Rs.", "INR"}),
        "hyphen_fragments":     sum(1 for t in tokens if t == "-"),
        "case_duplicates":      len({t for t in tokens if t.isalpha()}) - len({t.lower() for t in tokens if t.isalpha()}),
        "single_char_tokens":   sum(1 for t in tokens if len(t) == 1 and t.isalpha()),
    }

ex2 = pd.DataFrame({name: problems(toks) for name, toks in builtin_tokens.items()}).T
print(ex2)

probe = "The CTC is ₹12,00,000 per annum, subject to Section 27 and Clause 4.2(a); a non-compete applies w.e.f. 15 April 2026."
print("\nHow each built-in tokenizer handles one domain sentence:\n")
for name, fn in BUILTIN.items():
    print(f"{name:16s} {fn(probe)}")

from spacy.util import filter_spans

ABBREVIATIONS = r"(?:Pvt|Ltd|Co|Inc|No|Rs|Sr|Jr|Dr|Mr|Mrs|Ms|viz|approx|Govt|Dept)\.|w\.e\.f\.|i\.e\.|e\.g\.|a\.m\.|p\.m\.|B\.Tech|M\.Tech|B\.E\.|M\.B\.A\.|etc\."

# ORDER MATTERS: most specific first.
CUSTOM_PATTERNS = [
    ("MONEY",   r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?(?:/-)?(?:\s?(?:LPA|lakhs?|crores?))?"),
    ("STATUTE", r"(?:[A-Z][a-z]+\s)+Act,?\s\d{4}|(?:Section|Sec\.)\s\d+[A-Z]?(?:\(\d+\))?"),
    ("CLAUSE",  r"(?:Clause|Article)\s\d+(?:\.\d+)*(?:\([a-z]{1,3}\))?|\b\d+\.\d+(?:\.\d+)*\([a-z]{1,3}\)"),
    ("DATE",    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("TIME",    r"\b\d{1,2}(?::\d{2})?\s?(?:a\.m\.|p\.m\.)"),
    ("ABBR",    ABBREVIATIONS),
    ("HYPHEN",  r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b"),
    ("NUMBER",  r"\b\d+(?:,\d+)*(?:\.\d+)?\b"),
    ("WORD",    r"[A-Za-z]+(?:'[a-z]+)?"),
    ("PUNCT",   r"[^\w\s]"),
]
CUSTOM_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in CUSTOM_PATTERNS))
PROTECTED = {"MONEY", "STATUTE", "CLAUSE", "DATE", "TIME", "ABBR", "HYPHEN"}

def custom_tokenize(text, with_types=False):
    out = [(m.group(), m.lastgroup) for m in CUSTOM_RE.finditer(text)]
    return out if with_types else [t for t, _ in out]

def hybrid_doc(text):
    """spaCy tokenization, then merge every span the custom rules protect."""
    d = nlp.tokenizer(text)
    spans = []
    for m in CUSTOM_RE.finditer(text):
        if m.lastgroup in PROTECTED:
            sp = d.char_span(m.start(), m.end(), alignment_mode="expand")
            if sp is not None and len(sp) > 1:
                spans.append(sp)
    with d.retokenize() as retok:
        for sp in filter_spans(spans):
            retok.merge(sp)
    return d

def hybrid_tokenize(text):
    return [t.text for t in hybrid_doc(text) if not t.is_space]

TOKENIZERS = {
    "nltk":   lambda t: word_tokenize(t),
    "spacy":  lambda t: [tok.text for tok in nlp.tokenizer(t) if not tok.is_space],
    "custom": custom_tokenize,
    "hybrid": hybrid_tokenize,
}
print(custom_tokenize(probe, with_types=True))

TOKEN_GOLD = [
    ("The CTC is ₹12,00,000 per annum.",
     ["The", "CTC", "is", "₹12,00,000", "per", "annum", "."]),
    ("Payable as Rs. 8,50,000/- only.",
     ["Payable", "as", "Rs. 8,50,000/-", "only", "."]),
    ("The Employee shall not sign a non-compete.",
     ["The", "Employee", "shall", "not", "sign", "a", "non-compete", "."]),
    ("This is subject to Section 27 of the Indian Contract Act, 1872.",
     ["This", "is", "subject", "to", "Section 27", "of", "the", "Indian Contract Act, 1872", "."]),
    ("Notice is governed by Clause 4.2(a).",
     ["Notice", "is", "governed", "by", "Clause 4.2(a)", "."]),
    ("Acme Pvt. Ltd. appoints the Employee w.e.f. 15 April 2026.",
     ["Acme", "Pvt.", "Ltd.", "appoints", "the", "Employee", "w.e.f.", "15", "April", "2026", "."]),
    ("A B.Tech graduate may work-from-home.",
     ["A", "B.Tech", "graduate", "may", "work-from-home", "."]),
    ("Joining date is 01/04/2026 at 9 a.m.",
     ["Joining", "date", "is", "01/04/2026", "at", "9 a.m.", ]),
]

def token_f1(pred, gold):
    pc, gc = Counter(pred), Counter(gold)
    overlap = sum((pc & gc).values())
    p = overlap / max(len(pred), 1); r = overlap / max(len(gold), 1)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

score_rows, example_rows = [], []
for name, fn in TOKENIZERS.items():
    exact = f1s = 0
    for sent, gold in TOKEN_GOLD:
        pred = fn(sent)
        exact += int(pred == gold)
        f1s += token_f1(pred, gold)
    score_rows.append({"tokenizer": name,
                       "exact_match_%": round(100 * exact / len(TOKEN_GOLD), 1),
                       "token_F1": round(f1s / len(TOKEN_GOLD), 3)})
for sent, gold in TOKEN_GOLD:
    row = {"input": sent, "gold": " | ".join(gold)}
    for name, fn in TOKENIZERS.items():
        row[name] = " | ".join(fn(sent))
    example_rows.append(row)

tok_scores = pd.DataFrame(score_rows)
print("CAUTION: the seed gold set and the custom rules were written together, so custom/hybrid scoring 100%\n"
      "on it is circular, not evidence. The comparison becomes meaningful once you add sentences from YOUR corpus.\n")
tok_examples = pd.DataFrame(example_rows)
save(tok_examples, "tokenization_comparison.csv")
print(tok_scores)
tok_examples

# Corpus-level effect of each tokenizer (same dictionary view as Exercise 1).
rows = []
for name, fn in TOKENIZERS.items():
    toks = [tok for t in docs.values() for tok in fn(t)]
    rows.append({"tokenizer": name, "tokens": len(toks), "token_dictionary": len(set(toks)),
                 "term_dictionary": len(term_dictionary(toks))})
corpus_tok = pd.DataFrame(rows)
print(corpus_tok)

best = tok_scores.sort_values("token_F1", ascending=False).iloc[0]["tokenizer"]
print(f"Highest token F1 on the gold set: {best}")
print("Selected for the rest of the pipeline: HYBRID. Rationale: it keeps spaCy's handling of ordinary English\n"
      "(which the pure-regex custom tokenizer only approximates) while fixing every domain construct in the gold set.\n"
      "If your gold results disagree, change SELECTED_TOKENIZER and justify it in the report.")
SELECTED_TOKENIZER = "hybrid"

bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
bpe.normalizer = normalizers.NFKC()
bpe.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=CONFIG["BPE_VOCAB_SIZE"], min_frequency=2,
                              special_tokens=["[UNK]"])
bpe.train_from_iterator(list(docs.values()), trainer=trainer)
print(f"Corpus BPE trained. Vocabulary size: {bpe.get_vocab_size():,}")

try:
    gpt2 = Tokenizer.from_pretrained("gpt2")
    print(f"GPT-2 pretrained BPE loaded. Vocabulary size: {gpt2.get_vocab_size():,}")
except Exception as e:
    gpt2 = None
    print(f"GPT-2 tokenizer unavailable ({type(e).__name__}); continuing with corpus BPE only.")

def bpe_tokens(text):  return bpe.encode(text).tokens
def gpt2_tokens(text): return gpt2.encode(text).tokens if gpt2 else []

ALL_TOKENIZERS = dict(TOKENIZERS, bpe=bpe_tokens)
rows = []
for name, fn in ALL_TOKENIZERS.items():
    toks = [tok for t in docs.values() for tok in fn(t)]
    rows.append({"tokenizer": name, "token_count": len(toks), "vocabulary_size": len(set(toks))})
if gpt2:
    toks = [tok for t in docs.values() for tok in gpt2_tokens(t)]
    rows.append({"tokenizer": "gpt2_bpe", "token_count": len(toks), "vocabulary_size": len(set(toks))})
bpe_compare = pd.DataFrame(rows)
bpe_compare

# Common, rare and domain-specific words, and how each tokenizer splits them.
word_freq = Counter(t.lower() for t in custom_tokenize(corpus) if t.isalpha() and len(t) > 3)
common = [w for w, _ in word_freq.most_common(8)]
rare = sorted(w for w, c in word_freq.items() if c == 1)[:8]
DOMAIN_WORDS = ["non-compete", "non-solicitation", "confidentiality", "remuneration", "probation",
                "termination", "indemnify", "stipend", "gratuity", "work-from-home", "hereinafter",
                "jurisdiction"]

rows, n = [], 1
for category, words in [("common", common), ("rare", rare), ("domain", DOMAIN_WORDS)]:
    for w in words:
        b = bpe_tokens(w)
        row = {"S.No.": n, "Word": w, "category": category,
               "BPE Tokens": " ".join(b), "bpe_subwords": len(b),
               "nltk": " ".join(word_tokenize(w)), "spacy": " ".join(t.text for t in nlp.tokenizer(w)),
               "custom": " ".join(custom_tokenize(w))}
        if gpt2:
            g = gpt2_tokens(w); row["GPT-2 BPE"] = " ".join(g); row["gpt2_subwords"] = len(g)
        rows.append(row); n += 1
bpe_results = pd.DataFrame(rows)
save(bpe_results, "bpe_results.csv")

agg = bpe_results.groupby("category")[[c for c in ["bpe_subwords", "gpt2_subwords"] if c in bpe_results]].mean().round(2)
print("\nMean subwords per word (lower = the tokenizer 'knows' the word):")
print(agg)
bpe_results

MODALS = {"shall", "may", "must", "will", "should", "can"}

def obligations(sd):
    rows = []
    for tok in sd:
        if tok.lower_ in MODALS and tok.head.pos_ in {"VERB", "AUX"}:
            verb = tok.head
            subj = next((c for c in verb.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
            neg = any(c.dep_ == "neg" for c in verb.children)
            obj = next((c for c in verb.children if c.dep_ in {"dobj", "attr", "oprd"}), None)
            rows.append({"subject": " ".join(t.text for t in subj.subtree) if subj else "",
                         "modal": tok.text, "negated": neg, "verb": verb.lemma_,
                         "object": " ".join(t.text for t in obj.subtree) if obj else "",
                         "type": ("PROHIBITION" if neg else "OBLIGATION") if tok.lower_ in {"shall", "must"}
                                 else ("RIGHT" if not neg else "RESTRICTED RIGHT")})
    return rows

obl_rows = []
for did, sd in spacy_docs.items():
    for r in obligations(sd):
        obl_rows.append({"doc_id": did, **r})
obl_df = pd.DataFrame(obl_rows)
print(f"(i) Verb + noun phrase task: {len(obl_df)} obligations/rights extracted. Sample:")
print(obl_df.head(12))

np_counts = Counter(" ".join(t.lemma_.lower() for t in chunk if not t.is_stop and not t.is_punct)
                    for sd in spacy_docs.values() for chunk in sd.noun_chunks)
np_counts = Counter({k: v for k, v in np_counts.items() if len(k.split()) >= 2})
print("\n(ii) Noun-phrase-only task: most frequent multi-word noun phrases (clause-type candidates):")
pd.DataFrame(np_counts.most_common(15), columns=["noun_phrase", "frequency"])

DUR_PAREN = re.compile(r"\b(\d+)\s*\(\s*[a-z\- ]+\s*\)\s*(days?|weeks?|months?|years?)", re.IGNORECASE)
DURATION = re.compile(r"\b\d+\s+(?:days?|weeks?|months?|years?)\b", re.IGNORECASE)

def normalise_numbers(text):
    return DUR_PAREN.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)

ex = "The Employee shall be on probation for a period of 6 (six) months and give 30 (thirty) days' notice."
print("Before:", ex)
print("After: ", normalise_numbers(ex))

numeric_rows = []
for did, t in docs.items():
    nt = normalise_numbers(t)
    typed = custom_tokenize(nt, with_types=True)
    numeric_rows.append({"doc_id": did,
                         "money": sum(1 for _, k in typed if k == "MONEY"),
                         "dates": sum(1 for _, k in typed if k == "DATE"),
                         "durations": len(DURATION.findall(nt)),
                         "other_numbers": sum(1 for _, k in typed if k == "NUMBER")})
numeric_df = pd.DataFrame(numeric_rows)
print(f"\nDocuments containing at least one money amount: {(numeric_df.money > 0).sum()} of {len(numeric_df)}")
print(f"Documents containing at least one duration:     {(numeric_df.durations > 0).sum()} of {len(numeric_df)}")
numeric_df

from nltk.corpus import stopwords

NLTK_STOP = set(stopwords.words("english"))
SPACY_STOP = set(nlp.Defaults.stop_words)
PROTECT = {"not", "no", "nor", "never", "without", "unless", "except", "neither", "none", "cannot",
           "n't", "shall", "may", "must", "will", "should", "can", "any", "all", "only",
           "before", "after", "during", "within", "until", "against"}
ARCHAIC = {"hereby", "herein", "hereto", "hereof", "thereof", "therein", "whereas", "witnesseth",
           "hereinafter", "aforesaid"}
CUSTOM_STOP = (NLTK_STOP - PROTECT) | ARCHAIC

print(f"NLTK stop list:   {len(NLTK_STOP)} words | protected words it removes: {sorted(PROTECT & NLTK_STOP)}")
print(f"spaCy stop list:  {len(SPACY_STOP)} words | protected words it removes: {sorted(PROTECT & SPACY_STOP)}")
print(f"Custom stop list: {len(CUSTOM_STOP)} words | protected words it removes: {sorted(PROTECT & CUSTOM_STOP)}\n")

MEANING_TESTS = [
    "The Employee shall not disclose any confidential information.",
    "The Employer may terminate the Employee without notice.",
    "No bond amount is payable unless the Employee resigns before 24 months.",
    "The Employee must never solicit any client during the term.",
]
rows = []
for s in MEANING_TESTS:
    toks = [t.lower() for t in word_tokenize(s) if t.isalpha()]
    rows.append({"sentence": s,
                 "NLTK removal": " ".join(t for t in toks if t not in NLTK_STOP),
                 "spaCy removal": " ".join(t for t in toks if t not in SPACY_STOP),
                 "custom removal": " ".join(t for t in toks if t not in CUSTOM_STOP)})
pd.DataFrame(rows)

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
STEMMERS = {"porter": PorterStemmer(), "snowball": SnowballStemmer("english"), "lancaster": LancasterStemmer()}

OVER_GROUPS = [  # must stay DISTINCT
    ["employee", "employer", "employment"], ["confidential", "confidence"], ["terminal", "termination"],
    ["provision", "provide"], ["policy", "police"], ["general", "generate"], ["organ", "organization"],
]
UNDER_GROUPS = [  # should be the SAME
    ["compensate", "compensation"], ["resign", "resignation"], ["solicit", "solicitation"],
    ["indemnify", "indemnity"], ["renew", "renewal"], ["absorb", "absorption"], ["terminate", "termination"],
]

rows = []
for kind, groups in [("over-stemming test", OVER_GROUPS), ("under-stemming test", UNDER_GROUPS)]:
    for g in groups:
        row = {"test": kind, "words": ", ".join(g)}
        for name, st in STEMMERS.items():
            stems = [st.stem(w) for w in g]
            row[name] = ", ".join(stems)
            collapsed = len(set(stems)) < len(g)
            row[f"{name}_error"] = collapsed if kind.startswith("over") else not collapsed
        rows.append(row)
stem_tests = pd.DataFrame(rows)

err = {name: {"over-stemming errors": int(stem_tests.loc[stem_tests.test.str.startswith("over"), f"{name}_error"].sum()),
              "under-stemming errors": int(stem_tests.loc[stem_tests.test.str.startswith("under"), f"{name}_error"].sum())}
       for name in STEMMERS}
print(pd.DataFrame(err).T)
stem_tests[["test", "words"] + list(STEMMERS)]

err_df = pd.DataFrame(err).T
SELECTED_STEMMER = err_df.sort_values(["over-stemming errors", "under-stemming errors"]).index[0]
stemmer = STEMMERS[SELECTED_STEMMER]
print(f"Selected stemmer for Pipeline A: {SELECTED_STEMMER} "
      f"({err_df.loc[SELECTED_STEMMER, 'over-stemming errors']} over-stemming, "
      f"{err_df.loc[SELECTED_STEMMER, 'under-stemming errors']} under-stemming errors)")

porter = STEMMERS["porter"]
base = [t.lower() for t in word_tokenize(corpus) if t.isalpha()]

stop_then_stem = [porter.stem(t) for t in base if t not in NLTK_STOP]
stem_then_stop = [s for s in (porter.stem(t) for t in base) if s not in NLTK_STOP]
leaked = Counter(porter.stem(t) for t in base if t in NLTK_STOP and porter.stem(t) not in NLTK_STOP)

order_df = pd.DataFrame([
    {"order": "stop-word removal -> stemming", "tokens": len(stop_then_stem), "vocabulary": len(set(stop_then_stem)),
     "stop words surviving": 0},
    {"order": "stemming -> stop-word removal", "tokens": len(stem_then_stop), "vocabulary": len(set(stem_then_stop)),
     "stop words surviving": sum(leaked.values())},
])
print(order_df)
print("Stop words that survived because stemming ran first (stem: count):")
print(dict(leaked.most_common(12)))

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn

wnl = WordNetLemmatizer()
def wn_pos(tag):
    return {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}.get(tag[0], wn.NOUN)

TABLE_C_WORDS = ["employees", "employer", "employment", "terminated", "terminating", "agreed", "obligations",
                 "confidential", "provisions", "resigned", "compensation", "was", "binding", "better",
                 "solicited", "renewed", "indemnities", "paid"]
rows = []
for w in TABLE_C_WORDS:
    tag = pos_tag([w])[0][1]
    rows.append({"Word": w,
                 "Porter": STEMMERS["porter"].stem(w), "Snowball": STEMMERS["snowball"].stem(w),
                 "Lancaster": STEMMERS["lancaster"].stem(w),
                 "WordNet (no POS)": wnl.lemmatize(w),
                 "WordNet (POS)": wnl.lemmatize(w, wn_pos(tag)),
                 "spaCy": nlp(w)[0].lemma_})
table_c = pd.DataFrame(rows)
save(table_c, "stemming_lemmatization.csv")
table_c

def run_pipes(d):
    """Run the loaded spaCy components over an already-tokenized Doc."""
    for _, proc in nlp.pipeline:
        d = proc(d)
    return d

def pipeline_A_sents(text):
    out = []
    for sent in sent_tokenize(clean_text(text)):
        toks = [t.lower() for t in word_tokenize(sent) if re.search(r"\w", t)]
        out.append([stemmer.stem(t) for t in toks if t not in NLTK_STOP])
    return [x for x in out if x]

def pipeline_B_sents(text):
    d = nlp(clean_text(text))
    out = [[t.lemma_.lower() for t in sent if not t.is_space and not t.is_punct] for sent in d.sents]
    return [x for x in out if x]

def final_doc(text):
    return run_pipes(hybrid_doc(normalise_numbers(clean_text(text))))

def final_token(t):
    # merged domain tokens (money, statutes, hyphenated terms) keep their surface form
    return t.text.lower() if (" " in t.text or "-" in t.text or not t.is_alpha) else t.lemma_.lower()

def pipeline_final_sents(text):
    out = []
    for sent in final_doc(text).sents:
        toks = [final_token(t) for t in sent if not t.is_space and not t.is_punct]
        out.append([t for t in toks if t not in CUSTOM_STOP])
    return [x for x in out if x]

# Flat token streams (for counting and indexing) are the sentences concatenated.
def flatten(sents): return [t for s in sents for t in s]
def pipeline_A(text):     return flatten(pipeline_A_sents(text))
def pipeline_B(text):     return flatten(pipeline_B_sents(text))
def pipeline_final(text): return flatten(pipeline_final_sents(text))

PIPELINES = {"Pipeline A": pipeline_A, "Pipeline B": pipeline_B, "Final Pipeline": pipeline_final}
PIPELINE_SENTS = {"Pipeline A": pipeline_A_sents, "Pipeline B": pipeline_B_sents, "Final Pipeline": pipeline_final_sents}
processed, processed_sents, pipe_time = {}, {}, {}
for name, fn in PIPELINE_SENTS.items():
    t0 = time.perf_counter()
    processed_sents[name] = {did: fn(t) for did, t in docs.items()}
    processed[name] = {did: flatten(ss) for did, ss in processed_sents[name].items()}
    pipe_time[name] = time.perf_counter() - t0
    print(f"{name:15s} processed {len(docs)} documents in {pipe_time[name]:.2f}s")

s = MEANING_TESTS[1]
print(f"\n'{s}'")
for name, fn in PIPELINES.items():
    print(f"  {name:15s} {fn(s)}")

before_tokens = all_tokens
after_tokens = [t for toks in processed["Final Pipeline"].values() for t in toks]
table_a = pd.DataFrame([
    {"Measure": "Documents", "Before Processing": len(docs), "After Processing": len(processed["Final Pipeline"])},
    {"Measure": "Tokens", "Before Processing": len(before_tokens), "After Processing": len(after_tokens)},
    {"Measure": "Unique Tokens", "Before Processing": len(set(before_tokens)), "After Processing": len(set(after_tokens))},
    {"Measure": "Vocabulary Size", "Before Processing": len(term_dictionary(before_tokens)),
     "After Processing": len(term_dictionary(after_tokens))},
])
table_a["Change %"] = (100 * (table_a["After Processing"] - table_a["Before Processing"])
                       / table_a["Before Processing"]).round(1)
save(table_a, "preprocessing_results.csv")
table_a

POS_GOLD_TEST = [
    [("The","DT"),("Employee","NN"),("shall","MD"),("not","RB"),("sign","VB"),("a","DT"),("non-compete","NN"),(".",".")],
    [("The","DT"),("Company","NN"),("will","MD"),("pay","VB"),("a","DT"),("CTC","NN"),("of","IN"),("₹12,00,000","CD"),("per","IN"),("annum","NN"),(".",".")],
    [("Either","DT"),("party","NN"),("may","MD"),("terminate","VB"),("this","DT"),("Agreement","NN"),("by","IN"),("giving","VBG"),("notice","NN"),(".",".")],
    [("The","DT"),("Employee","NN"),("shall","MD"),("serve","VB"),("the","DT"),("bond","NN"),("for","IN"),("24","CD"),("months","NNS"),(".",".")],
    [("The","DT"),("Intern","NN"),("will","MD"),("receive","VB"),("a","DT"),("stipend","NN"),("w.e.f.","IN"),("15","CD"),("April","NNP"),("2026","CD"),(".",".")],
    [("The","DT"),("Employee","NN"),("shall","MD"),("not","RB"),("solicit","VB"),("any","DT"),("client","NN"),(".",".")],
    [("Confidential","JJ"),("information","NN"),("shall","MD"),("not","RB"),("be","VB"),("disclosed","VBN"),(".",".")],
    [("The","DT"),("Employer","NN"),("shall","MD"),("notice","VB"),("the","DT"),("breach","NN"),("promptly","RB"),(".",".")],
    [("A","DT"),("non-solicitation","NN"),("clause","NN"),("applies","VBZ"),("during","IN"),("the","DT"),("notice","NN"),("period","NN"),(".",".")],
    [("The","DT"),("Consultant","NN"),("shall","MD"),("bond","VB"),("the","DT"),("materials","NNS"),(".",".")],
]
POS_DOMAIN_TRAIN = [
    [("The","DT"),("Freelancer","NN"),("shall","MD"),("not","RB"),("disclose","VB"),("any","DT"),("data","NNS"),(".",".")],
    [("The","DT"),("Employer","NN"),("may","MD"),("notice","VB"),("a","DT"),("delay","NN"),(".",".")],
    [("The","DT"),("Worker","NN"),("must","MD"),("sign","VB"),("the","DT"),("bond","NN"),(".",".")],
    [("A","DT"),("CTC","NN"),("of","IN"),("₹8,00,000","CD"),("is","VBZ"),("payable","JJ"),(".",".")],
    [("The","DT"),("non-compete","NN"),("lasts","VBZ"),("12","CD"),("months","NNS"),("w.e.f.","IN"),("today","NN"),(".",".")],
    [("The","DT"),("Contractor","NN"),("shall","MD"),("bond","VB"),("the","DT"),("equipment","NN"),(".",".")],
]

def tag_spacy(words):
    return [t.tag_ for t in run_pipes(spacy.tokens.Doc(nlp.vocab, words=words))]

def tag_nltk(words):
    return [t for _, t in pos_tag(words)]

# ---- Custom tagger 1: rule/dictionary-based, layered on top of spaCy ----
DOMAIN_LEXICON = {w: "NN" for w in ["employee", "employer", "company", "intern", "consultant", "contractor",
                                     "freelancer", "worker", "candidate", "agreement", "ctc", "stipend",
                                     "non-compete", "non-solicitation", "gratuity", "annum"]}
DOMAIN_LEXICON.update({"w.e.f.": "IN"})
NOUN_VERB = {"notice", "bond", "sign", "release", "claim", "breach", "service"}
MONEY_TOKEN = re.compile(r"^(?:₹|Rs\.?|INR)\s?\d")

def tag_rules(words):
    tags = tag_spacy(words)
    for i, w in enumerate(words):
        lw = w.lower()
        if MONEY_TOKEN.match(w):
            tags[i] = "CD"
        elif lw in DOMAIN_LEXICON:
            tags[i] = DOMAIN_LEXICON[lw]
        elif lw in NOUN_VERB and i > 0:
            prev = tags[i - 1]
            if prev in {"MD", "TO"} or (prev == "RB" and i > 1 and tags[i - 2] == "MD"):
                tags[i] = "VB"
            elif prev in {"DT", "JJ", "PRP$", "POS"}:
                tags[i] = "NN"
    return tags

print("spaCy :", tag_spacy(["The", "Employee", "shall", "bond", "the", "CTC", "."]))
print("rules :", tag_rules(["The", "Employee", "shall", "bond", "the", "CTC", "."]))

# ---- Custom tagger 2: ML (multinomial logistic regression on hand-crafted features) ----
from nltk.corpus import treebank

def features(words, i):
    w = words[i]
    return {"w": w.lower(), "suf3": w[-3:].lower(), "suf2": w[-2:].lower(), "pre2": w[:2].lower(),
            "is_title": w.istitle(), "is_upper": w.isupper(), "has_digit": any(c.isdigit() for c in w),
            "has_hyphen": "-" in w, "has_currency": bool(MONEY_TOKEN.match(w)),
            "in_domain_lexicon": w.lower() in DOMAIN_LEXICON,
            "prev": words[i - 1].lower() if i > 0 else "<S>",
            "prev2": words[i - 2].lower() if i > 1 else "<S>",
            "next": words[i + 1].lower() if i < len(words) - 1 else "</S>"}

def to_xy(tagged_sents):
    X, y = [], []
    for sent in tagged_sents:
        sent = [(w, t) for w, t in sent if t != "-NONE-"]      # drop Treebank trace tokens
        words = [w for w, _ in sent]
        for i, (_, t) in enumerate(sent):
            X.append(features(words, i)); y.append(t)
    return X, y

tb = treebank.tagged_sents()
t0 = time.perf_counter()
X_tb, y_tb = to_xy(tb)
vec_general = DictVectorizer()
ml_general = LogisticRegression(max_iter=300).fit(vec_general.fit_transform(X_tb), y_tb)

# Domain-adapted: same features, Treebank plus domain sentences (upweighted so a handful counts)
X_dom, y_dom = to_xy(POS_DOMAIN_TRAIN)
DOMAIN_WEIGHT = 20
vec_domain = DictVectorizer()
X_all = X_tb + X_dom; y_all = y_tb + y_dom
w_all = [1.0] * len(X_tb) + [float(DOMAIN_WEIGHT)] * len(X_dom)
ml_domain = LogisticRegression(max_iter=300).fit(vec_domain.fit_transform(X_all), y_all, sample_weight=w_all)
print(f"Trained on {len(X_tb):,} Treebank tokens (+{len(X_dom)} domain tokens) in {time.perf_counter()-t0:.1f}s")

def tag_ml(model, vec):
    return lambda words: list(model.predict(vec.transform([features(words, i) for i in range(len(words))])))

TAGGERS = {"NLTK default": tag_nltk, "spaCy default": tag_spacy, "Custom rules": tag_rules,
           "Custom ML (Treebank)": tag_ml(ml_general, vec_general),
           "Custom ML (Treebank + domain)": tag_ml(ml_domain, vec_domain)}

DOMAIN_TERMS = set(DOMAIN_LEXICON) | NOUN_VERB | {"shall"}
acc_rows, detail_rows = [], []
for name, fn in TAGGERS.items():
    correct = total = dom_correct = dom_total = 0
    for sent in POS_GOLD_TEST:
        words = [w for w, _ in sent]; gold = [t for _, t in sent]
        pred = fn(words)
        for w, g, p in zip(words, gold, pred):
            total += 1; correct += (g == p)
            if w.lower() in DOMAIN_TERMS or MONEY_TOKEN.match(w):
                dom_total += 1; dom_correct += (g == p)
    acc_rows.append({"tagger": name, "accuracy_%": round(100 * correct / total, 1),
                     "domain_term_accuracy_%": round(100 * dom_correct / max(dom_total, 1), 1)})
pos_acc = pd.DataFrame(acc_rows)
print(pos_acc)

# Table D: every domain-term token in the gold set, default vs custom
for sent in POS_GOLD_TEST:
    words = [w for w, _ in sent]; gold = [t for _, t in sent]
    d, r, m = tag_spacy(words), tag_rules(words), TAGGERS["Custom ML (Treebank + domain)"](words)
    for i, w in enumerate(words):
        if w.lower() in DOMAIN_TERMS or MONEY_TOKEN.match(w):
            detail_rows.append({"Word": w, "Context": " ".join(words), "Gold": gold[i],
                                "Default POS (spaCy)": d[i], "Custom POS (rules)": r[i], "Custom POS (ML)": m[i],
                                "Default Correct": d[i] == gold[i], "Rules Correct": r[i] == gold[i],
                                "ML Correct": m[i] == gold[i]})
table_d = pd.DataFrame(detail_rows)
save(table_d, "pos_tagging_results.csv")
base = pos_acc.set_index("tagger")["accuracy_%"]
print(f"\nImprovement of rule tagger over spaCy default: {base['Custom rules'] - base['spaCy default']:+.1f} points")
table_d

nlp_ner = spacy.load("en_core_web_sm")
DOMAIN_ENT_PATTERNS = [
    ("MONEY", CUSTOM_PATTERNS[0][1]),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),       # spaCy misses slash dates common in Indian contracts
    ("LAW", r"(?:Section|Sec\.)\s\d+[A-Z]?(?:\(\d+\))?|(?:[A-Z][a-z]+\s)+Act,?\s\d{4}"),
    ("DURATION", r"\b\d+\s*(?:\([a-z\- ]+\)\s*)?(?:days?|weeks?|months?|years?)\b"),
    ("DESIGNATION", r"\b(?:(?:Senior|Junior|Lead|Principal|Associate|Chief)\s)?(?:(?:Software|Data|Research|Sales|"
                    r"Marketing|HR|Finance|Product|Technical)\s)?(?:Engineer|Developer|Analyst|Manager|Consultant|"
                    r"Executive|Scientist|Designer|Officer|Architect)\b"),
]

@Language.component("domain_entities")
def domain_entities(doc):
    new = []
    for label, pat in DOMAIN_ENT_PATTERNS:
        for m in re.finditer(pat, doc.text):
            sp = doc.char_span(m.start(), m.end(), label=label, alignment_mode="expand")
            if sp is not None:
                new.append(sp)
    kept = [e for e in doc.ents if not any(e.start < n.end and n.start < e.end for n in new)]
    doc.ents = filter_spans(new + kept)
    return doc

nlp_custom = spacy.load("en_core_web_sm")
nlp_custom.add_pipe("domain_entities", after="ner")
print("Custom NER pipeline:", nlp_custom.pipe_names)

s = "Priya Nair shall pay Rs. 2,00,000 if she leaves before 24 months, subject to Section 27."
print("\nDefault:", [(e.text, e.label_) for e in nlp_ner(s).ents])
print("Custom: ", [(e.text, e.label_) for e in nlp_custom(s).ents])

NER_GOLD = [
    ("This Agreement is made on 01/04/2026 between Acme Technologies Pvt. Ltd. and Rahul Sharma.",
     [("01/04/2026", "DATE"), ("Acme Technologies Pvt. Ltd.", "ORG"), ("Rahul Sharma", "PERSON")]),
    ("The Employee shall receive a CTC of ₹12,00,000 per annum.", [("₹12,00,000", "MONEY")]),
    ("The Employee is appointed as Senior Software Engineer in Bengaluru.",
     [("Senior Software Engineer", "DESIGNATION"), ("Bengaluru", "GPE")]),
    ("Either party may terminate this Agreement by giving 30 days notice.", [("30 days", "DURATION")]),
    ("This is subject to Section 27 of the Indian Contract Act, 1872.",
     [("Section 27", "LAW"), ("Indian Contract Act, 1872", "LAW")]),
    ("Priya Nair shall pay Rs. 2,00,000 towards training costs if she leaves before 24 months.",
     [("Priya Nair", "PERSON"), ("Rs. 2,00,000", "MONEY"), ("24 months", "DURATION")]),
    ("The courts at Mumbai shall have exclusive jurisdiction.", [("Mumbai", "GPE")]),
    ("The Intern will receive a stipend of ₹15,000 per month from Zenith Software Private Limited.",
     [("₹15,000", "MONEY"), ("Zenith Software Private Limited", "ORG")]),
]

def score_ner(model):
    tot = cor = inc = mis = 0
    for sent, gold in NER_GOLD:
        pred = {(e.text, e.label_) for e in model(sent).ents}
        g = set(gold)
        cor += len(pred & g); inc += len(pred - g); mis += len(g - pred); tot += len(g)
    p = cor / max(cor + inc, 1); r = cor / max(tot, 1)
    return {"gold entities": tot, "correct": cor, "incorrect (spurious or wrong type/span)": inc,
            "missed": mis, "precision": round(p, 3), "recall": round(r, 3),
            "F1": round(0 if p + r == 0 else 2 * p * r / (p + r), 3)}

ner_scores = pd.DataFrame({"spaCy default": score_ner(nlp_ner), "Custom": score_ner(nlp_custom)}).T
print(ner_scores)

rows = []
for sent, gold in NER_GOLD:
    d = {e.text: e.label_ for e in nlp_ner(sent).ents}
    c = {e.text: e.label_ for e in nlp_custom(sent).ents}
    for text, label in gold:
        rows.append({"Entity": text, "Gold Type": label, "Default Predicted": d.get(text, "(missed)"),
                     "Custom Predicted": c.get(text, "(missed)"),
                     "Default Correct?": d.get(text) == label, "Custom Correct?": c.get(text) == label})
table_e_gold = pd.DataFrame(rows)
table_e_gold

# Corpus-wide entity inventory. "Correct?" is left blank for you to fill in by hand (YOUR INPUT).
inv = Counter()
for did, t in docs.items():
    for e in nlp_custom(t).ents:
        inv[(e.text, e.label_)] += 1
default_types = {}
for t in docs.values():
    for e in nlp_ner(t).ents:
        default_types.setdefault(e.text, e.label_)
table_e = pd.DataFrame([{"Entity": txt, "Predicted Type (custom)": lab,
                         "Predicted Type (default)": default_types.get(txt, "(missed)"),
                         "Frequency": n, "Correct?": ""} for (txt, lab), n in inv.most_common()])
save(table_e, "ner_results.csv")
print("Entity counts by type (custom pipeline):")
print(table_e.groupby("Predicted Type (custom)")["Frequency"].sum().sort_values(ascending=False))
table_e.head(25)

from nltk.util import ngrams as nltk_ngrams

FUNCTION_WORDS = NLTK_STOP | ARCHAIC
def is_meaningful(gram):
    return gram[0] not in FUNCTION_WORDS and gram[-1] not in FUNCTION_WORDS

def ngram_counter(sent_docs, n):
    """sent_docs: {doc_id: [[tokens of sentence 1], ...]}. N-grams never cross a sentence boundary."""
    c = Counter()
    for sents in sent_docs.values():
        for sent in sents:
            c.update(nltk_ngrams(sent, n))
    return c

final_tokens = processed_sents["Final Pipeline"]
table_f, ngram_counters = [], {}
names = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "4-Gram", 5: "5-Gram"}
for n, label in names.items():
    c = ngram_counter(final_tokens, n); ngram_counters[n] = c
    meaningful = {g: f for g, f in c.items() if f >= 2 and is_meaningful(g)}
    table_f.append({"N-Gram": label, "Total": sum(c.values()), "Unique": len(c),
                    "Meaningful (freq>=2)": len(meaningful),
                    "Top N-Grams": "; ".join(f"{' '.join(g)} ({f})" for g, f in c.most_common(10))})
table_f = pd.DataFrame(table_f)
save(table_f, "ngram_results.csv")

for n, fname in [(1, "unigram_results.csv"), (2, "bigram_results.csv"), (3, "trigram_results.csv")]:
    save(pd.DataFrame([{"ngram": " ".join(g), "frequency": f, "meaningful": is_meaningful(g)}
                       for g, f in ngram_counters[n].most_common()]), fname)

term_dict = len(term_dictionary(after_tokens))
print(f"\nTerm-based dictionary (unique unigram terms): {term_dict:,}")
for n in range(2, 6):
    print(f"  {names[n]:8s} dictionary: {len(ngram_counters[n]):,}  ({len(ngram_counters[n]) / term_dict:.1f}x the term dictionary)")
table_f

print("Top 10 MEANINGFUL n-grams (the domain phrases worth indexing):\n")
for n in range(2, 6):
    top = [(" ".join(g), f) for g, f in ngram_counters[n].most_common() if f >= 2 and is_meaningful(g)][:10]
    print(f"{names[n]}:")
    for g, f in top:
        print(f"   {f:4d}  {g}")

DOMAIN_PROBES = ["employee", "employer", "employment", "confidential", "confidence", "terminal", "termination",
                 "non-compete", "non-solicitation", "shall not", "without notice", "₹12,00,000",
                 "Section 27", "notice period"]

def domain_terms_preserved(fn):
    """A probe is preserved if the pipeline keeps it (non-empty) AND it does not collide with another probe."""
    outs = {p: tuple(fn(p)) for p in DOMAIN_PROBES}
    counts = Counter(outs.values())
    return sum(1 for p, o in outs.items() if o and counts[o] == 1), outs

def negation_kept(fn):
    tests = ["The Employee shall not disclose.", "The Employer may terminate without notice.",
             "The Employee must never solicit clients."]
    return sum(1 for t in tests if any(w in fn(t) for w in ["not", "without", "never"])), len(tests)

def meaningful_ngrams(sent_docs):
    total = 0
    for n in (2, 3):
        c = ngram_counter(sent_docs, n)
        total += sum(1 for g, f in c.items() if f >= 2 and is_meaningful(g))
    return total

comparison_rows, probe_outputs = [], {}
for name, fn in PIPELINES.items():
    toks = [t for d in processed[name].values() for t in d]
    kept, outs = domain_terms_preserved(fn)
    neg, neg_total = negation_kept(fn)
    probe_outputs[name] = outs
    comparison_rows.append({"Pipeline": name, "Token Count": len(toks), "Vocabulary Size": len(set(toks)),
                            "Meaningful N-Grams": meaningful_ngrams(processed_sents[name]),
                            "Domain Terms Preserved": f"{kept}/{len(DOMAIN_PROBES)}",
                            "Negations Preserved": f"{neg}/{neg_total}",
                            "Preprocessing Time (s)": round(pipe_time[name], 2)})
pipe_structural = pd.DataFrame(comparison_rows)
print(pipe_structural)

print("How each pipeline represents the domain probes (collisions show as identical outputs):")
pd.DataFrame(probe_outputs).map(lambda t: " ".join(t) if t else "(removed)")

def build_index(token_docs):
    index = defaultdict(lambda: defaultdict(list))
    for did, toks in token_docs.items():
        for pos, tok in enumerate(toks):
            index[tok][did].append(pos)
    return {t: dict(p) for t, p in index.items()}

INDEXES = {name: build_index(processed[name]) for name in PIPELINES}
ALL_DOCS = set(docs)

# Surface form -> the lemma the corpus most often gave it, per lemmatizing pipeline.
surf_B, surf_F = defaultdict(Counter), defaultdict(Counter)
for t in docs.values():
    for tok in nlp(clean_text(t)):
        if not tok.is_space and not tok.is_punct:
            surf_B[tok.text.lower()][tok.lemma_.lower()] += 1
    for tok in final_doc(t):
        if not tok.is_space and not tok.is_punct:
            surf_F[tok.text.lower()][final_token(tok)] += 1
CORPUS_LEMMA = {"Pipeline B": {k: c.most_common(1)[0][0] for k, c in surf_B.items()},
                "Final Pipeline": {k: c.most_common(1)[0][0] for k, c in surf_F.items()}}

def query_B(text):
    return [CORPUS_LEMMA["Pipeline B"].get(t.text.lower(), t.lemma_.lower())
            for t in nlp(clean_text(text)) if not t.is_space and not t.is_punct]

def query_final(text):
    out = []
    for t in final_doc(text):
        if t.is_space or t.is_punct: continue
        tok = CORPUS_LEMMA["Final Pipeline"].get(t.text.lower(), final_token(t))
        if tok not in CUSTOM_STOP: out.append(tok)
    return out

CONFIG.setdefault("CORPUS_CONSISTENT_QUERIES", True)
QUERY_FN_CONSISTENT = {"Pipeline A": pipeline_A, "Pipeline B": query_B, "Final Pipeline": query_final}
QUERY_FN_NAIVE = dict(PIPELINES)   # plain pipeline on the isolated query: the version with the bug
QUERY_FN = QUERY_FN_CONSISTENT if CONFIG["CORPUS_CONSISTENT_QUERIES"] else QUERY_FN_NAIVE
print("Query 'training' -> naive:", QUERY_FN_NAIVE["Final Pipeline"]("training"),
      "| corpus-consistent:", QUERY_FN_CONSISTENT["Final Pipeline"]("training"))

final_index = INDEXES["Final Pipeline"]
with open(os.path.join(CONFIG["RESULTS_DIR"], "inverted_index.json"), "w", encoding="utf-8") as f:
    json.dump({t: {"df": len(p), "postings": p} for t, p in sorted(final_index.items())}, f, ensure_ascii=False, indent=1)
print(f"  saved -> {CONFIG['RESULTS_DIR']}/inverted_index.json")

for name, idx in INDEXES.items():
    print(f"{name:15s} index terms: {len(idx):,}")
print("\nSample posting lists (Final Pipeline):")
for term in ["employee", "non-compete", "confidential", "notice", "not"]:
    if term in final_index:
        print(f"  {term:14s} -> {', '.join(sorted(final_index[term]))}")

def phrase_docs(index, terms):
    """Documents where `terms` occur consecutively."""
    if not terms: return set()
    if len(terms) == 1: return set(index.get(terms[0], {}))
    candidates = set.intersection(*(set(index.get(t, {})) for t in terms))
    hits = set()
    for did in candidates:
        starts = set(index[terms[0]][did])
        for k, t in enumerate(terms[1:], start=1):
            starts &= {p - k for p in index[t][did]}
            if not starts: break
        if starts: hits.add(did)
    return hits

def tokenize_query(q):
    """Split into operands (quoted phrases or word runs) and operators AND/OR/NOT."""
    parts = re.findall(r'"[^"]+"|\bAND\b|\bOR\b|\bNOT\b|[^\s"]+', q)
    out, buf = [], []
    for p in parts:
        if p in {"AND", "OR", "NOT"}:
            if buf: out.append(("TERM", " ".join(buf))); buf = []
            out.append(("OP", p))
        elif p.startswith('"'):
            if buf: out.append(("TERM", " ".join(buf))); buf = []
            out.append(("TERM", p.strip('"')))
        else:
            buf.append(p)
    if buf: out.append(("TERM", " ".join(buf)))
    return out

def evaluate_query(q, pipeline_name, operand_fn=None, qfns=None):
    """Boolean evaluation with precedence NOT > AND > OR. operand_fn(text) -> set of doc ids."""
    idx, pfn = INDEXES[pipeline_name], (qfns or QUERY_FN)[pipeline_name]
    operand_fn = operand_fn or (lambda text: phrase_docs(idx, pfn(text)))
    toks = tokenize_query(q)
    or_groups, and_group, negate, pending_and = [], None, False, False
    def push(s):
        nonlocal and_group, negate
        s = ALL_DOCS - s if negate else s
        negate = False
        and_group = s if and_group is None else and_group & s
    for kind, val in toks:
        if kind == "TERM":
            push(operand_fn(val))
        elif val == "NOT":
            negate = not negate
        elif val == "OR":
            if and_group is not None: or_groups.append(and_group)
            and_group = None
    if and_group is not None: or_groups.append(and_group)
    return set().union(*or_groups) if or_groups else set()

def query_type(q):
    if " NOT " in f" {q} ": return "Boolean NOT"
    if " AND " in f" {q} ": return "Boolean AND"
    if " OR " in f" {q} ": return "Boolean OR"
    n = len(q.strip('"').split())
    return {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(n, f"{n}-gram phrase")

def rank(docs_found, q, pipeline_name, qfns=None):
    """Order retrieved documents by summed log-tf x idf of the positive query terms (for P@K / R@K)."""
    idx, pfn, N = INDEXES[pipeline_name], (qfns or QUERY_FN)[pipeline_name], len(ALL_DOCS)
    terms = [t for kind, v in tokenize_query(q) if kind == "TERM" for t in pfn(v)]
    def score(did):
        s = 0.0
        for t in terms:
            post = idx.get(t, {})
            if did in post:
                s += (1 + math.log(len(post[did]))) * math.log(N / len(post))
        return s
    return sorted(docs_found, key=lambda d: (-score(d), d))

def search(q, pipeline_name="Final Pipeline", qfns=None):
    t0 = time.perf_counter()
    found = evaluate_query(q, pipeline_name, qfns=qfns)
    ranked = rank(found, q, pipeline_name, qfns=qfns)
    return ranked, (time.perf_counter() - t0) * 1000

for q in ["non-compete", "notice period", "probation AND termination", "stipend OR remuneration",
          "confidential AND NOT non-compete"]:
    r, ms = search(q)
    print(f"{query_type(q):12s} {q!r:40s} -> {len(r):2d} docs {r[:8]} ({ms:.2f} ms)")

QUERIES = [  # YOUR INPUT: adapt to your corpus
    ("Q01", "stipend"), ("Q02", "non-compete"), ("Q03", "gratuity"),
    ("Q04", "notice period"), ("Q05", "confidential information"), ("Q06", "intellectual property"),
    ("Q07", "governing law"),
    ("Q08", "shall not disclose"), ("Q09", "shall not solicit"), ("Q10", "work from home"),
    ("Q11", "probation AND termination"), ("Q12", "bond AND training"),
    ("Q13", "stipend OR remuneration"), ("Q14", "confidential OR proprietary"),
    ("Q15", "confidential AND NOT non-compete"),
]

retrieval_rows = []
for qid, q in QUERIES:
    for name in PIPELINES:
        ranked, ms = search(q, name)
        retrieval_rows.append({"Query ID": qid, "Query": q, "Type": query_type(q), "Pipeline": name,
                               "Retrieved Documents": " ".join(ranked), "# Results": len(ranked),
                               "Time (ms)": round(ms, 3)})
retrieval_df = pd.DataFrame(retrieval_rows)
save(retrieval_df, "retrieval_results.csv")
retrieval_df[retrieval_df.Pipeline == "Final Pipeline"].drop(columns="Pipeline")

for q in ["shall not disclose", "shall disclose"]:
    print(f"Query: {q!r}")
    for name in PIPELINES:
        r, _ = search(q, name)
        print(f"   {name:15s} processed as {QUERY_FN[name](q)!s:36s} -> {len(r):2d} docs")
    print()

REL_PATH = os.path.join(CONFIG["ANNOTATION_DIR"], "relevance.csv")
raw_lower = {did: re.sub(r"\s+", " ", t.lower()) for did, t in docs.items()}

def provisional_relevance(q):
    """Literal substring evaluation on raw text. A rough stand-in, NOT a real relevance judgement."""
    return evaluate_query(q, "Final Pipeline",
                          operand_fn=lambda text: {d for d, t in raw_lower.items() if text.lower() in t})

if not os.path.exists(REL_PATH):
    pd.DataFrame([{"query_id": qid, "query": q, "relevant_docs": ""} for qid, q in QUERIES]).to_csv(REL_PATH, index=False)
    print(f"Created a blank relevance template at {REL_PATH}. Fill it in and re-run this module.")

rel_df = pd.read_csv(REL_PATH, dtype=str).fillna("")
filled = rel_df["relevant_docs"].str.strip().astype(bool).any()
RELEVANCE, REL_SOURCE = {}, {}
for qid, q in QUERIES:
    row = rel_df[rel_df.query_id == qid]
    if filled and len(row) and row.iloc[0]["relevant_docs"].strip():
        RELEVANCE[qid] = set(row.iloc[0]["relevant_docs"].split()); REL_SOURCE[qid] = "manual"
    else:
        RELEVANCE[qid] = provisional_relevance(q); REL_SOURCE[qid] = "PROVISIONAL"

n_prov = sum(v == "PROVISIONAL" for v in REL_SOURCE.values())
if n_prov:
    print(f"!! {n_prov} of {len(QUERIES)} queries use PROVISIONAL relevance. Do not report these numbers.")
else:
    print("All queries use your manual relevance judgements.")

K = CONFIG["TOP_K"]
def prf(ret, rel):
    ret_s = set(ret); tp = len(ret_s & rel)
    p = tp / len(ret_s) if ret_s else (1.0 if not rel else 0.0)
    r = tp / len(rel) if rel else np.nan
    f = np.nan if np.isnan(r) else (0.0 if p + r == 0 else 2 * p * r / (p + r))
    topk = ret[:K]; tpk = len(set(topk) & rel)
    pk = tpk / len(topk) if topk else (1.0 if not rel else 0.0)
    rk = tpk / len(rel) if rel else np.nan
    return p, r, f, pk, rk

eval_rows = []
for qid, q in QUERIES:
    rel = RELEVANCE[qid]
    for name in PIPELINES:
        ranked, ms = search(q, name)
        p, r, f, pk, rk = prf(ranked, rel)
        eval_rows.append({"Query ID": qid, "Query": q, "Type": query_type(q), "Pipeline": name,
                          "Relevant": len(rel), "Retrieved": len(ranked), "Precision": p, "Recall": r,
                          "F1": f, f"P@{K}": pk, f"R@{K}": rk, "Time (ms)": ms,
                          "Relevance Source": REL_SOURCE[qid]})
eval_df = pd.DataFrame(eval_rows)
save(eval_df.round(3), "evaluation_results.csv")

per_type = (eval_df[eval_df.Pipeline == "Final Pipeline"]
            .groupby("Type")[["Precision", "Recall", "F1", f"P@{K}", f"R@{K}"]].mean().round(3))
print("Final pipeline, averaged by query type (NaN = no relevant documents exist for that query):")
per_type

abl = []
for label, qfns in [("naive query lemmatization", QUERY_FN_NAIVE), ("corpus-consistent (used)", QUERY_FN_CONSISTENT)]:
    ps, rs, fs, changed = [], [], [], []
    for qid, q in QUERIES:
        ranked, _ = search(q, "Final Pipeline", qfns=qfns)
        p, r, f, _, _ = prf(ranked, RELEVANCE[qid])
        ps.append(p); rs.append(r); fs.append(f)
    abl.append({"Final pipeline, queries processed by": label, "Precision": np.nanmean(ps),
                "Recall": np.nanmean(rs), "F1": np.nanmean(fs)})
print(pd.DataFrame(abl).round(3))
diff = [(qid, q) for qid, q in QUERIES
        if search(q, "Final Pipeline", QUERY_FN_NAIVE)[0] != search(q, "Final Pipeline", QUERY_FN_CONSISTENT)[0]]
print("Queries whose results change with the fix:", diff if diff else "none")

metrics = (eval_df.groupby("Pipeline")[["Precision", "Recall", "F1", "Time (ms)"]]
           .mean().reindex(list(PIPELINES)))

# Table H: pipeline comparison
table_h = pipe_structural.set_index("Pipeline").join(metrics[["Precision", "Recall", "F1"]].round(3))
table_h = table_h.T.reset_index().rename(columns={"index": "Measure"})
save(table_h, "pipeline_comparison.csv")
print("TABLE H: Pipeline comparison")
print(table_h)

# Table I: retrieval results, one representative query per type (Final Pipeline)
fin = retrieval_df[retrieval_df.Pipeline == "Final Pipeline"]
table_i = fin.groupby("Type").first().reset_index()[["Query", "Type", "Retrieved Documents", "# Results", "Time (ms)"]]
print("\nTABLE I: Retrieval results (Final Pipeline, first query of each type)")
print(table_i)

# Table J: overall performance
table_j = metrics.copy()
table_j["Preprocessing Time (s)"] = [round(pipe_time[p], 2) for p in table_j.index]
table_j = table_j.round(3).reset_index().rename(columns={"Pipeline": "Method/Pipeline",
                                                         "Time (ms)": "Avg Query Time (ms)"})
print("\nTABLE J: Overall performance")
print(table_j)
if n_prov:
    print(f"\n!! These tables use PROVISIONAL relevance for {n_prov} queries. Fill annotations/relevance.csv first.")

sel = table_j.set_index("Method/Pipeline")[["F1"]].copy()
frac = lambda v: int(str(v).split("/")[0])
sel["domain_terms"] = pipe_structural.set_index("Pipeline")["Domain Terms Preserved"].map(frac)
sel["negations"] = pipe_structural.set_index("Pipeline")["Negations Preserved"].map(frac)
sel["meaningful_ngrams"] = pipe_structural.set_index("Pipeline")["Meaningful N-Grams"]
# Rank by F1, then domain terms preserved, then negations preserved, then n-gram quality
# (all four are comparison criteria named in the assignment).
sel = sel.sort_values(["F1", "domain_terms", "negations", "meaningful_ngrams"], ascending=False, kind="stable")
best_name = sel.index[0]
tied = sel[sel.F1 == sel.F1.iloc[0]].index.tolist()
print(f"Selected pipeline: {best_name}")
if len(tied) > 1:
    print(f"  F1 is TIED between {tied} ({sel.F1.iloc[0]:.3f}); the tie is broken by domain terms preserved,")
    print("  then negations preserved, then meaningful n-grams. A tie usually means the relevance set is too easy or provisional:")
    print("  real, manual judgements on negation-sensitive queries should separate the pipelines.\n")
else:
    print(f"  Highest mean F1 ({sel.F1.iloc[0]:.3f}).\n")
print(sel)
print("Selection reasoning to adapt for your report:")
print(" - F1 is the primary criterion because it balances missing relevant contracts (recall)")
print("   against returning wrong ones (precision); both have a real cost to someone reviewing a contract.")
print(" - Pipeline A's smaller vocabulary looks efficient, but it merges distinct legal terms and drops")
print("   negations, so it answers 'shall disclose' and 'shall not disclose' identically (Module 5).")
print(" - Pipeline B keeps meaning but indexes every function word, which inflates the index and")
print("   fills the n-grams with fragments like 'of the'.")
print(" - The Final pipeline is designed to keep what matters (negation, modals, money, statutes,")
print("   party names) and drop what does not. If your real numbers disagree, report that honestly.")


"""
Extracts pages from the papers in PDF format with additional processing of the tables
and numbered equations.
"""

import re
import unicodedata
from bisect import bisect_left, bisect_right
from pathlib import Path

import pdfplumber
from pdfplumber.utils import extract_words

SYMBOL_FIXES = {
    "¼": "=",
    "þ": "+",
    "(cid:2)": "×",
    "(cid:3)": "−",
}

SCRIPT_RATIO = 0.8 # threshold super/subscript
MIN_CHAR_SIZE = 4   
X_TOLERANCE = 1.5 # avoid gluing words together

NUMBER_RE = re.compile(r"^[-−]?\d+(\.\d+)*%?$")
CAPTION_RE = re.compile(r"^(TABLE\s+[IVXLC]+\.|Table\s+S?\d+)(:|\.|\s|$)")
EQ_NUMBER_RE = re.compile(r"\(\s*(S?\d{1,2})\s*\)\s*$") # an equation number like (3) or (S3)


### Characters ###

def fix_chars(page):
    """
    Returns the page characterss rewritten (e.g. ^3 or _E)
    """
    chars = [dict(c) for c in page.chars if c["size"] >= MIN_CHAR_SIZE and c.get("upright", True)]

    # Avoid decimal point ',' turning into ':'
    for a, c, b in zip(chars, chars[1:], chars[2:]):
        if c["text"] == ":" and a["text"].isdigit() and b["text"].isdigit() and c["fontname"] != a["fontname"]:
            c["text"] = "."

    if not chars:
        return chars
    largest = max(c["size"] for c in chars)
    visible = sorted((c for c in chars if c["text"].strip()), key=lambda c: c["x1"])
    visible_x1 = [c["x1"] for c in visible]
    small = sorted((c for c in visible if c["size"] < SCRIPT_RATIO * largest), key=lambda c: (round(c["top"]), c["x0"]))

    runs = []  # consecutive small characterss forming one script (e.g. '1.5')
    for c in small:
        last = runs[-1][-1] if runs else None
        if last and abs(last["size"] - c["size"]) < 0.2 and abs(last["top"] - c["top"]) < 1 and -1 < c["x0"] - last["x1"] < 1.5:
            runs[-1].append(c)
            continue
        center = (c["top"] + c["bottom"]) / 2
        lo, hi = bisect_left(visible_x1, c["x0"] - 1.5), bisect_right(visible_x1, c["x0"] + 2.5)
        refs = [r for r in visible[lo:hi]
                if c["size"] < SCRIPT_RATIO * r["size"]
                and r["top"] - 0.3 * r["size"] < center < r["bottom"] + 0.3 * r["size"]]
        runs.append([c])
        if refs:
            ref = min(refs, key=lambda r: abs(c["x0"] - r["x1"]))
            c["_ref"] = ref
            c["_mark"] = "^" if c["matrix"][5] - ref["matrix"][5] > 1 else "_"

    for run in runs:
        if "_ref" not in run[0]:
            continue  # a small characters with nothing next to it (footnote line)
        ref, mark = run[0]["_ref"], run[0]["_mark"]
        for c in run:
            for k in ("top", "bottom", "doctop", "y0", "y1"):
                c[k] = ref[k]
        if len(run) == 1:
            run[0]["text"] = mark + run[0]["text"]
        else:
            run[0]["text"] = mark + "{" + run[0]["text"]
            run[-1]["text"] += "}"
    return chars


def clean(text):
    for bad, good in SYMBOL_FIXES.items():
        text = text.replace(bad, good)
    return unicodedata.normalize("NFKC", text)  # to plain letters

### Lines ###

def group_lines(words):
    """
    Groups words into lines. Matches on the vertical center instead of the top.
    """
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        center = (w["top"] + w["bottom"]) / 2
        if lines and lines[-1][0]["top"] - 1 <= center <= lines[-1][0]["bottom"] + 1:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def is_numeric(text):
    # Detects numbers like 0.9723±0.0041
    return all(NUMBER_RE.match(part) for part in text.replace(",", "").split("±"))


def numeric_fraction(words):
    tokens = [w for w in words if w["text"] not in ("±", "|")]
    return sum(is_numeric(w["text"]) for w in tokens) / len(tokens) if tokens else 0.0


def is_data_row(words, min_fraction: float = 0.4):
    # 0.5 separates header lines from the data rows
    return sum(is_numeric(w["text"]) for w in words) >= 2 and numeric_fraction(words) >= min_fraction


def ordered_lines(page, chars):
    """
    Lines are in reading order. If a line has a gap in the middle of the page, 
    it is a two-column block: output the entire left column first, then the entire right column.

    If a table follows the block, its caption and header lines are kept with the table.
    """
    mid = page.width / 2
    words = extract_words(chars, x_tolerance=X_TOLERANCE)
    for w in words:
        w["text"] = clean(w["text"])  # math in bold
    out, block = [], []  # block: (line, left part, right part) 

    def flush(table_top=None):
        head = []
        if table_top is not None:
            edge = table_top
            for line, _, _ in reversed(block):
                if edge - line[0]["bottom"] > 14 or len(head) == 6:
                    break
                head.insert(0, line)
                edge = line[0]["top"]
                if CAPTION_RE.match(line_text(line)) or CAPTION_RE.match(line_text(line[:3])):
                    break
        rest = block[:len(block) - len(head)]
        out.extend(l for _, l, _ in rest if l)
        out.extend(r for _, _, r in rest if r)
        out.extend(head)
        block.clear()

    for line in group_lines(words):
        l = [w for w in line if w["x1"] <= mid + 2]
        r = [w for w in line if w["x0"] >= mid - 2]

        crosses = len(l) + len(r) < len(line) or (l and r and r[0]["x0"] - l[-1]["x1"] < 8)
        full_table_row = l and r and is_data_row(line, 0.3) and numeric_fraction(l) >= 0.3 and numeric_fraction(r) >= 0.3
        if crosses or full_table_row:
            flush(line[0]["top"] if full_table_row else None)
            out.append(line)
        else:
            block.append((line, l, r))
    flush()
    return out


def line_text(line):
    return clean(" ".join(w["text"] for w in line))


### Tables ###

def columns(rows, gap: float = 4.0):
    """
    Column x-ranges: union of the word extents of all data rows.
    """
    spans = sorted((w["x0"], w["x1"]) for row in rows for w in row)
    cols = []
    for x0, x1 in spans:
        if cols and x0 <= cols[-1][1] + gap:
            cols[-1] = (cols[-1][0], max(cols[-1][1], x1))
        else:
            cols.append((x0, x1))
    return cols


def assign(words, cols):
    cells = [[] for _ in cols]
    for w in words:
        c = (w["x0"] + w["x1"]) / 2
        i = min(range(len(cols)), key=lambda i: 0 if cols[i][0] <= c <= cols[i][1] else min(abs(c - cols[i][0]), abs(c - cols[i][1])))
        cells[i].append(w["text"])
    return [clean(" ".join(cell)) for cell in cells]


def parse_table(lines, start):
    """
    Parses the table whose caption is at lines[start]. Returns
    (caption, header, rows, end_index) or None if no data rows follow.
    """
    caption = [lines[start]]
    i = start + 1
    while i < len(lines) and len(caption) < 5 and not line_text(caption[-1]).endswith(".") and not is_data_row(lines[i]):
        caption.append(lines[i])
        i += 1
    header_lines = []
    while i < len(lines) and not is_data_row(lines[i]) and len(header_lines) < 3:
        header_lines.append(lines[i])
        i += 1
    rows = []
    while i < len(lines) and is_data_row(lines[i]):
        rows.append(lines[i])
        i += 1
    if len(rows) < 2:
        return None

    cols = columns(rows)
    header = assign([w for line in header_lines for w in line], cols) if header_lines else [""] * len(cols)
    cells = [assign(row, cols) for row in rows]
    caption_text = " ".join(line_text(line) for line in caption)
    return caption_text, header, cells, i


def unstack(header, rows):
    """
    Tables printed as stacked copies of the same columns
    Topology | Value | Topology | Value 
    become one long table
    """
    n = len(header)
    for k in range(1, n // 2 + 1):
        if n % k == 0 and n > k and all(header[i] and header[i] == header[i % k] for i in range(n)):
            long_rows = [row[j:j + k] for row in rows for j in range(0, n, k)]
            return header[:k], [r for r in long_rows if r[0] and any(r[1:])]
    return header, rows

def table_to_text(caption, header, rows):
    """
    Caption plus markdown table to keep the keywords like 'prefactors' or 'yield stress'
    """
    lines = [caption, "| " + " | ".join(h or "-" for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def table_to_sentences(caption, header, rows):
    """
    One plain text sentence per table row with the caption. 
    This allows better embedding of the table info.
    """
    label_name = header[0] or "Row"
    sentences = []
    for row in rows:
        parts = [f"{h or 'value'} = {v}" for h, v in zip(header[1:], row[1:]) if v]
        if row[0] and parts:
            row_name = f"{label_name} {row[0]}"
            sentences.append(f"{row_name} — {caption} {row_name}: {'; '.join(parts)}.")
    return sentences


### Equations ###

def extract_equations(lines):
    """
    Finds equations as a line containing '=' that ends in '(n)'.
    Keeps two lines before and one after as context.
    """
    found = []
    for i, line in enumerate(lines):
        m = EQ_NUMBER_RE.search(line)
        if not m or "=" not in line or len(line) > 200:
            continue
        found.append({
            "number": m.group(1),
            "formula": line,
            "context": " ".join(lines[max(0, i - 2):i]),
            "after": lines[i + 1] if i + 1 < len(lines) else "",
        })
    return found


### Pages ###

def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            lines = ordered_lines(page, fix_chars(page))

            tables, table_sentences = [], []
            i = 0
            while i < len(lines):
                parsed = parse_table(lines, i) if CAPTION_RE.match(line_text(lines[i])) else None
                if parsed:
                    caption, header, rows, i = parsed
                    header, rows = unstack(header, rows)
                    tables.append(table_to_text(caption, header, rows))
                    table_sentences.extend(table_to_sentences(caption, header, rows))
                else:
                    i += 1

            text_lines = [line_text(line) for line in lines]
            pages.append({
                "source": Path(pdf_path).name,
                "page": page_num,
                "text": "\n".join(text_lines),
                "tables": tables,
                "table_sentences": table_sentences,
                "equations": extract_equations(text_lines),
            })
    return pages


def extract_all(papers_dir):
    all_pages = []
    for pdf_file in sorted(Path(papers_dir).glob("*.pdf")):
        print(f"Extracting {pdf_file.name}...")
        pages = extract_pages(str(pdf_file))
        non_empty = sum(1 for p in pages if p["text"].strip())
        n_tables = sum(len(p["tables"]) for p in pages)
        n_eqs = sum(len(p["equations"]) for p in pages)
        print(f"  {len(pages)} pages, {non_empty} with extractable text, {n_tables} table(s), {n_eqs} equation(s) found")
        all_pages.extend(pages)
    return all_pages


if __name__ == "__main__":
    import sys
    papers_dir = sys.argv[1] if len(sys.argv) > 1 else "data/papers"
    pages = extract_all(papers_dir)
    print(f"\nTotal: {len(pages)} pages extracted")
import os
import re
import json
import fitz      # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress KMeans convergence warning (optional)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#-------------------------------
# Helpers
#-------------------------------
def extract_spans(doc):
    """Flatten text spans with positional and style info."""
    spans = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                y0 = round(line["bbox"][1], 1)
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text: continue
                    spans.append({
                        "pno": pno,
                        "y0": y0,
                        "text": text,
                        "size": span["size"],
                        "font": span["font"],
                        "bbox": span["bbox"]
                    })
    return spans


def cluster_font_sizes(sizes, n=4):
    """Cluster font sizes into up to n groups and return sorted centers."""
    unique_sizes = list(set(sizes))
    if len(unique_sizes) < n:
        n = len(unique_sizes)
    if n == 0:
        return []
    X = np.array(sizes).reshape(-1,1)
    km = KMeans(n_clusters=n, random_state=42).fit(X)
    return sorted(km.cluster_centers_.flatten())


def map_cluster(size, centers):
    """Map a font size to its nearest cluster index."""
    diffs = [abs(size - c) for c in centers]
    return int(np.argmin(diffs))


def is_numbered(text):
    """True if text begins with numbering (e.g. '1.', '1.1')."""
    return bool(re.match(r'^\d+\.?\d*', text))

#-------------------------------
# Core extraction
#-------------------------------
def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    spans = extract_spans(doc)
    if not spans:
        return "", []

    # 1) Cluster sizes into up to 4 groups (0=smallest ... 3=largest)
    sizes = [s['size'] for s in spans]
    centers = cluster_font_sizes(sizes, n=4)

    # 2) Group spans by (page, y0)
    lines = {}
    for s in spans:
        key = (s['pno'], s['y0'])
        lines.setdefault(key, []).append(s)

    # 3) Build mapping of text -> pages (to detect running headers/footers)
    text_pages = {}
    for s in spans:
        text_pages.setdefault(s['text'], set()).add(s['pno'])

    headings = []
    for (pno, y0), group in lines.items():
        # Skip margins (headers/footers)
        page_h = doc.load_page(pno).rect.height
        if y0 < 30 or y0 > page_h - 30:
            continue

        # Merge spans horizontally
        group.sort(key=lambda s: s['bbox'][0])
        text = ' '.join(s['text'] for s in group)

        # Filters: repeated headers, numeric-only, table rows
        if len(text_pages.get(text, [])) > 2: continue
        if re.fullmatch(r'\d+\.?', text): continue
        if len(group) > 5: continue
        if re.fullmatch(r'\d+', text): continue

        # Determine cluster indices
        cluster_idxs = [map_cluster(s['size'], centers) for s in group]
        max_c = max(cluster_idxs)

        # Skip smallest-sized spans unless all italic/bold
        if any(c == 0 for c in cluster_idxs):
            fonts = [s['font'] for s in group]
            if not all(('Italic' in f or 'Bold' in f) for f in fonts):
                continue
            max_c = 1  # demote stylized text to next cluster

        # Skip body text cluster entirely (<2)
        if max_c < 2:
            continue

        # Assign level (e.g., 3 -> H1, 2 -> H2, etc.)
        level = f"H{len(centers) - max_c}"
        if is_numbered(text) and level != 'H1':
            level = f"H{int(level[1]) - 1}"

        headings.append({'page': pno+1, 'y': y0, 'level': level, 'text': text + ' '})

    # Sort by page and position
    headings.sort(key=lambda h: (h['page'], h['y']))

    # Extract title from top two H1 on page 1
    p1_h1 = [h for h in headings if h['page']==1 and h['level']=='H1']
    p1_h1 = sorted(p1_h1, key=lambda h: h['y'])[:2]
    title = ''.join(h['text'] for h in p1_h1).strip()

    # Remove title entries from headings
    title_set = {h['text'] for h in p1_h1}
    headings = [h for h in headings if not (h['page']==1 and h['level']=='H1' and h['text'] in title_set)]

    outline = [{'level':h['level'], 'text':h['text'], 'page':h['page']} for h in headings]
    return title, outline


def process_pdfs():
    os.makedirs('output', exist_ok=True)
    for fn in sorted(os.listdir('input')):
        if not fn.lower().endswith('.pdf'): continue
        title, outline = extract_outline(os.path.join('input', fn))
        with open(os.path.join('output', fn.replace('.pdf','.json')), 'w', encoding='utf-8') as f:
            json.dump({'title': title, 'outline': outline}, f, ensure_ascii=False, indent=4)
        print('Processed', fn)


if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("completed processing pdfs")

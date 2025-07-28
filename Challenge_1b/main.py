#!/usr/bin/env python3
"""
Challenge 1B: Persona-Driven Document Intelligence
Complete solution in a single file that can be run directly.

Usage: python main.py

Author: Challenge 1B Solution
Date: 2025
"""

import os
import json
import re
import string
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.error("PyMuPDF not found. Please install: pip install PyMuPDF")
    exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
except ImportError:
    logger.error("scikit-learn not found. Please install: pip install scikit-learn")
    exit(1)

# ---------- Utility Functions ----------

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!\?;:\-()]+', ' ', text)
    return text.strip()

stop_words = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'is','are','was','were','be','been','have','has','had','do','does','did',
    'will','would','could','should','this','that','these','those','i','you','he',
    'she','it','we','they','me','him','her','us','them','my','your','his','its',
    'our','their','from','as','so','all','any','can','may','not'
])

def extract_keywords(text: str, min_length:int=3) -> List[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w)>=min_length and w not in stop_words and not w.isdigit()]

# ---------- Document Analyzer ----------

class DocumentAnalyzer:
    def __init__(self, max_sections:int=10, max_subs:int=5):
        self.max_sections = max_sections
        self.max_subs = max_subs
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))

    def extract_sections(self, pdf_path:str) -> List[Dict[str,Any]]:
        doc = fitz.open(pdf_path)
        spans=[]
        for p in range(doc.page_count):
            page=doc.load_page(p)
            for b in page.get_text('dict')['blocks']:
                if b['type']!=0: continue
                for line in b['lines']:
                    y0=round(line['bbox'][1],1)
                    txt=' '.join(span['text'].strip() for span in line['spans'] if span['text'].strip())
                    if txt:
                        spans.append({'p':p,'y':y0,'text':txt,'spans':line['spans']})
        sections=[]
        current=None
        for item in sorted(spans, key=lambda x:(x['p'],x['y'])):
            txt=item['text']
            # heading heuristic: all caps or ends with ':'
            if txt.isupper() or txt.endswith(':'):
                if current:
                    sections.append(current)
                current={'title':clean_text(txt),'page':item['p']+1,'content':''}
            else:
                if current:
                    current['content']+= ' ' + clean_text(txt)
                else:
                    current={'title':'Introduction','page':item['p']+1,'content':clean_text(txt)}
        if current:
            sections.append(current)
        doc.close()
        return [s for s in sections if len(s['content'])>20]

    def rank_sections(self, sections:List[Dict], persona:str, job:str)->List[Dict]:
        docs=[s['title']+' '+s['content'] for s in sections]
        query=persona+' '+job
        tfidf=self.vectorizer.fit_transform(docs+[query])
        sims=cosine_similarity(tfidf[-1],tfidf[:-1]).flatten()
        for i,s in enumerate(sections): s['score']=sims[i]
        ranked=sorted(sections, key=lambda x:-x['score'])[:self.max_sections]
        for idx,s in enumerate(ranked,1): s['importance_rank']=idx
        return ranked

    def extract_subsections(self, ranked:List[Dict], persona:str, job:str)->List[Dict]:
        result=[]
        for sec in ranked[:self.max_subs]:
            sents=re.split(r'(?<=[.!?]) +', sec['content'])
            sel=sents[:3]
            text=' '.join(sel)
            result.append({'document':sec['document'],'refined_text':text,'page_number':sec['page']})
        return result

# ---------- Collection Processor ----------

def process_collection(path:str)->None:
    cfg=json.load(open(os.path.join(path,'challenge1b_input.json')))
    persona=cfg['persona']['role']; job=cfg['job_to_be_done']['task']
    docs=cfg['documents']
    analyzer=DocumentAnalyzer()
    all_secs=[]
    for d in docs:
        pdf=os.path.join(path,'PDFs',d['filename'])
        secs=analyzer.extract_sections(pdf)
        for s in secs: s['document']=d['filename']
        all_secs+=secs
    ranked=analyzer.rank_sections(all_secs,persona,job)
    subs=analyzer.extract_subsections(ranked,persona,job)
    out={'metadata':{
            'input_documents':[d['filename'] for d in docs],
            'persona':persona,'job_to_be_done':job,
            'processing_timestamp':datetime.now().isoformat()
        },
        'extracted_sections':[{'document':s['document'],'section_title':s['title'],
                               'importance_rank':s['importance_rank'],'page_number':s['page']}
                              for s in ranked],
        'subsection_analysis':subs
    }
    with open(os.path.join(path,'challenge1b_output.json'),'w',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    logger.info(f"Wrote output for {os.path.basename(path)}")

# ---------- Main ----------

def main():
    # Get the current directory or specify the path where Collection folders are located
    ROOT = os.getcwd()  # Current directory
    # Alternative: ROOT = "."  # Current directory
    # Alternative: ROOT = "path/to/collections"  # Specific path if collections are elsewhere
    
    try:
        directories = os.listdir(ROOT)
        logger.info(f"Found directories in {ROOT}: {directories}")
    except FileNotFoundError:
        logger.error(f"Directory not found: {ROOT}")
        logger.info("Please ensure you're running the script from the correct directory or update the ROOT variable")
        return
    
    collection_found = False
    for col in sorted(directories):
        cp = os.path.join(ROOT, col)
        if os.path.isdir(cp) and col.startswith('Collection'):
            collection_found = True
            logger.info(f"Processing {col}")
            try:
                process_collection(cp)
            except Exception as e:
                logger.error(f"Error processing {col}: {e}")
    
    if not collection_found:
        logger.warning("No Collection directories found. Please check if you're in the correct directory.")
        logger.info(f"Current directory: {ROOT}")
        logger.info("Expected to find directories starting with 'Collection'")

if __name__=='__main__':
    main()
from pyserini.search.lucene import LuceneSearcher
import json
from tqdm import tqdm
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sentence_transformers.util import cos_sim
import os
import numpy as np
from typing import List

INDEX_DIR = '../indexes'
CHECKPOINT_DIR = '../Checkpoint'

# nlp = spacy.load("en_core_web_md")
ssearcher = LuceneSearcher(os.path.join(INDEX_DIR, 'index-wikipedia-kilt-doc-20210421-f29307/'))
sentence_embedding_model = SentenceTransformer(os.path.join(CHECKPOINT_DIR, 'sentence-transformers/all-mpnet-base-v2'))


def information_retrieval(query, topk):
    hits = ssearcher.search(query, topk)
    paragraphs = []
    titles = []
    for i in range(len(hits)):
        doc = ssearcher.doc(hits[i].docid)
        json_doc = json.loads(doc.raw())
        doc_text = json_doc['contents']
        title = doc_text.split('\n')[0]
        # print(title)
        paragraphs.append(json_doc['contents'])
        titles.append(title)
    return paragraphs, titles


def get_fact_candidate_similarity(query, candidates, sentence_embedding_model): #calculate similarity
    question_embeddings = sentence_embedding_model.encode([query])[0]
    candidates_embedding = sentence_embedding_model.encode(candidates)
    sim_list = np.zeros(len(candidates))
    for index in range(len(candidates)):
        sim_list[index] = cos_sim(question_embeddings, candidates_embedding[index])
    return sim_list


def select_candidates(query, candidates, titles, sentence_embedding_model, topk=1):
    sim_list = get_fact_candidate_similarity(query, titles, sentence_embedding_model)

    sorted_index = np.argsort(sim_list)
    
    sorted_index = sorted_index[::-1][:topk]
    # return [(candidates[idx], sim_list[idx]) for idx in sorted_index]
    return [candidates[idx] for idx in sorted_index]


def search(question, topk, sentence_embedding_model) -> List[str]:
    if sentence_embedding_model:
        paragraphs, titles = information_retrieval(question, 10)
    
        paragraphs = select_candidates(question, paragraphs, titles, sentence_embedding_model, topk)
    else:
        paragraphs, titles = information_retrieval(question, topk)
    processed_paragraphs = []
    for p in paragraphs:
        lines = p.split('\n')
        title = lines[0]
        content = ' '.join(lines[1:])
        
        truncated_p = ' '.join(content.split(' ')[:100])
        processed_paragraphs.append(title + '\n' + truncated_p)
        
    return '\n'.join(processed_paragraphs)



        


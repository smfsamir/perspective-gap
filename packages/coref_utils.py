from typing import List

def compute_paragraph_boundaries(paragraphs):
    paragraph_boundaries =[]
    start = 0
    end = 0
    for i, paragraph in enumerate(paragraphs):
        if i == len(paragraphs) - 1:
            end = start + len(paragraph)
        else: 
            end = start + len(paragraph) + 1
        paragraph_boundaries.append((start, end))
        start = end
    return paragraph_boundaries

def compute_fastcoref_annotation(fcoref_instance, 
                                 article_paragraphs: List[str], 
                                 article_name):
    full_text = " ".join(article_paragraphs)
    paragraph_boundaries = compute_paragraph_boundaries(article_paragraphs)
    preds = fcoref_instance.predict(texts=[full_text])[0]
    clusters = preds.get_clusters()
    cluster_indices = preds.get_clusters(as_strings=False)
    # TODO: have to put the target paragraphs into the clusters.
    return {
        'article': article_name,
        'cluster_indices': cluster_indices,
        'clusters': clusters,
        'full_text': full_text,
        'paragraph_boundaries': paragraph_boundaries
    }


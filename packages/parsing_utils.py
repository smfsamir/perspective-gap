from typing import Dict
import ipdb
import re
import unicodedata
import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Iterable

@dataclass
class CorefEntityMetadata:
    cluster_strings: List[str] # this will be provided as input
    cluster_indices: List[Tuple[int, int]] # this will be provided as input
    auto_paragraph_indices: List[int] # this will be provided as input
    valid_entity: bool # this needs to be predicted at inference time
    police_aligned: str # yes/no, or NA if not valid_entity. This needs to be predicted at inference time
    entity_name: str # the name of the entity, NA if not valid_entity. Otherwise, the name should be present in paragraph_indices. This needs to be predicted at inference time

@dataclass
class CorefEntityInferenceMetadata:
    cluster_strings: List[str] # this will be provided as input
    cluster_indices: List[Tuple[int, int]] # this will be provided as input
    auto_paragraph_indices: List[int] # this will be provided as input

def get_manual_annotation_occurrences(manual_annotation_obj, entity) -> List[int]:
    occurrences = []
    for paragraph_key, entity_list in manual_annotation_obj['task2'].items():
        if entity in entity_list:
            occurrences.append(int(paragraph_key.split(' ')[1]))
    return occurrences

def get_paragraph_occurrences(paragraph_boundaries: List[Tuple],
                             entity_occurrences: List[Tuple]):
    occurrences = []
    for i, occurrence in enumerate(entity_occurrences):
        start, end = occurrence
        for j, (para_start, para_end) in enumerate(paragraph_boundaries):
            if start >= para_start and end <= para_end:
                occurrences.append(j + 1)  # +1 to make it 1-indexed
                break
    return occurrences

def load_training_data_annotations_for_person(person_name: str, outlet: str,
                                              identifier: Optional[int] = -1, 
                                              fname="data/training_data.json") -> Dict:
    with open(fname, 'r') as f:
        training_data_annotations = json.load(f)
    training_data_annotation = None
    matched_annotations = []
    for k, v in training_data_annotations.items():
        if person_name in k and outlet in k:
            if identifier != -1 and (not f"{identifier}" == k.split('_')[0]):
                continue
            training_data_annotation = v
            matched_annotations.append(training_data_annotation)
    assert training_data_annotation is not None, f"Could not find training data annotation for {person_name} and {outlet} and {identifier}."
    assert len(matched_annotations) == 1, f"Found multiple training data annotations for {person_name} and {outlet} and {identifier}: {matched_annotations}."
    return training_data_annotation

def load_article_paragraphs(article: str, path="data/articles") -> List[str]:
    with open(f'{path}/{article}', 'r') as f:
        paragraphs = json.load(f)
    return paragraphs

def load_unsupervised_article_paragraphs(article: str, path="unsupervised_articles") -> List[str]:
    with open(f'{path}/{article}', 'r') as f:
        paragraphs = json.load(f)['article']
    return paragraphs

def retrieve_entity_from_coref_objs(coref_objs: List[CorefEntityMetadata], 
                                    entity_name) -> CorefEntityMetadata:
    for coref_obj in coref_objs:
        if coref_obj.entity_name.lower() == entity_name.lower():
            return coref_obj
    raise ValueError(f"Could not find coref object for entity {entity_name}.")

def load_repaired_articles(path: str="data/repaired_coref_annotations") -> Iterable[str]:
    articles = os.listdir(path)
    # remove _repaired suffix, just before the .json extension. Keep the .json extension
    # articles = set([article.replace('_repaired', '') for article in articles])
    return articles

def load_all_articles(path: str="data/articles") -> Iterable[str]:
    articles = os.listdir(path)
    # return only the filename (+ extension), not the full path
    return articles

def extract_enumerated_paragraphs(text: str):
    """
    Extracts enumerated paragraphs (e.g., 1. ..., 2. ..., etc.)
    from a block of text and returns them as a list of strings.
    """
    start = " Here are the paragraphs that mention them:\n"
    end = f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 
    relevant_lines = text[text.index(start) + len(start):text.index(end)].strip().split('\n')
    # remove the enumeration (e.g., "1. ", "2. ", etc.) from each line
    return [line[3:] for line in relevant_lines]

def get_pb_entities_training(annotation_object, include_perspectives_only: bool):
    if include_perspectives_only:
        police_aligned_entities = annotation_object['task1']['Police-aligned']
        police_aligned_entities = [(entity[:entity.rindex(" (")] if " (" in entity else entity)  for entity in police_aligned_entities] # remove the (id) part
        perspective_entities = set([])
        # iterate through task 2 paragraph entities
        for _, entities in annotation_object['task2'].items():
            for entity in entities:
                if entity in police_aligned_entities:
                    perspective_entities.update(set([entity]))
        return list(perspective_entities)
    else:
        entities = annotation_object['task1']['Police-aligned'] 
        entities = [entity[:entity.rindex(" (")] for entity in entities] # remove the (id) part
        return entities

def get_civ_entities_training(annotation_obj, include_perspectives_only, count_victim=False):
    if include_perspectives_only:
        victim_aligned_entities = annotation_obj['task1']['Victim-aligned']
        victim_aligned_entities = [(entity[:entity.rindex(" (")] if " (" in entity else entity) for entity in victim_aligned_entities]
        perspective_entities = set([])
        # iterate through task 2 paragraph entities
        for _, entities in annotation_obj['task2'].items():
            for entity in entities:
                if entity in victim_aligned_entities:
                    perspective_entities.update(set([entity]))
        return list(perspective_entities)
    else:
        entities = annotation_obj['task1']['Victim-aligned']
        assert '(victim)' in entities[0] 
        entities = [entity[:entity.rindex(" (")] for entity in entities] # remove the (id) part

    return entities if count_victim else entities[1:]

def compute_paragraph_to_affinities(gt_annotations) -> Dict:
    paragraph_to_affinity = {}
    bureaucrats = get_pb_entities_training(gt_annotations, 
                                           include_perspectives_only=True)
    civ_entities = get_civ_entities_training(gt_annotations,
                                             include_perspectives_only=True)
    for paragraph_index, entities in gt_annotations['task2'].items(): 
        entity = entities[0] # entities is a list but usually only has one item.
        if entity in bureaucrats:
            paragraph_to_affinity[int(paragraph_index.split(' ')[1])] = 'police-aligned'
        elif entity in civ_entities:
            paragraph_to_affinity[int(paragraph_index.split(' ')[1])] = 'victim-aligned'
        else:
            raise ValueError(f"Entity {entity} not found in either police-aligned or victim-aligned entities.")
    return paragraph_to_affinity


def normalize_whitespace(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text).split()
    )
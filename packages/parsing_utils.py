import loguru
import pandas as pd
import polars as pl
from typing import Dict
import ipdb
import re
import unicodedata
import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Iterable, Callable

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
        article_obj = json.load(f)
        if type(article_obj) == list:
            paragraphs = article_obj
        elif type(article_obj) == dict:
            paragraphs = article_obj['article']
        else:
            raise ValueError(f"Unexpected type for article object: {type(article_obj)}")
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
    # sort the articles by date published. The date published is contained in the .json file, as 'date_published'
    articles = sorted(articles, key=lambda x: json.load(open(f'{path}/{x}', 'r'))['date_published'])
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

def load_ti_frame(timc_v2_path) -> pl.DataFrame:
    ti_frame = pd.read_csv(f'{timc_v2_path}/tracking_injustice.csv', keep_default_na=False, na_values=[''])[['CASE', 'DATE', 'NAME_FIRST', 'NAME_LAST', 'NAME_MIDDLE', 'NICKNAME',
       'AGE', 'GENDER', 'RACE', 'PROVINCE', 'POLICE_SERVICE',
       'HIGHEST_LEVEL_FORCE', 'ARMED_TYPE', 'PERSON_IN_CRISIS',
       'CITY_TOWN_REGION', 'NEIGHBOURHOOD', 'INTERSECTION', 'ADDRESS',
       'HOSPITAL/BUILDING LOCATION (IF APPLICABLE)', 'SUMMARY', 'Source',
       'Postal Code', 'Address Type', 'Notes']]
    ti_frame = pl.from_pandas(ti_frame)
    return ti_frame

def load_cd_frame(timc_v2_path) -> pl.DataFrame:
    cd_frame = pd.read_csv(f'{timc_v2_path}/CustodyDeaths.csv')
    cd_frame = pl.from_pandas(cd_frame)
    return cd_frame

def get_all_incidents_and_articles(timc_v2_path) -> List[Tuple[int, int]]:
    lookup_frame = pl.read_csv(f'{timc_v2_path}/lookup.csv').filter(
        (pl.col('TRACKING_INJUSTICE_CASE_ID').is_not_null()) |
        (pl.col('CUSTODY_DEATHS_ID').is_not_null())
    )

    # TODO: have to make sure that even when TI and CD case IDs are not null, that they are actually in the respective CSVs
    # because some of them were deleted.
    # the paths are "tracking_injustice.csv" and "CustodyDeaths.csv"
    # implement this logic here:
    valid_ti_cases = set((load_ti_frame(timc_v2_path))['CASE'].to_list())
    valid_cd_cases = set((load_cd_frame(timc_v2_path))['Case'].to_list())

    logger = loguru.logger
    no_main_text_files = []
    incident_articles_lst = []
    stale_case_id_incidents = []
    global_news_tag_articles = []
    for row in lookup_frame.iter_rows(named=True):
        extended_set_index = row['INCIDENT_ID']
        tracking_case_id = row['TRACKING_INJUSTICE_CASE_ID']
        custody_case_id = row['CUSTODY_DEATHS_ID']

        if tracking_case_id is not None:
            resolvable = tracking_case_id in valid_ti_cases
        else:
            resolvable = custody_case_id in valid_cd_cases

        if not resolvable:
            stale_case_id_incidents.append(extended_set_index)
            continue
        try:
            with open(f"{timc_v2_path}/{extended_set_index}.json") as f:
                try:
                    json_obj = json.load(f)
                except json.decoder.JSONDecodeError:
                    logger.warning(f"Failed to load {extended_set_index}.json")
                    continue
                all_articles = json_obj['articles']
                for i in range(len(all_articles)):
                    article_obj = all_articles[i]
                    body_lines = article_obj['maintext']          # <- adjust: see note below
                    url = article_obj['url']
                    if body_lines is None or len(body_lines) == 0:
                        no_main_text_files.append(
                            (f"{extended_set_index}.json", 
                             article_obj['publication'])
                        )
                        continue
                    elif url and "globalnews.ca/tag" in url:
                        global_news_tag_articles.append((extended_set_index, i))
                        continue
                    incident_articles_lst.append((extended_set_index, i))
        except FileNotFoundError:
            continue
    
    if stale_case_id_incidents:
        logger.warning(
            f"{len(stale_case_id_incidents)} incident(s) skipped -- their lookup.csv case ID "
            f"isn't found in its source CSV (likely deleted since lookup.csv was built): "
            f"{stale_case_id_incidents}"
        )
    if global_news_tag_articles:
        logger.warning(f"{len(global_news_tag_articles)} Global News tag pages were filtered out.")
    logger.warning(f"The following files have no main text: {no_main_text_files}")
    return incident_articles_lst

def filter_text(incident_id: int, article_index_id: int, 
                        publisher: str, text_lines: List[str], filter_json_obj: Dict):
    filter_json_index = f"{incident_id}_{article_index_id}_{publisher}_{len(text_lines)}"
    try:
        filter_lst = [ind - 1 for ind in filter_json_obj[filter_json_index]['dropped_indices']] # switch to 0-based indexing
    except KeyError:
        loguru.logger.warning(f"{filter_json_index} is not in the keymap, which contains {len(filter_json_obj)} entries.")
        return text_lines
    valid_text = []
    for i in range(0, len(text_lines)):
        if i in filter_lst:
            continue
        else:
            valid_text.append(text_lines[i])
    return valid_text

def obtain_article_from_timc(timc_v2_path, incident_id: int, article_index: int,
                             filter_json: Optional[Dict[str, Dict]] = None):
    with open(f"{timc_v2_path}/{incident_id}.json") as f:
        articles = json.load(f)['articles']
    article = articles[article_index]
    if filter_json is not None:
        article['maintext'] = filter_text(
            incident_id, article_index, article['publication'], article['maintext'], filter_json
        )
    return article
def convert_lookup_indices_to_ti_cd_indices(timc_v2_path):
    lookup_frame = pl.read_csv(f'{timc_v2_path}/lookup.csv').filter(
        (pl.col('TRACKING_INJUSTICE_CASE_ID').is_not_null()) |
        (pl.col('CUSTODY_DEATHS_ID').is_not_null())
    )
    lookup_to_database_indices = {}
    for row in lookup_frame.rows(named=True):
        lookup_to_database_indices[row['INCIDENT_ID']] =\
            {
                'ti_index': row['TRACKING_INJUSTICE_CASE_ID'] if\
                      row['TRACKING_INJUSTICE_CASE_ID'] else 'unavailable',
                'cd_index': row['CUSTODY_DEATHS_ID'] if\
                    row['CUSTODY_DEATHS_ID'] else 'unavailable'
             }
    return lookup_to_database_indices

def incident_loader(timc_v2_path: str):
    lookup_to_database_indices = convert_lookup_indices_to_ti_cd_indices(timc_v2_path)
    ti_frame = load_ti_frame(timc_v2_path)
    cd_frame = load_cd_frame(timc_v2_path)
    def load_by_incident_index(incident_id: int):
        row = lookup_to_database_indices[incident_id]
        if row['ti_index'] != 'unavailable':
            index = row['ti_index']
            ti_row = ti_frame.filter(pl.col('CASE') == index).row(0, named=True)
            return {
                'PROVINCE': ti_row['PROVINCE'],
                'NAME': (ti_row['NAME_FIRST']) +\
                     (ti_row['NAME_MIDDLE'] if ti_row['NAME_MIDDLE'] else "")+\
                     (ti_row['NAME_LAST'])
            }

        if row['cd_index'] != 'unavailable':
            index = row['cd_index']
            cd_row = cd_frame.filter(pl.col('Case') == index).row(0, named=True)
            return {
                'PROVINCE': cd_row['Province'],
                'NAME': cd_row['Full Name']
            }
        raise Exception("Shoudln't be here.")
    return load_by_incident_index
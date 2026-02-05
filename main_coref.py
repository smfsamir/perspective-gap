from tqdm import tqdm
from termcolor import colored
import pandas as pd
import polars as pl
import numpy as np
import os
import json
import click
from typing import List, Tuple, Dict, Iterable
import ipdb
import loguru
from dataclasses import dataclass

from packages.parsing_utils import CorefEntityMetadata
from packages.coref_utils import compute_paragraph_boundaries

logger = loguru.logger

def compute_weighted_median(coverage_fractions: List[str]):
    assert len(coverage_fractions) > 0, "No coverage fractions provided."
    vals = []
    for frac in coverage_fractions:
        num, den = map(int, frac.split('/'))
        vals.append((num/den if den != 0 else 0, den))

    # Sort by value
    vals.sort(key=lambda x: x[0])

    # Compute weighted median
    total_weight = sum(w for _, w in vals)
    half_weight = total_weight / 2

    cumulative = 0
    weighted_median = -1
    for value, weight in vals:
        cumulative += weight
        if cumulative >= half_weight:
            weighted_median = value
            break
    return weighted_median

def retrieve_all_coref_mentions_of_entity(coref_annotation_obj: Dict, 
                                          entity_name: str) -> List[CorefEntityMetadata]:
    coref_metadata_objs = [CorefEntityMetadata(**obj) for obj in coref_annotation_obj]
    return [obj for obj in coref_metadata_objs if obj.entity_name == entity_name]



@click.command()
def compute_fastcoref_annotations():
    from fastcoref import FCoref
    model = FCoref(device='cuda:0')
    all_articles = os.listdir('data/articles')
    # get a 50-25-25 random split of the articles, for a train/valid/test set
    train_indices = np.random.choice(len(all_articles), size=int(0.5*len(all_articles)), replace=False)
    remaining_indices = list(set(range(len(all_articles))) - set(train_indices))
    valid_indices = np.random.choice(remaining_indices, size=int(0.25*len(all_articles)), replace=False)
    test_indices = list(set(remaining_indices) - set(valid_indices))

    train_articles = [all_articles[i] for i in train_indices]
    valid_articles = [all_articles[i] for i in valid_indices]
    test_articles = [all_articles[i] for i in test_indices]
    logger.info(f"Train articles: {train_articles}")
    logger.info(f"Valid articles: {valid_articles}")
    logger.info(f"Test articles: {test_articles}")

    # articles = ["78_Radford James Good Dagger_Global News.json", 
    #             "99_Hudson Daryl Willis_Surrey Now-Leader.json",
    #             "28_Alex Wettlaufer_Global News.json",
    #             "44_Gerald Rattu_Durham Region.json",
    #             "80_Joey Knapaysweet_TimminsToday.com.json",
    #             "73_Sterling Ross Cardinal_CBC.json",
    #             "15_Bill Saunders_CTV.json",
    #             "97_Maurizio Angelo Facchin_Burnaby Now.json",
    #             "89_Jason Gary Roy_Calgary Herald.json"] # TODO: fill in the 10 articles.

    fcoref_annotations = []
    for article in all_articles:
        with open(f'data/articles/{article}', 'r') as f:
            paragraphs = json.load(f)
        full_text = " ".join(paragraphs)
        paragraph_boundaries = compute_paragraph_boundaries(paragraphs)
        preds = model.predict(texts=[full_text])[0]
        clusters = preds.get_clusters()
        cluster_indices = preds.get_clusters(as_strings=False)
        # TODO: have to put the target paragraphs into the clusters.
        fcoref_annotations.append({
            'article': article,
            'cluster_indices': cluster_indices,
            'clusters': clusters,
            'full_text': full_text,
            'paragraph_boundaries': paragraph_boundaries
        })
    with open('data/all_fcoref_annotations.json', 'w') as f:
        json.dump(fcoref_annotations, f, indent=4)

def check_if_entity_name_in_paragraphs(paragraphs: List[str], 
                                       paragraph_indices: List[int],
                                       entity_name) -> bool:
    """
    Check if the complete entity name is present in any of the paragraphs.
    """
    for index in paragraph_indices:
        paragraph = paragraphs[index - 1]
        if entity_name.lower() in paragraph.lower():
            return True
    return False


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

def get_manual_annotation_occurrences(manual_annotation_obj, entity):
    occurrences = []
    for paragraph_key, entity_list in manual_annotation_obj['task2'].items():
        if entity in entity_list:
            occurrences.append(paragraph_key)
    return occurrences

def write_annotation_report(article: str,
                            matched_entities: List[CorefEntityMetadata], 
                            unmatched_entities: List[CorefEntityMetadata],
                            dirname: str = "data/linked_coref_annotations"):
    article = article.split('.')[0]  # remove the .json extension
    article_fname =f"{dirname}/{article}.json"
    with open(f"{article_fname}", 'w') as f:
        # write all the metadata as a json list.
        # do the matched entities first, and then the unmatched entities.
        all_metadata = [entity.__dict__ for entity in matched_entities] + \
            [entity.__dict__ for entity in unmatched_entities]
        json.dump(all_metadata, f, indent=4)

def load_article_paragraphs(article: str) -> List[str]:
    with open(f'data/articles/{article}', 'r') as f:
        paragraphs = json.load(f)
    return paragraphs

def get_civ_entities_training(annotation_obj, 
                              include_perspectives_only, 
                              count_victim=False):
    if include_perspectives_only:
        victim_aligned_entities = annotation_obj['task1']['Victim-aligned']
        victim_aligned_entities = [entity[:entity.index(" (")] for entity in victim_aligned_entities]
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
        entities = [entity[:entity.index(" (")] for entity in entities] # remove the (id) part
    return entities if count_victim else entities[1:]

def get_pb_entities_training(annotation_object, include_perspectives_only: bool):
    if include_perspectives_only:
        police_aligned_entities = annotation_object['task1']['Police-aligned']
        police_aligned_entities = [entity[:entity.index(" (")] for entity in police_aligned_entities] # remove the (id) part
        perspective_entities = set([])
        # iterate through task 2 paragraph entities
        for _, entities in annotation_object['task2'].items():
            for entity in entities:
                if entity in police_aligned_entities:
                    perspective_entities.update(set([entity]))
        return list(perspective_entities)
    else:
        entities = annotation_object['task1']['Police-aligned'] 
        entities = [entity[:entity.index(" (")] for entity in entities] # remove the (id) part
        return entities

def get_civ_entities_training(annotation_obj, include_perspectives_only, count_victim=False):
    if include_perspectives_only:
        victim_aligned_entities = annotation_obj['task1']['Victim-aligned']
        victim_aligned_entities = [entity[:entity.index(" (")] for entity in victim_aligned_entities]
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
        entities = [entity[:entity.index(" (")] for entity in entities] # remove the (id) part
    return entities if count_victim else entities[1:]
        
# TODO: modify the parse so it automatically picks up any adjacent quotes.
def produce_annotation_report(article, 
                              coref_annotation_obj,
                              manual_annotation_obj):
    print(f"===={article}=====")
    civ_entities = get_civ_entities_training(
        manual_annotation_obj, 
        include_perspectives_only=True, 
        count_victim=False
    )
    pb_entities = get_pb_entities_training(
        manual_annotation_obj, 
        include_perspectives_only=True
    )
    annotated_entities = set(civ_entities + pb_entities)

    # remove any entities who don't have 


    # remove the parenthetical element from the entity
    annotated_entities = [entity.split(' (')[0] for entity in annotated_entities]
    clusters = coref_annotation_obj['clusters']
    total_matches = 0
    found_entities = []

    # load the paragraphs from data/articles
    with open(f'data/articles/{article}', 'r') as f:
        paragraphs = json.load(f)

    valid_cluster_indices = []
    metadata_valid_entities = []
    metadata_invalid_entities = []

    valid_entity_paragraph_match_ratios = []

    for entity in annotated_entities:
        found_cluster_index = -1
        print(f"----Searching for entity: {entity}-----")
        found = False
        for i, cluster in enumerate(clusters):
            for automatic_entity in cluster:
                automatic_entity = automatic_entity.lower()
                if entity in automatic_entity:
                    total_matches += 1
                    found = True
                    found_cluster_index = i
                    break
            if found:
                break

        if not found:
            logger.warning(f"Entity {entity} not found in article {article}.")
        else:
            assert found_cluster_index != -1
            coref_paragraph_occurrences = get_paragraph_occurrences(
                coref_annotation_obj['paragraph_boundaries'],
                coref_annotation_obj['cluster_indices'][found_cluster_index]
            ) 
            annotation_indices = get_manual_annotation_occurrences(
                manual_annotation_obj, entity)
            annotation_indices = [int(index.split(' ')[1]) for index in annotation_indices]
            found_entities.append(entity)
            print("Coref indices:", coref_paragraph_occurrences)
            print("Manual annotation indices:", annotation_indices)

            overlap_count = len(set(coref_paragraph_occurrences) & set(annotation_indices))
            
            valid_entity_paragraph_match_ratios.append(
                f"{overlap_count}/{len(annotation_indices)}"
            )
            # # if len(annotation_indices) - overlap_count  > 5:
            # #     logger.warning(f"entity {entity} has a large number of unmatched paragraphs: {len(annotation_indices) - overlap_count} unmatched paragraphs.")
            #     ipdb.set_trace()

        
            # construct the metadata object
            is_police_aligned = any([iter_entity.startswith(entity) for iter_entity in manual_annotation_obj['task1']['Police-aligned']])
            metadata_valid_entity = CorefEntityMetadata(
                cluster_strings=clusters[found_cluster_index],
                cluster_indices=coref_annotation_obj['cluster_indices'][found_cluster_index],
                auto_paragraph_indices=coref_paragraph_occurrences,
                valid_entity=True,
                police_aligned='yes' if is_police_aligned else 'no',
                entity_name=entity
            )
            metadata_valid_entities.append(metadata_valid_entity)
            valid_cluster_indices.append(found_cluster_index)
            # assert that the entity is present in the text in the paragraph indices
            assert check_if_entity_name_in_paragraphs(
                paragraphs, 
                coref_paragraph_occurrences, 
                entity
            ), f"Entity {entity} not found in paragraphs for article {article}."
    logger.info(f"{total_matches}/{len(annotated_entities)} entities matched for article {article}.")

    # iterate over the cluster indices that were not validated, and create metadata entries for them
    unfound_entities = set(annotated_entities) - set(found_entities)
    unfound_entity_paragraph_match_ratios = []
    

    for i, cluster in enumerate(coref_annotation_obj['clusters']):
        if i not in valid_cluster_indices:
            # create a metadata object for this cluster
            coref_paragraph_occurrences = get_paragraph_occurrences(
                coref_annotation_obj['paragraph_boundaries'],
                coref_annotation_obj['cluster_indices'][i]
            )
            if len(unfound_entities) == 0:
                overlap_f1s = [0.0] * len(coref_paragraph_occurrences)
            else:
                overlap_f1s = []
            for entity in unfound_entities:
                manual_paragraph_occurrences = get_manual_annotation_occurrences(
                    manual_annotation_obj,
                    entity)  
                # compute the f1 overlap with the manual annotation occurrences
                overlap = set(coref_paragraph_occurrences) & set(manual_paragraph_occurrences)
                if len(overlap) > 0:
                    overlap_f1 = len(overlap) / (len(coref_paragraph_occurrences) + len(manual_paragraph_occurrences) - len(overlap))
                    overlap_f1s.append(overlap_f1)
                else:
                    overlap_f1s.append(0.0)
            metadata_invalid_entity = CorefEntityMetadata(
                cluster_strings=cluster,
                cluster_indices=coref_annotation_obj['cluster_indices'][i],
                auto_paragraph_indices=coref_paragraph_occurrences,
                valid_entity=False,
                police_aligned='NA',
                entity_name='NA'
            )
            # TODO: sort by max f1-overlap with the missing entities.
            metadata_invalid_entities.append((metadata_invalid_entity, max(overlap_f1s)))
    # sort the invalid entities by the max overlap f1 score
    metadata_invalid_entities.sort(key=lambda x: x[1], reverse=True)
    metadata_invalid_entities = [entity[0] for entity in metadata_invalid_entities]
    write_annotation_report(article, 
                            metadata_valid_entities, 
                            metadata_invalid_entities 
                            )
    return {
        'matched_entities': metadata_valid_entities,
        'unfound_entities': list(unfound_entities),
        'total_matches': total_matches,
        'matched_entities_match_ratios': valid_entity_paragraph_match_ratios
    }
    
def load_training_data_annotations_for_person(person_name: str, outlet: str):
    with open("data/training_data.json", 'r') as f:
        training_data_annotations = json.load(f)
    training_data_annotation = None
    for k, v in training_data_annotations.items():
        if person_name in k and outlet in k:
            training_data_annotation = v
            break
    assert training_data_annotation is not None, f"Could not find training data annotation for {person_name} and {outlet}."
    return training_data_annotation

def get_all_entities_names_only(training_data_obj: Dict, include_victim=False) -> Iterable[str]:
    civ_entities = get_civ_entities_training(
        training_data_obj, 
        include_perspectives_only=True, 
        count_victim=False
    )
    pb_entities = get_pb_entities_training(
        training_data_obj, 
        include_perspectives_only=True
    )
    all_entities = set(civ_entities + pb_entities)
    all_entities = [entity.split(' (')[0] for entity in all_entities]
    return all_entities

def get_num_paragraphs_for_entity(entity, annotation_object) -> int:
    assert '(' not in entity # entity role should be stripped off
    count = 0
    for _, entities in annotation_object['task2'].items():
        if entity in entities:
            count += 1
    return count

def get_paragraphs_for_entity(entity, annotation_object) -> List[int]:
    assert '(' not in entity # entity role should be stripped off
    paragraphs = []
    for paragraph_key, entities in annotation_object['task2'].items():
        if entity in entities:
            paragraphs.append(int(paragraph_key.split(' ')[1]))
    return paragraphs

@click.command()
def inspect_annotations():
    include_unfound_entity_ratios = True

    with open('data/all_fcoref_annotations.json', 'r') as f:
        coref_annotations = json.load(f)
    with open("data/training_data.json", 'r') as f:
        training_data_annotations = json.load(f)
    
    discovered_counts = []
    total_counts = []
    match_ratios_found = []
    perfect_articles = []
    imperfect_articles = []
    article_to_match_ratios = {}
    for article_annotations in coref_annotations:
        article = article_annotations['article']
        manual_annotations = training_data_annotations[article]
        result_dict = produce_annotation_report(article, 
                                  article_annotations, 
                                  manual_annotations)
        discovered_counts.append(len(result_dict['matched_entities']))
        match_ratios_found.extend(result_dict['matched_entities_match_ratios'])
        num_paragraphs_for_unfound_entities = [get_num_paragraphs_for_entity(entity, manual_annotations) for entity in result_dict['unfound_entities']]
        match_ratios_for_unfound_entities = [f"0/{num_paragraphs}" for num_paragraphs in num_paragraphs_for_unfound_entities]
        if include_unfound_entity_ratios:
            match_ratios_found.extend(match_ratios_for_unfound_entities)
        total_counts.append(len(result_dict['unfound_entities']) + len(result_dict['matched_entities']))
        # article_to_recall_ratios[article] = f"{len(result_dict['matched_entities'])}/{len(result_dict['unfound_entities']) + len(result_dict['matched_entities'])}"

        coref_perfect = all([(int(ratio.split('/')[0]) / int(ratio.split('/')[1])) == 1 for ratio in result_dict['matched_entities_match_ratios']])
        if len(result_dict['unfound_entities']) == 0 and coref_perfect:
            print(f"All entities found {len(result_dict['matched_entities'])} and perfectly matched ({result_dict['matched_entities_match_ratios']}) for article {article}.")
            # perfect_articles += 1
            perfect_articles.append(article)
        else:
            imperfect_articles.append(article)
        
        total_num_paragraphs_for_all_entities = 0
        for entity in get_all_entities_names_only(manual_annotations):
            total_num_paragraphs_for_all_entities += get_num_paragraphs_for_entity(entity, manual_annotations)
        article_to_match_ratios[article] = f"{sum([int(ratio.split('/')[0]) for ratio in result_dict['matched_entities_match_ratios']])}/{total_num_paragraphs_for_all_entities}"
    entity_match_ratios = [f"{discovered}/{total}" for discovered, total in zip(discovered_counts, total_counts)]
    print(f"Entity Match Ratios: {entity_match_ratios}")
    print(f"Overall: {sum(discovered_counts)}/{sum(total_counts)} entities matched.")
    # get the median entity match ratio
    median_entity_match_ratio = np.median([int(ratio.split('/')[0]) / int(ratio.split('/')[1]) for ratio in entity_match_ratios])
    print(f"Median Entity Match Ratio: {median_entity_match_ratio:.2f}")

    print(f"Match Ratios: {sorted(match_ratios_found, key=lambda x: float(x.split('/')[0]) / float(x.split('/')[1]), reverse=True)}")
    match_total_numerator = sum([int(ratio.split('/')[0]) for ratio in match_ratios_found])
    match_total_denominator = sum([int(ratio.split('/')[1]) for ratio in match_ratios_found])
    print(f"Overall Match Ratio: {match_total_numerator}/{match_total_denominator} = {match_total_numerator / match_total_denominator:.2f}")
    # compute the median match, weighted by the denominator
    match_ratios_found = [int(ratio.split('/')[0]) / int(ratio.split('/')[1]) for ratio in match_ratios_found]
    median_match_ratio = np.median(match_ratios_found)
    print(f"Median Match Ratio: {median_match_ratio:.2f}")

    logger.info(f"Perfect Articles: {len(perfect_articles)}/{len(coref_annotations)}. They are {perfect_articles}")
    # write imperfect articles to a CSV, with one column
    with open('data/imperfect_articles.csv', 'w') as f:
        f.write("article,\n")
        for article in imperfect_articles:
            f.write(f"{article},\n")
    # print the article to match ratio, sorted
    sorted_article_to_match_ratios = dict(sorted(article_to_match_ratios.items(), key=lambda item: float(item[1].split('/')[0]) / float(item[1].split('/')[1]), reverse=True))
    print("Article to Match Ratios:")
    for article, ratio in sorted_article_to_match_ratios.items():
        print(f"{article}: {ratio} = {int(ratio.split('/')[0]) / int(ratio.split('/')[1]):.2f}")
    
    # compute the overall ratios for each article



        
# model = FCoref(device='cuda:0')
@click.group()
def main():
    pass

def identify_undiscovered_entities():
    pass

@click.command()
def repair_article_annotations():
    person_outlet = input("Enter the name of the coref annotation you want to repair (e.g., Erixon_Kabera_CBC): ")
    all_imperfect_articles = pd.read_csv('data/imperfect_articles.csv')['article'].to_list()
    def retrieve_matching_article(articles):
        matched_articles = [article for article in articles if person_outlet in article]
        assert len(matched_articles) == 1, f"Expected exactly one match for {person_outlet}, found {len(matched_articles)} matches."
        return matched_articles[0]
    matched_person_article = retrieve_matching_article(all_imperfect_articles)
    with open(f'data/linked_coref_annotations/{matched_person_article}', 'r') as f:
        coref_obj = json.load(f)
        coref_objs = [CorefEntityMetadata(**obj) for obj in coref_obj]
        unmatched_entities = [obj for obj in coref_objs if not obj.valid_entity]
        matched_entities = [obj for obj in coref_objs if obj.valid_entity]
    training_data_annotation = load_training_data_annotations_for_person(
        person_name = person_outlet.split('_')[0],
        outlet = person_outlet.split('_')[1]
    )
    article_paragraphs = load_article_paragraphs(matched_person_article)
    all_entities = get_all_entities_names_only(training_data_annotation)
    matched_entity_names = [entity.entity_name for entity in matched_entities]
    undiscovered_entities = set(all_entities) - set(matched_entity_names)
    still_unmatched = []
    for entity in tqdm(unmatched_entities):
        # TODO: should any entities be merged?
        print(f"The currently undiscovered entities are: {undiscovered_entities}")
        auto_paragraph_indices = list(sorted(list(set(entity.auto_paragraph_indices))))
        # TODO: display these paragraphs from these articles.
        paragraphs = "\n".join([f"- {article_paragraphs[index - 1]}" for index in auto_paragraph_indices])
        print(f"Entity cluster strings: {entity.cluster_strings}\n")
        print(f"Consider these paragraphs for the entity:\n{paragraphs}\n")
        print(f"Do these describe the perspectives of either:\n{colored(matched_entity_names, 'green')}\nOR\n{colored(undiscovered_entities, 'red')}?")
        user_input = input("Enter the entity name if it matches one of the above, or 'no' if it does not match: ")
        while user_input != 'no' and (user_input not in matched_entity_names) and (user_input not in undiscovered_entities):
            user_input = input(f"Invalid input. Please enter one of the following entity names: {colored(matched_entity_names, 'red')} or {colored(undiscovered_entities, 'green')}, or 'no' if it does not match: ")
        if user_input.lower() == 'no':
            still_unmatched.append(entity)
            continue
        else:
            matched_entities.append(CorefEntityMetadata(
                cluster_strings=entity.cluster_strings,
                cluster_indices=entity.cluster_indices,
                auto_paragraph_indices=entity.auto_paragraph_indices,
                valid_entity=True,
                # TODO: might have to update this a bit
                police_aligned='yes' if user_input in get_pb_entities_training(training_data_annotation, include_perspectives_only=True) else 'no',
                entity_name=user_input
            ))
            if user_input in matched_entity_names: # just a merge
                pass
            else: # newly discovered
                undiscovered_entities.remove(user_input)
                matched_entity_names.append(user_input)
            # TODO: do some updating of the entity arrays
            ## TODO: remove from undiscovered entities (names)
    # write back the updated annotations
    new_article_name = matched_person_article.split('.')[0]  # remove the .json extension + 
    new_article_name = new_article_name + '_repaired.json'
    write_annotation_report(new_article_name, 
                            matched_entities, 
                            still_unmatched, 
                            dirname="data/repaired_coref_annotations")
# @click.command()
# def annotate_missing_entities():

# TODO: fill in.
def compute_paragraph_recall(repaired_annotation, 
                             original_annotation, 
                             training_annotation):
    pass

@click.command()
def compare_recall_auto_vs_repaired():
    # TODO: implement this.
    total_original_recall_proportions = []
    total_repaired_recall_proportions = []
    for article_fname in os.listdir('data/repaired_coref_annotations'):
        person_name = article_fname.split('_')[1]
        outlet = article_fname.split('_')[2]
        training_data_annotation = load_training_data_annotations_for_person(
            person_name = person_name,
            outlet = outlet
        )

        with open(f'data/repaired_coref_annotations/{article_fname}', 'r') as repaired_f:
            repaired_annotation = json.load(repaired_f)
        with open(f'data/linked_coref_annotations/{article_fname.replace("_repaired", "")}', 'r') as original_f:
            original_annotation = json.load(original_f)
        annotated_entities = get_all_entities_names_only(training_data_annotation)
        original_recall_proportions = []
        repaired_recall_proportions = []
        for entity in annotated_entities:
            annotated_paragraphs = get_paragraphs_for_entity(entity, training_data_annotation)
            repaired_coref_mentions = retrieve_all_coref_mentions_of_entity(repaired_annotation, entity)
            repaired_coref_paragraphs = []
            for mention in repaired_coref_mentions:
                repaired_coref_paragraphs.extend(mention.auto_paragraph_indices)
            repaired_coref_paragraphs = list(set(repaired_coref_paragraphs))
            original_coref_mentions = retrieve_all_coref_mentions_of_entity(original_annotation, entity)
            original_coref_paragraphs = []
            for mention in original_coref_mentions:
                original_coref_paragraphs.extend(mention.auto_paragraph_indices)
            original_coref_paragraphs = list(set(original_coref_paragraphs))
            assert len(annotated_paragraphs) > 0, f"Entity {entity} has no annotated paragraphs."
            original_recall_numerator = len(set(original_coref_paragraphs) & set(annotated_paragraphs))
            repaired_recall_numerator = len(set(repaired_coref_paragraphs) & set(annotated_paragraphs))
            recall_denom = len(annotated_paragraphs)
            original_recall_proportions.append(f"{original_recall_numerator}/{recall_denom}")
            repaired_recall_proportions.append(f"{repaired_recall_numerator}/{recall_denom}")
            total_original_recall_proportions.append(f"{original_recall_numerator}/{recall_denom}")
            total_repaired_recall_proportions.append(f"{repaired_recall_numerator}/{recall_denom}")

        print(f"====={article_fname}=====")
        print(f"Original weighted median: {compute_weighted_median(original_recall_proportions)}")
        print(f"Repaired weighted median: {compute_weighted_median(repaired_recall_proportions)}")
        print("=====")
    print("=====Overall=====")
    print(f"Original weighted median: {compute_weighted_median(total_original_recall_proportions)}")
    print(f"Repaired weighted median: {compute_weighted_median(total_repaired_recall_proportions)}")

main.add_command(compute_fastcoref_annotations)
main.add_command(inspect_annotations)   
main.add_command(repair_article_annotations)
main.add_command(compare_recall_auto_vs_repaired)

if __name__ == "__main__":
    main()
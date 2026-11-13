import numpy as np
import pandas as pd
from dotenv import dotenv_values
from collections import defaultdict, Counter
import ipdb
from aquarel import load_theme
import polars as pl
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import click 
import loguru

logger = loguru.logger
from packages.coref_utils import compute_fastcoref_annotation
from packages.parsing_utils import CorefEntityInferenceMetadata, get_paragraph_occurrences, load_unsupervised_article_paragraphs,\
    obtain_article_from_timc, get_all_incidents_and_articles, incident_loader
from packages.constants import MEDIA_STORM_PROVINCE_MAP
from packages.qwen import load_model_generator, train_peft_on_dataset, eval_model, obtain_perspective_labels_qwen

CONFIG = dotenv_values(".env")

@click.command()
def inspect_distributions_unsupervised():
    outlets = []
    year_to_v_ids = defaultdict(list)
    for article in os.listdir('unsupervised_articles'):
        with open(f'unsupervised_articles/{article}', 'r') as f:
            obj = json.load(f)
            date = obj['incident_date']
            year = date.split('-')[0]
            year_to_v_ids[year].append(obj['victim_id'])
            outlet = obj['publisher']
            outlets.append(outlet)
    theme = load_theme("solarized_light")
    theme.apply()
    # create a dataframe with the columns total number of articles, number of unique victim IDs, year
    years = []
    num_articles = []
    is_unique = []
    for year in range(2000, 2026):
        year = str(year)
        v_ids = year_to_v_ids[year]
        if not year.isdigit():
            continue
        years.append(year)
        years.append(year)
        num_articles.append(len(v_ids))
        is_unique.append('Total articles')
        num_articles.append(len(set(v_ids)))
        is_unique.append('Num. deaths')
        # set grid 
    # sort by year
    # rotate xtick labels
    sns.barplot(x=years, y=num_articles, hue=is_unique)
    plt.title(f'Distribution of articles by year (n = {len(os.listdir("unsupervised_articles"))})')
    plt.ylabel('Number of articles')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.savefig('unsupervised_article_years_distribution.png')
    # save high quality
    plt.savefig('unsupervised_article_years_distribution.png', dpi=300)
    # set x-axis label to Year, y-axis label to Number of articles

    # create another figure where we show the distribution of the outlets in decreasing order.
    # Use the top 20 outlets, and put everything else in other
    plt.figure()
    series = pd.Series(outlets)
    outlet_counts = series.value_counts(ascending=False)
    top_outlets = outlet_counts[:30]
    other_count = outlet_counts[30:].sum()
    top_outlets['Other'] = other_count
    sns.barplot(x=top_outlets.index, y=top_outlets.values)
    plt.xticks(rotation=45) 
    plt.xlabel('News Outlet (Top 30)')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig('unsupervised_article_outlet_distribution.png', dpi=300)
    ipdb.set_trace()

    # count the number of articles per victim_id
    c = Counter()
    for article in os.listdir('unsupervised_articles'):
        with open(f'unsupervised_articles/{article}', 'r') as f:
            obj = json.load(f)
            v_id = obj['victim_id']
            c[v_id] += 1
    # what is the median number of articles per victim_id?
    counts = list(c.values())
    logger.info(f"Median number of articles per victim ID: {pd.Series(counts).median()}")




    # only include the top 30 outlets, and put the rest in other
    # outlet_counts = {}
    # for outlet in outlets:
    #     if outlet not in outlet_counts:
    #         outlet_counts[outlet] = 0
    #     outlet_counts[outlet] += 1

    # # create a new figure
    # plt.figure()
    # sorted_outlet_counts = sorted(outlet_counts.items(), key=lambda x: x[1], reverse=True)
    # top_outlets = sorted_outlet_counts[:30]
    # top_outlet_names = [x[0] for x in top_outlets]
    # top_outlet_values = [x[1] for x in top_outlets]
    # other_count = sum([x[1] for x in sorted_outlet_counts[30:]])
    # top_outlet_names.append('Other')
    # top_outlet_values.append(other_count)
    # sns.barplot(x=top_outlet_names, y=top_outlet_values)
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig('unsupervised_article_outlet_distribution.png')

    # print the counts of IDs sorted downwards by frequency
    # id_counts = {}
    # for id in ids:
    #     if id not in id_counts:
    #         id_counts[id] = 0
    #     id_counts[id] += 1
    # sorted_id_counts = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)
    # with open('unsupervised_article_id_counts.txt', 'w') as f:
    #     for id, count in sorted_id_counts:
    #         f.write(f"{id}: {count}\n")

@click.command()
@click.argument('folder', type=str)
@click.argument('output_filename', type=str)
def compute_fastcoref_annotations(folder: str, output_filename: str):
    from fastcoref import FCoref
    model = FCoref(device='cuda:0')

    all_articles = os.listdir(folder)
    fcoref_annotations = []
    for article in all_articles:
        with open(f'{folder}/{article}', 'r') as f:
            article_obj = json.load(f)
            if type(article_obj) == list:
                paragraphs = article_obj
            elif type(article_obj) == dict:
                paragraphs = article_obj['article']
            else:
                raise ValueError(f"Unexpected type for the articles: {type(article_obj)}")
        fcoref_annotations.append(
            compute_fastcoref_annotation(model, paragraphs, article) 
        )
    with open(f'data/{output_filename}_fcoref_annotations.json', 'w') as f:
        json.dump(fcoref_annotations, f, indent=4)

@click.command()
@click.argument('input_file_prefix', type=str)
def compute_article_basis_coref_objects(input_file_prefix):
    with open(f'data/{input_file_prefix}_fcoref_annotations.json', 'r') as f:
        fcoref_annotations = json.load(f)
    for article_annotations in tqdm(fcoref_annotations):
        coref_inference_objects = []
        article = article_annotations['article']
        cluster_strings = article_annotations['clusters']
        cluster_occurrence_indices = article_annotations['cluster_indices']
        for i in range(len(cluster_strings)):
            coref_paragraph_occurrences = get_paragraph_occurrences(
                    article_annotations['paragraph_boundaries'],
                    cluster_occurrence_indices[i]
                ) 
            coref_inference_objects.append(
                CorefEntityInferenceMetadata(
                    cluster_strings=cluster_strings[i],
                    cluster_indices=cluster_occurrence_indices[i],
                    auto_paragraph_indices=coref_paragraph_occurrences
                )
            )
            # save to file
        # create directory if it doesn't exist
        os.makedirs(f'data/{input_file_prefix}_coref_annotations', exist_ok=True)
        with open(f'data/{input_file_prefix}_coref_annotations/{article}', 'w') as f:
            json.dump([obj.__dict__ for obj in coref_inference_objects], f, indent=4)

def construct_length_limited_inference_prompt(victim_name: str, 
                                    coref_entity_obj: CorefEntityInferenceMetadata,
                                    all_paragraphs: List[str], 
                                    coref_auto_indices: List[int], 
                                    ) -> Dict:
    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    #### Constructing the input
    coref_auto_paragraphs = "\n".join([f"{i+1}. {all_paragraphs[index - 1]}" for i, index in enumerate(coref_auto_indices)])
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 
    # NOTE: did we need the object below for something...?
    # index_to_auto_paragraph_index = {index: i+1 for i, index in enumerate(coref_auto_indices)}


    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    #### Constructing the input
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 
    return {'prompt': task_instruction_str} 

def _create_inference_instance(victim_name: str, 
                              all_paragraphs: List[str], 
                              coref_metadata_obj: CorefEntityInferenceMetadata
                              ) -> List[Dict]:
    # TODO: need to return multiple instances, containing at most 5 paragraphs.

    MAX_PARAGRAPHS = 5
    # TODO: get the intersection of the paragraphs
    ## TODO: watch out for multiple paragraphs
    unique_auto_indices = list(sorted(list(set(coref_metadata_obj.auto_paragraph_indices))))
    num_prompts_required = len(unique_auto_indices) // MAX_PARAGRAPHS + (1 if len(unique_auto_indices) % MAX_PARAGRAPHS > 0 else 0)
    training_instances = []
    for i in range(num_prompts_required):
        coref_indices = unique_auto_indices[i*MAX_PARAGRAPHS:(i+1)*MAX_PARAGRAPHS]
        training_instances.append(construct_length_limited_inference_prompt(
            victim_name=victim_name,
            coref_entity_obj=coref_metadata_obj,
            all_paragraphs=all_paragraphs,
            coref_auto_indices=coref_indices
        ))
    # NOTE: uncomment this after figuring out how to load the set of perspective paragraphs in the natural language format
    #     if coref_metadata_obj.valid_entity:
    #         output_dict = json.loads(training_instances[-1]['completion'])
    #         if len(output_dict['perspective_paragraphs']) > 0:
    #             all_perspective_paragraphs_empty = False

    # if coref_metadata_obj.valid_entity and all_perspective_paragraphs_empty:
    #     logger.warning("Valid entity but no perspective paragraphs found")
    #     ipdb.set_trace()
    return training_instances

@click.command()
@click.argument('input_file_prefix', type=str)
def create_coref_inference_dataset(input_file_prefix: str):
    from datasets import Dataset
    inference_articles = os.listdir(f"data/{input_file_prefix}_coref_annotations")
    outlets = []
    victim_names = []
    inference_instances = []
    article_indices = []
    for article in tqdm(inference_articles):
        article_index = article.split('_')[0]
        person_name = article.split('_')[1]
        outlet = article.split('_')[2]

        paragraphs = load_unsupervised_article_paragraphs(article, path=f"{input_file_prefix}_articles")
        coref_metadata_objects = [CorefEntityInferenceMetadata(**obj) for obj in json.load(open(os.path.join(f"data/{input_file_prefix}_coref_annotations", article)))]
        for coref_metadata_obj in coref_metadata_objects:
            current_inference_instances = _create_inference_instance(
                    victim_name=person_name,
                    all_paragraphs=paragraphs,
                    coref_metadata_obj=coref_metadata_obj
                )
            inference_instances.extend(
               current_inference_instances
            )
            outlets.extend([outlet] * len(current_inference_instances))
            victim_names.extend([person_name] * len(current_inference_instances))
            article_indices.extend([article_index] * len(current_inference_instances))
    dataset = Dataset.from_dict({
        'victim_name': victim_names,
        'outlet': outlets,
        'prompt': [instance['prompt'] for instance in inference_instances],
        'article_index': article_indices
    })
    # dataset.to_json("data/distillation_data/coref_inference_dataset.json")
    dataset.to_json(f"data/distillation_data/{input_file_prefix}_coref_inference_dataset.json")
    return dataset

@click.command()
def compute_perspective_distributions_timc_v1():
    CIV_LABEL = 'State'
    STATE_LABEL = 'Civilian'
    SHORT_NAMES = {CIV_LABEL: 'Civilian', STATE_LABEL: 'State official'}
    TOP_OUTLETS = ['Toronto Star', 'The Globe and Mail', 'CBC', 'Global News', 'CTV']


    PREDICTIONS_PATH = "timc_v1_qwen_predictions"
    TIMC_V1_PATH = "/Users/smfsamir/sympathy-in-media/unsupervised_articles"

    rows = []
    for article in sorted(os.listdir(PREDICTIONS_PATH)):
        with open(os.path.join(TIMC_V1_PATH, article)) as f:
            obj = json.load(f)
        year = obj['incident_date'].split('-')[0]
        if not year.isdigit():
            logger.warning(f"{article}: unparseable incident year {year!r}; skipped")
            continue
        with open(os.path.join(PREDICTIONS_PATH, article)) as f:
            labels = json.load(f)
        n = len(labels)
        rows.append({
            'year': int(year),
            'publication_date': obj['date_published'],
            'outlet': obj['publisher'],
            'v_id': obj['victim_id'],
            'article_id': article.split('_')[0],
            'num_paragraphs': len(obj['article']),
            CIV_LABEL: sum(x == 'civilian-aligned' for x in labels) / n,
            'no_entity_proportion': sum(x == 'no entity' for x in labels) / n,
            STATE_LABEL: sum(x == 'state-aligned' for x in labels) / n,
        })
    df = pl.DataFrame(rows)

    # --- example articles ---
    long_enough = pl.col('num_paragraphs') >= 5
    print("Articles with high punishment bureaucrat representation:")
    print(df.with_columns((pl.col(STATE_LABEL) - pl.col(CIV_LABEL)).alias('proportion_diff'))
            .filter(long_enough & (pl.col(CIV_LABEL) > 0))
            .sort('proportion_diff', descending=True).head(10))
    print("Articles with high civilian-aligned representation:")
    print(df.with_columns((pl.col(CIV_LABEL) - pl.col(STATE_LABEL)).alias('proportion_diff'))
            .filter(long_enough & (pl.col(STATE_LABEL) > 0))
            .sort('proportion_diff', descending=True).head(10))
    print("Articles with pluralistic representation:")
    print(df.filter(long_enough & (pl.col(CIV_LABEL) > 0.1) & (pl.col(STATE_LABEL) > 0.1)))

    # One row per (article, perspective type), civilian and state only
    df_long = df.unpivot(
        index=['year', 'publication_date', 'outlet', 'v_id', 'article_id', 'num_paragraphs'],
        on=[CIV_LABEL, STATE_LABEL],
        variable_name='proportion_type',
        value_name='proportion',
    )
    civ_median = df[CIV_LABEL].median()
    state_median = df[STATE_LABEL].median()
    logger.info(f"Median civilian proportion: {civ_median:.3f}")
    logger.info(f"Median state proportion: {state_median:.3f}")
    print(df_long.filter(pl.col('proportion_type') == CIV_LABEL)
                 .group_by('year').agg(pl.col('proportion').median().alias('median_proportion'))
                 .sort('median_proportion', descending=True).head(5))

    theme = load_theme("boxy_light")
    theme.apply()

    # --- 1. median proportion by year ---
    plt.figure()
    sns.lineplot(data=df_long.to_pandas(), x='year', y='proportion', hue='proportion_type',
                 marker='o', errorbar=('se', 1), estimator=np.median)
    plt.axhline(y=civ_median, color='blue', linestyle='--', label='Civilian-aligned overall median')
    plt.axhline(y=state_median, color='orange', linestyle='--', label='Punishment bureaucracy overall median')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.ylabel('Proportion of article')
    plt.title('Proportion of article devoted to perspectives of each group')
    plt.legend()
    plt.tight_layout()
    plt.savefig('unsupervised_narration_distribution_over_time.png', dpi=300)

    # --- 2. distribution of proportions per actor ---
    plt.figure()
    box_frame = df_long.with_columns(pl.col('proportion_type').replace(SHORT_NAMES).alias('entity_type'))
    sns.boxplot(data=box_frame.to_pandas(), x='entity_type', y='proportion')
    plt.title('Average point-of-view representation')
    plt.ylabel('Proportion of article')
    plt.xlabel('Actor')
    plt.tight_layout()
    plt.savefig('unsupervised_narration_average_proportions.png', dpi=300)

    # --- 3. median civilian vs state share for the top outlets, renormalized to sum to 1 ---
    outlet_medians = (df.filter(pl.col('outlet').is_in(TOP_OUTLETS))
                        .group_by('outlet')
                        .agg(pl.col(CIV_LABEL).median(), pl.col(STATE_LABEL).median())
                        .with_columns((pl.col(CIV_LABEL) + pl.col(STATE_LABEL)).alias('total'))
                        .filter(pl.col('total') > 0)
                        .with_columns(pl.col(CIV_LABEL) / pl.col('total'), pl.col(STATE_LABEL) / pl.col('total')))
    for row in outlet_medians.iter_rows(named=True):
        logger.info(f"{row['outlet']} - civilian: {row[CIV_LABEL]:.3f}, state: {row[STATE_LABEL]:.3f}")
    bar_frame = outlet_medians.unpivot(index='outlet', on=[CIV_LABEL, STATE_LABEL],
                                       variable_name='actor', value_name='share')
    plt.figure()
    sns.barplot(data=bar_frame.to_pandas(), y='outlet', x='share', hue='actor', errorbar=None,
                order=[o for o in TOP_OUTLETS if o in outlet_medians['outlet'].to_list()])
    plt.xlabel('Share of median perspective (civilian + state = 1)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('unsupervised_narration_median_proportions_top_outlets.png', dpi=300)

@click.command()
def _produce_trajectories_individual_record(v_id, outlet, df):
    # filter the df for the given v_id and outlet
    df_filtered = df.filter((pl.col('v_id') == v_id) & (pl.col('outlet') == outlet)).sort('publication_date')
    print(df_filtered)

    # plot the trajectory of the proportions over time for this record
    plt.figure()
    sns.lineplot(data=df_filtered.to_pandas(), x='publication_date', y='Civillian advocate entities', marker='o', label='Civillian advocate entities')
    sns.lineplot(data=df_filtered.to_pandas(), x='publication_date', y='Punishment bureaucrats', marker='o', label='Punishment bureaucrats')
    plt.xticks(rotation=45)
    plt.xlabel('Incident Date')
    plt.ylabel('Proportion of article')
    plt.title(f'Narration trajectory for victim ID {v_id} in {outlet}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'narration_trajectory_{v_id}_{outlet.replace(" ", "_")}.png', dpi=300)

@click.command()
def compute_perspective_distributions_timc_v2():
    CIV_LABEL = 'Civilian advocate entities'
    STATE_LABEL = 'Punishment bureaucrats'
    SHORT_NAMES = {CIV_LABEL: 'Civilian', STATE_LABEL: 'State official'}
    TOP_OUTLETS = ['Toronto Star', 'The Globe and Mail', 'CBC', 'Global News', 'CTV']

    TIMC_V2_PATH = CONFIG['TIMC_V2_PATH']
    predictions_save_dir = 'timc_v2_qwen_predictions'

    def prediction_path(incident_id):
        return os.path.join(predictions_save_dir, f"{incident_id}.json")

    # victim ids are per-incident metadata from lookup.csv, as in get_extended_article_distribution()
    lookup_frame = pl.read_csv(f'{TIMC_V2_PATH}/lookup.csv').filter(
        (pl.col('TRACKING_INJUSTICE_CASE_ID').is_not_null()) |
        (pl.col('CUSTODY_DEATHS_ID').is_not_null())
    )
    lookup_by_incident = {row['INCIDENT_ID']: row for row in lookup_frame.iter_rows(named=True)}

    incident_article_pairs = get_all_incidents_and_articles(TIMC_V2_PATH)

    predictions_by_incident = {}                                 # incident_id -> "articles" list, loaded once
    n_missing = 0
    rows = []
    for incident_id, article_index in incident_article_pairs:
        if incident_id not in predictions_by_incident:
            if os.path.exists(prediction_path(incident_id)):
                with open(prediction_path(incident_id)) as f:
                    predictions_by_incident[incident_id] = json.load(f)['articles']
            else:
                predictions_by_incident[incident_id] = []
        articles = predictions_by_incident[incident_id]
        entry = articles[article_index] if article_index < len(articles) else None
        if entry is None:                                        # no prediction for this article
            n_missing += 1
            continue

        article_obj = obtain_article_from_timc(TIMC_V2_PATH, incident_id, article_index)
        published_at = article_obj['published_at'] or ''
        year = published_at.split('-')[0]
        if not year.isdigit():
            logger.warning(f"{incident_id}_{article_index}: unparseable publication year {year!r}; skipped")
            continue
        labels = [x for x in entry['labels'] if x != 'filtered']   # boilerplate lines don't count
        n = len(labels)
        if n == 0:
            continue
        lookup_row = lookup_by_incident.get(incident_id)
        rows.append({
            'year': int(year),
            'publication_date': published_at,
            'outlet': article_obj['publication'],
            'v_id': lookup_row['LEGACY_VICTIM_ID'] if lookup_row else None,
            'article_id': f"{incident_id}_{article_index}",
            'num_paragraphs': n,
            CIV_LABEL: sum(x == 'civilian-aligned' for x in labels) / n,
            'no_entity_proportion': sum(x == 'no entity' for x in labels) / n,
            STATE_LABEL: sum(x == 'state-aligned' for x in labels) / n,
        })
    logger.info(f"{len(rows)} articles with predictions; {n_missing} of {len(incident_article_pairs)} have none")
    df = pl.DataFrame(rows)

    # --- example articles ---
    long_enough = pl.col('num_paragraphs') >= 5
    print("Articles with high punishment bureaucrat representation:")
    print(df.with_columns((pl.col(STATE_LABEL) - pl.col(CIV_LABEL)).alias('proportion_diff'))
            .filter(long_enough & (pl.col(CIV_LABEL) > 0))
            .sort('proportion_diff', descending=True).head(10))
    print("Articles with high civilian-aligned representation:")
    print(df.with_columns((pl.col(CIV_LABEL) - pl.col(STATE_LABEL)).alias('proportion_diff'))
            .filter(long_enough & (pl.col(STATE_LABEL) > 0))
            .sort('proportion_diff', descending=True).head(10))
    print("Articles with pluralistic representation:")
    print(df.filter(long_enough & (pl.col(CIV_LABEL) > 0.1) & (pl.col(STATE_LABEL) > 0.1)))

    # One row per (article, perspective type), civilian and state only
    df_long = df.unpivot(
        index=['year', 'publication_date', 'outlet', 'v_id', 'article_id', 'num_paragraphs'],
        on=[CIV_LABEL, STATE_LABEL],
        variable_name='proportion_type',
        value_name='proportion',
    )
    civ_median = df[CIV_LABEL].median()
    state_median = df[STATE_LABEL].median()
    logger.info(f"Median civilian advocate proportion: {civ_median:.3f}")
    logger.info(f"Median punishment bureaucracy proportion: {state_median:.3f}")
    print(df_long.filter(pl.col('proportion_type') == CIV_LABEL)
                 .group_by('year').agg(pl.col('proportion').median().alias('median_proportion'))
                 .sort('median_proportion', descending=True).head(5))

    theme = load_theme("boxy_light")
    theme.apply()

    # --- 1. median proportion by year ---
    plt.figure()
    sns.lineplot(data=df_long.to_pandas(), x='year', y='proportion', hue='proportion_type',
                 marker='o', errorbar=('se', 1), estimator=np.median)
    plt.axhline(y=civ_median, color='blue', linestyle='--', label='Civilian-aligned overall median')
    plt.axhline(y=state_median, color='orange', linestyle='--', label='Punishment bureaucracy overall median')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.ylabel('Proportion of article')
    plt.title('Proportion of article devoted to perspectives of each group')
    plt.legend()
    plt.tight_layout()
    plt.savefig('extended_narration_distribution_over_time.png', dpi=300)

    # --- 2. distribution of proportions per actor ---
    plt.figure()
    box_frame = df_long.with_columns(pl.col('proportion_type').replace(SHORT_NAMES).alias('entity_type'))
    sns.boxplot(data=box_frame.to_pandas(), x='entity_type', y='proportion')
    plt.title('Average point-of-view representation')
    plt.ylabel('Proportion of article')
    plt.xlabel('Actor')
    plt.tight_layout()
    plt.savefig('extended_narration_average_proportions.png', dpi=300)

    # --- 3. median civilian vs state share for the top outlets, renormalized to sum to 1 ---
    outlet_medians = (df.filter(pl.col('outlet').is_in(TOP_OUTLETS))
                        .group_by('outlet')
                        .agg(pl.col(CIV_LABEL).median(), pl.col(STATE_LABEL).median())
                        .with_columns((pl.col(CIV_LABEL) + pl.col(STATE_LABEL)).alias('total'))
                        .filter(pl.col('total') > 0)
                        .with_columns(pl.col(CIV_LABEL) / pl.col('total'), pl.col(STATE_LABEL) / pl.col('total')))
    for row in outlet_medians.iter_rows(named=True):
        logger.info(f"{row['outlet']} - civilian: {row[CIV_LABEL]:.3f}, state: {row[STATE_LABEL]:.3f}")
    bar_frame = outlet_medians.unpivot(index='outlet', on=[CIV_LABEL, STATE_LABEL],
                                       variable_name='actor', value_name='share')
    plt.figure()
    sns.barplot(data=bar_frame.to_pandas(), y='outlet', x='share', hue='actor', errorbar=None,
                order=[o for o in TOP_OUTLETS if o in outlet_medians['outlet'].to_list()])
    plt.xlabel('Share of median perspective (civilian + state = 1)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('extended_narration_median_proportions_top_outlets.png', dpi=300)

@click.command()
def describe_annotated_dataset():
    outlets = []
    for fname in os.listdir("data/articles"):
        outlet = fname.split('_')[2][:-5]
        outlets.append(outlet)
    # print the counts of each outlet sorted descending, in latex format
    series = pd.Series(outlets)
    outlet_counts = series.value_counts(ascending=False)
    print(outlet_counts.to_latex())
      
@click.command()
def describe_dataset():
    # How many unique victim IDs are there?
    v_ids = []
    v_id_to_name = {}
    year_to_vids = defaultdict(set)
    outlets = []
    for article in os.listdir('unsupervised_articles'):
        with open(f'unsupervised_articles/{article}', 'r') as f:
            obj = json.load(f)
            v_ids.append(obj['victim_id'])
            v_id_to_name[obj['victim_id']] = obj['victim']
            date = obj['incident_date']
            year = date.split('-')[0]
            outlets.append(obj['publisher'])
            if year.isdigit():
                year_to_vids[year].add(obj['victim_id'])

    print(f"Number of unique victim IDs: {len(set(v_ids))}/{len(v_ids)}")
    # print the victims with the top 10 most articles
    from collections import Counter
    v_id_counts = Counter(v_ids)
    print("Top 10 victim IDs with most articles:") # print what year they were too
    for v_id, count in v_id_counts.most_common(20):
        print(f"{v_id_to_name[v_id]} ({v_id}): {count} articles")
        # which year this v_id from
        years = []
        for year, vids in year_to_vids.items():
            if v_id in vids:
                years.append(year)
        print(f"  Years: {', '.join(years)}")


    # get the number of unique articles per year
    print("Number of unique victim IDs per year:")
    num_deaths_sorted_by_year = []
    for year, vids in sorted(year_to_vids.items()):
        print(f"{year}: {len(vids)} unique victim IDs")
        if int(year) > 2022:
            continue
        num_deaths_sorted_by_year.append((len(vids), deaths_by_year[int(year)]))
    # check the correlation
    from scipy.stats import pearsonr
    x = [x[1] for x in num_deaths_sorted_by_year]
    y = [x[0] for x in num_deaths_sorted_by_year]
    corr, p_value = pearsonr(x, y)
    print(f"Correlation between number of deaths and number of unique victim IDs in articles: {corr} (p-value: {p_value})")
    # plot the correlation, and label the year
    plt.figure()
    plt.scatter(x, y)
    for i, (deaths, vids) in enumerate(num_deaths_sorted_by_year):
        plt.text(vids, deaths, str(2000 + i))
    plt.xlabel('Number of deadly force incidents (Tracking Injustice database)')
    plt.ylabel('Number of unique victim IDs in articles')
    plt.title(f'Correlation between number of deadly force incidents and unique victim IDs in articles ($r={corr:.2f}$)')
    plt.tight_layout()
    plt.savefig('deaths_vs_unique_victim_ids_correlation.png', dpi=300, bbox_inches='tight')

    # produce a CSV with victim ID, title, outlet, and URL
    with open('unsupervised_article_metadata.csv', 'w') as f:
        f.write('victim_id,title,outlet,url\n')
        for article in os.listdir('unsupervised_articles'):
            with open(f'unsupervised_articles/{article}', 'r') as f_article:
                obj = json.load(f_article)
                victim_id = obj['victim_id']
                try:
                    title = obj['title'].replace(',', ' ')
                except:
                    title = "N/A"
                outlet = obj['publisher'].replace(',', ' ')
                url = obj['url']
                f.write(f"{victim_id},{title},{outlet},{url}\n")
    # compute the median number of deaths per year
    death_counts = [len(vids) for year, vids in sorted(year_to_vids.items()) if year.isdigit() and int(year) <= 2022]
    print(f"Median number of unique victim IDs per year: {pd.Series(death_counts).median()}")

    # Compute the median number of articles per outlet.
    outlet_counts = Counter(outlets)
    print(f"Median number of articles per outlet: {pd.Series(list(outlet_counts.values())).median()}")
    # what is the top 5 outlets
    print("Top 5 outlets by number of articles:")
    for outlet, count in outlet_counts.most_common(5):
        print(f"{outlet}: {count} articles")
    # what are the number of outlets with only one article
    num_outlets_with_one_article = sum([1 for outlet, count in outlet_counts.items() if count == 1])
    print(f"Number of outlets with only one article: {num_outlets_with_one_article}")
    # print 10 of those outlets
    print("10 outlets with only one article:")
    one_article_outlets = [outlet for outlet, count in outlet_counts.items() if count == 1]
    for outlet in one_article_outlets[:10]:
        print(outlet)
    



@click.command()
def compute_ngram_occurrences_unsupervised_dataset():
    punishment_bureaucrat_terms = [
        'asirt', 'police chief', 'coroner', 
        'crown prosecutor', ''
        'union', 'mayor', 'councilor', 
        'defence lawyer'
    ]

    civillian_aligned_terms = [
        'family lawyer', "family's lawyer", 'brother', 
        'lawyer for the',
        'advocate', 'organization'
    ] 

    # go through all the unsupervised articles. Find the number of occurrences of each term
    # as well as the percentage of articles the term shows up in.
    term_to_num_occurrences = {term: 0 for term in punishment_bureaucrat_terms + civillian_aligned_terms}
    term_to_num_articles = {term: 0 for term in punishment_bureaucrat_terms + civillian_aligned_terms}
    all_articles = os.listdir('unsupervised_articles')
    for article in tqdm(all_articles):
        with open(f'unsupervised_articles/{article}', 'r') as f:
            paragraphs = json.load(f)['article']
            full_text = " ".join(paragraphs).lower()
            for term in punishment_bureaucrat_terms + civillian_aligned_terms:
                count = full_text.count(term)
                term_to_num_occurrences[term] += count
                if count > 0:
                    term_to_num_articles[term] += 1
    for term in punishment_bureaucrat_terms:
        print(f"Term '{term}' occurred {term_to_num_occurrences[term]} times in total, across {term_to_num_articles[term]} articles ({term_to_num_articles[term]/len(all_articles)*100:.2f}%)")
    print("---")
    for term in civillian_aligned_terms:
        print(f"Term '{term}' occurred {term_to_num_occurrences[term]} times in total, across {term_to_num_articles[term]} articles ({term_to_num_articles[term]/len(all_articles)*100:.2f}%)")

@click.command()
def find_same_outlet_followups():
    v_
    for article in os.listdir('unsupervised_articles'):
        with open(f'unsupervised_articles/{article}', 'r') as f:
            obj = json.load(f)
            v_ids.append(obj['victim_id'])
            v_id_to_name[obj['victim_id']] = obj['victim']
            date = obj['incident_date']
            year = date.split('-')[0]
            if year.isdigit():
                year_to_vids[year].add(obj['victim_id'])

@click.command()
def generate_article_hicode_labels():
    TIMC_V2_PATH = CONFIG['TIMC_V2_PATH']
    with open(f"{TIMC_V2_PATH}/qwen_filters_suffix_32B.json") as f:
        filter_json = json.load(f)

    def generate_classification_prompt(location, person, article):
        prompt = f"""
I'm working on classifying the types of articles published in coverage of police-involved deaths or custody deaths in Canada.

Consider the following news article about an incident in {location} about {person}:
```
{article}
```
What type of news article is this? The initial report of the incident? Reporting on outcome of an oversight agency investigation? Profile of the victim’s family? A coroner’s inquest? Something else? A combination of one or more things? Return your response in the form of a JSON dictionary where the key is a label (a short descriptor), and the value is  a one or two sentence explanation of the label. Only respond with this dictionary, nothing else.
"""
        return prompt
    
    loader = incident_loader(TIMC_V2_PATH)
    generator = load_model_generator("Qwen/Qwen3-32B", 'cuda')
    HICODE_DIR = 'hicode_generations'
    os.makedirs(HICODE_DIR, exist_ok=True)
    for province, incident_code in tqdm(MEDIA_STORM_PROVINCE_MAP.items(), desc='Incidents'):
        responses = []
        with open(f"{TIMC_V2_PATH}/{incident_code}.json") as f:
            # TODO: call obtain article from TIMC here.
            articles = json.load(f)['articles']
            for i in tqdm(range(len(articles)), desc=f"{province}-{incident_code} articles", leave=False):
                article_lines = obtain_article_from_timc(
                    TIMC_V2_PATH, 
                    incident_code, i, 
                    filter_json
                )['maintext']
                article = "\n".join(article_lines)
                metadata = loader(incident_code)
                prompt = generate_classification_prompt(province, metadata['NAME'], article)
                response = generator(prompt)
                responses.append(response)
        with open(os.path.join(HICODE_DIR, f"{province}_{incident_code}.json"), 'w') as f:
            json.dump(responses, f)

@click.group()
def main():
    pass

main.add_command(describe_annotated_dataset)
main.add_command(describe_dataset)
main.add_command(inspect_distributions_unsupervised)
main.add_command(compute_fastcoref_annotations) # run this first
main.add_command(compute_article_basis_coref_objects)
main.add_command(create_coref_inference_dataset) 
main.add_command(compute_perspective_distributions_timc_v1)
main.add_command(compute_ngram_occurrences_unsupervised_dataset)
main.add_command(compute_perspective_distributions_timc_v2)
main.add_command(generate_article_hicode_labels)
# main.add_command(find_same_outlet_followups)

if __name__ == '__main__':
    main()

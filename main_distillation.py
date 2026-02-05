from tqdm import tqdm
from scipy.stats import spearmanr,pearsonr
import polars as pl
import numpy as np
from collections import defaultdict
import pandas as pd
import random
import pathlib
import torch
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score, confusion_matrix
import ipdb
from functools import partial
import loguru
import json
import os
import click
from dotenv import dotenv_values
from typing import List, Dict, Optional, Iterable


from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForTokenClassification
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from packages.constants import DEV_SUBJECTS
from packages.parsing_utils import CorefEntityMetadata, get_manual_annotation_occurrences, load_all_articles,\
    load_article_paragraphs, load_repaired_articles,\
    load_training_data_annotations_for_person, extract_enumerated_paragraphs,\
    compute_paragraph_to_affinities, normalize_whitespace, load_unsupervised_article_paragraphs
from packages.flan_utils import compute_metrics_tokenized_batch, generate_singleton_prediction,\
    evaluate_entity_identified_single, generate_predictions,\
    generate_predictions_tokenized_batch, convert_text_to_entity_present_label,\
    is_valid_entity_present, is_police_aligned_entity, extract_relevant_paragraphs,\
    convert_to_ternary_label_list, reduce_affinities_to_individual_prediction,\
    convert_to_ternary_label_list_inference

config = dotenv_values(".env")
logger = loguru.logger
# message = ["Language modeling is "]
# inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# # optional verifying cuda
# # inputs = {k: v.to('cuda') for k,v in inputs.items()}
# # olmo = olmo.to('cuda')
# response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])


def preprocess_function(tokenizer, sample):
    model_inputs = tokenizer(sample['prompt']) # don't pad in preprocessing
    label_str = f"{sample['completion']}"
    # json stringifying the label_str
    labels = tokenizer((label_str))
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         # [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #         (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
    #     ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class CustomSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs['tokenizer']

    def evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ):
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            # Perform decoding and loss calculations here
            model = self.model
            tokenizer = self.tokenizer
            all_eval_entity_present_gt_labels = []
            all_eval_entity_present_prediction_labels = []
            all_is_correct_labels = []
            all_cer_metrics = []
            for i, _data in enumerate(eval_dataloader): # should be batch size set by trainer.
                batch = generate_predictions_tokenized_batch(model, tokenizer, _data)
                batch_metrics = compute_metrics_tokenized_batch(batch)
                batch_entity_present_gt_labels = convert_text_to_entity_present_label(batch['label_text']) # TODO: implement this function
                batch_entity_present_predicted_labels = convert_text_to_entity_present_label(batch['predicted_text']) # TODO: implement this function
                all_eval_entity_present_gt_labels.extend(batch_entity_present_gt_labels)
                all_eval_entity_present_prediction_labels.extend(batch_entity_present_predicted_labels)
                all_cer_metrics.extend(batch_metrics['cer']) 
                all_is_correct_labels.extend(batch_metrics['entity_present_correct'])
            labels = ['valid entity', 'no entity', 'unrecognized']
            f1_metric_dict_3_labels = f1_score(all_eval_entity_present_gt_labels, all_eval_entity_present_prediction_labels, average=None, labels=labels)
            accuracy_metric = sum(all_is_correct_labels) / len(all_is_correct_labels) # TODO
            cer_metric = np.median(all_cer_metrics)
            metrics = {'cer': cer_metric, 'accuracy': accuracy_metric, 'f1': f1_metric_dict_3_labels[labels.index('valid entity')]} # TODO: fix the value for f1
            logger.info(f"Eval metrics: {metrics}")
            return metrics


def tokenize_batch_flan_fn(tokenizer, samples):
    model_inputs = tokenizer(samples['prompt'], padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(samples['completion'], padding=True, truncation=True, return_tensors="pt")['input_ids']
    model_inputs['labels'] = labels
    return model_inputs

def tokenize_batch_flan_fn_inference(tokenizer, samples):
    model_inputs = tokenizer(samples['prompt'], padding=True, truncation=True, return_tensors="pt")
    return model_inputs

def compute_metrics_flan():
    pass

@click.command()
def distill_flant5():

    FLAN_TOKENIZER = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", 
        cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache")
    )

    train_dataset = load_dataset("json", data_files={'train': "data/distillation_data/coref_training_dataset.json"}, split='train')
    dev_dataset = load_dataset("json", data_files={'test': "data/distillation_data/coref_dev_dataset.json"}, split='test')

    # check the fraction of valid entities in the train and eval sets
    logger.info(f"Train valid entities proportion: {sum(train_dataset['valid_entity'])} / {len(train_dataset)}")
    logger.info(f"Eval valid entities proportion: {sum(dev_dataset['valid_entity'])} / {len(dev_dataset)}")

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    new_tokens = ["{", "}"]
    new_tokens = set(new_tokens) - set(FLAN_TOKENIZER.vocab.keys())
    new_tokens = list(new_tokens)
    FLAN_TOKENIZER.add_tokens(new_tokens)
    model.resize_token_embeddings(len(FLAN_TOKENIZER))
    train_dataset = train_dataset.map(
        partial(tokenize_batch_flan_fn, FLAN_TOKENIZER), 
        batched=True
    )
        # preprocess_flan_fn, 
        # remove_columns=['output', 'subject', 'outlet', 'current_mentioned_entities'],
    dev_dataset = dev_dataset.map(
        partial(tokenize_batch_flan_fn, FLAN_TOKENIZER), 
        batched=True
    )
    wandb.init(project="sympathy")
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=os.path.join(config['SCRATCH_DIR'], "sympathy_distillation"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=2000,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
        report_to="wandb"
    )
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        FLAN_TOKENIZER,
        model=model,
        label_pad_token_id=label_pad_token_id, 
        padding=True, 
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator, 
        tokenizer=FLAN_TOKENIZER
    )
    trainer.train(resume_from_checkpoint=False)

@click.group()
def main():
    pass

# Define hyperparameter ranges
param_ranges = {
    "learning_rate": (1e-5, 1e-3),   # float range (log-uniform recommended)
    "weight_decay": (0.0, 0.3),      # float range
    "training_steps": (100, 200)       # int range
}

def random_hyperparams(n_trials=10, seed=None):
    if seed is not None:
        random.seed(seed)

    configs = []
    for _ in range(n_trials):
        training_steps = random.randint(*param_ranges["training_steps"])
        config = {
            # log-uniform sampling for learning rate
            "learning_rate": 10 ** random.uniform(-5, -3),  
            "weight_decay": random.uniform(*param_ranges["weight_decay"]),
            "training_steps": training_steps,
            "warmup_steps": random.randint(0, training_steps - 1)  # strictly less
        }
        configs.append(config)
    return configs

@click.command()
def compute_required_memory():
    model = get_model("olmo-7b") 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Memory for weights (float16): {num_params * 2 / 1024**3:.2f} GB")
    pass

@click.command()
def assess_baseline_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache"))
    eval_dataset = load_dataset("json", data_files={'test': "data/distillation_data/distill_examples.json"}, split='test')

    eval_dataset = eval_dataset.map(
        lambda samples: tokenizer(samples['prompt'], padding=True, truncation=True, return_tensors="pt"), 
        batched=True,
    )
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

def get_predicted_article_paragraph_mappings(article_dataset: Dataset,
                                             article_name: str,
                                             article_folder_prefix: str) -> List[str]: # dataset filtered by article index
    paragraphs_ordered = [normalize_whitespace(paragraph) for paragraph in load_unsupervised_article_paragraphs(article_name, f"{article_folder_prefix}_articles")]
    paragraph_index_to_assignments = defaultdict(list)
    for i in range(len(article_dataset)):
        paras_extracted = extract_enumerated_paragraphs(article_dataset['prompt'][i])
        coref_prediction_text = article_dataset['predicted_text'][i]
        if is_valid_entity_present(coref_prediction_text) == 'valid entity':
            relevant_paragraph_indices = extract_relevant_paragraphs(coref_prediction_text)
            try:
                relevant_paragraphs = [normalize_whitespace(paras_extracted[idx - 1]) for idx in relevant_paragraph_indices if idx - 1 < len(paras_extracted)]
            except IndexError:
                logger.warning(f"Index error for {article_name} with indices {relevant_paragraph_indices} and paragraphs {paras_extracted}")
                ipdb.set_trace()
            try:
                whole_article_indices = [paragraphs_ordered.index(para) + 1 for para in relevant_paragraphs]
            except ValueError as e:
                # print traceback of e
                logger.warning(f"Value error for {article_name} with paragraphs {relevant_paragraphs}: {e}")
                raise e

            police_aligned = is_police_aligned_entity(coref_prediction_text)
            for index in whole_article_indices:
                paragraph_index_to_assignments[index]\
                    .append('police-aligned' if police_aligned else 'victim-aligned')
    pred_paragraph_to_affinities = reduce_affinities_to_individual_prediction(paragraph_index_to_assignments)
    y_pred = convert_to_ternary_label_list_inference(pred_paragraph_to_affinities, len(paragraphs_ordered))
    return y_pred

def evaluate_proportion_distribution_metric(dataset): #TODO: might have accidentally broken this at 9PM on Wednesday Oct 1 
    article_indices = dataset['article_index']
    unique_article_indices = set(article_indices)
    all_y_true = []
    all_y_pred = []
    aligned_ratios_gt = [] 
    aligned_ratios_predicted = []
    critical_ratios_gt = []
    critical_ratios_predicted = []

    for index in unique_article_indices:
        article_subset = dataset.filter(lambda example: example['article_index'] == index)
        article = f"{index}_{article_subset[0]['victim_name']}_{article_subset[0]['outlet']}"
        article = article + ".json" if not article.endswith('.json') else article
        gt_annotations = load_training_data_annotations_for_person(
            person_name=article_subset[0]['victim_name'], 
            outlet=article_subset[0]['outlet'], 
            identifier=index
        )
        paragraphs_ordered = [normalize_whitespace(paragraph) for paragraph in load_article_paragraphs(article)]
        paras_extracted = [normalize_whitespace(paragraph) for paragraph in extract_enumerated_paragraphs(article_subset['prompt'][0])]
        for para in paras_extracted:
            assert para in paragraphs_ordered, f"Extracted paragraph not in original paragraphs: {para}"
        logger.info(f"Good for {article}")
        # get the coref entities that were classified as valid. 
        paragraph_index_to_assignments = defaultdict(list)
        for i in range(len(article_subset)):
            paras_extracted = extract_enumerated_paragraphs(article_subset['prompt'][i])
            coref_prediction_text = article_subset['predicted_text'][i]
            if is_valid_entity_present(coref_prediction_text) == 'valid entity':
                relevant_paragraph_indices = extract_relevant_paragraphs(coref_prediction_text)
                try:
                    relevant_paragraphs = [normalize_whitespace(paras_extracted[idx - 1]) for idx in relevant_paragraph_indices if idx - 1 < len(paras_extracted)]
                except IndexError:
                    logger.warning(f"Index error for {article} with indices {relevant_paragraph_indices} and paragraphs {paras_extracted}")
                    ipdb.set_trace()
                try:
                    whole_article_indices = [paragraphs_ordered.index(para) + 1 for para in relevant_paragraphs]
                except ValueError as e:
                    # print traceback of e
                    logger.warning(f"Value error for {article} with paragraphs {relevant_paragraphs}: {e}")

                    ipdb.set_trace()


                police_aligned = is_police_aligned_entity(coref_prediction_text)
                for index in whole_article_indices:
                    paragraph_index_to_assignments[index]\
                        .append('police-aligned' if police_aligned else 'victim-aligned')
        gt_paragraph_to_affinities = compute_paragraph_to_affinities(gt_annotations)
        pred_paragraph_to_affinities = reduce_affinities_to_individual_prediction(paragraph_index_to_assignments)
        y_true, y_pred = convert_to_ternary_label_list(gt_paragraph_to_affinities, pred_paragraph_to_affinities, len(paragraphs_ordered))
        aligned_ratios_gt.append(y_true.count('police-aligned') / len(y_true))
        aligned_ratios_predicted.append(y_pred.count('police-aligned') / len(y_pred))
        critical_ratios_gt.append(y_true.count('victim-aligned') / len(y_true))
        critical_ratios_predicted.append(y_pred.count('victim-aligned') / len(y_pred))
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    logger.info("Showing correlations between article-level ratios")
    print(f"Police-aligned ratio correlation: {pearsonr(aligned_ratios_gt, aligned_ratios_predicted).statistic:.2f}")
    print(f"Victim-aligned ratio correlation: {pearsonr(critical_ratios_gt, critical_ratios_predicted).statistic:.2f}")
    # print the individual four arrays
    print("Police-aligned ratios GT:", aligned_ratios_gt)
    print("Police-aligned ratios Predicted:", aligned_ratios_predicted)
    print("Victim-aligned ratios GT:", critical_ratios_gt)
    print("Victim-aligned ratios Predicted:", critical_ratios_predicted)

    report = classification_report(all_y_true, all_y_pred, labels=['police-aligned', 'victim-aligned', 'no entity'], output_dict=True)
    print(report)
    # get the weighted f1 score
    weighted_f1 = report['weighted avg']['f1-score']
    # compute a boostrapped test for the confidence interval, checking whether it is significantly less than 0.78
    print(f"Weighted F1 score: {weighted_f1}")
    num_greater = 0
    for b in range(1000):
        indices = np.random.choice(len(all_y_true), len(all_y_true), replace=True)
        y_true_sampled = [all_y_true[i] for i in indices]
        y_pred_sampled = [all_y_pred[i] for i in indices]
        report_sampled = classification_report(y_true_sampled, y_pred_sampled, labels=['police-aligned', 'victim-aligned', 'no entity'], output_dict=True)
        weighted_f1_sampled = report_sampled['weighted avg']['f1-score']
        if weighted_f1_sampled >= 0.78:
            num_greater += 1
    p_value = num_greater / 1000
    print(f"P-value for weighted F1 >= 0.78: {p_value}")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=['police-aligned', 'victim-aligned', 'no entity'])
    print("Confusion Matrix:")
    print(cm)


    random.seed(42)
    all_y_random = []
    for _ in range(len(all_y_true)):
        all_y_random.append(random.choice(['police-aligned', 'victim-aligned', 'no entity']))
    random_report = classification_report(all_y_true, all_y_random, labels=['police-aligned', 'victim-aligned', 'no entity'])
    print("Random classifier report:")
    print(random_report)


    # compute the confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=['police-aligned', 'victim-aligned', 'no entity'])
    print("Confusion Matrix:")
    print(cm)
    # number of 

    # compute a dummy classifier that predicts randomly
    random.seed(42)
    all_y_random = []
    for _ in range(len(all_y_true)):
        all_y_random.append(random.choice(['police-aligned', 'victim-aligned', 'no entity']))
    random_report = classification_report(all_y_true, all_y_random, labels=['police-aligned', 'victim-aligned', 'no entity'])
    print("Random classifier report:")
    print(random_report)


@click.command()
def assess_ft_flan_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", 
        cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache")
    )
    flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=os.path.join(
            config['SCRATCH_DIR'], 
            "sympathy_distillation", 
            "checkpoint-1300")
    ).to('cuda')

    eval_dataset = load_dataset("json", 
                           data_files={'train': "data/distillation_data/coref_test_dataset.json"},
                           split='train')

    def evaluate_entity_identified_batch(example): # not batched
        no_entity_str = "there is no valid entity providing a perspective here."
        entity_present_str = "the entity name is"
        ground_truth = example['completion'].lower()
        prediction = example['predicted_text'].lower()
        if entity_present_str in ground_truth:
            if entity_present_str in prediction:
                is_correct = True
            else:
                is_correct = False
        elif no_entity_str in ground_truth:
            if no_entity_str in prediction:
                is_correct = True
            else:
                is_correct = False
        else:
            logger.warning(f"Ground truth not in expected format: {ground_truth}")
            raise ValueError(f"Ground truth not in expected format: {ground_truth}")
        example['entity_identified_correct'] = is_correct
        return example

    eval_dataset = eval_dataset.map(
        partial(tokenize_batch_flan_fn, tokenizer), 
        batched=True
    )
    assert 'input_ids' in eval_dataset.column_names
    assert 'labels' in eval_dataset.column_names
    eval_dataset = eval_dataset.map(partial(generate_predictions, flan_t5, tokenizer),
                                    batched=True, 
                                    batch_size=16)
    eval_dataset = eval_dataset.map(evaluate_entity_identified_batch)
    evaluate_proportion_distribution_metric(eval_dataset)


@click.command()
@click.argument("input_file_prefix", type=str)
def run_large_scale_inference(input_file_prefix):
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", 
        cache_dir=os.path.join(config['SCRATCH_DIR'], "transformers_cache")
    )
    flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(
        "smfsamir/perspective-gap"
        # pretrained_model_name_or_path=os.path.join(
        #     config['SCRATCH_DIR'], 
        #     "sympathy_distillation", 
        #     "checkpoint-1300")
    ).to('cuda')

    inference_dataset = load_dataset("json", 
                           data_files={'train': f"data/distillation_data/{input_file_prefix}_coref_inference_dataset.json"},
                           split='train')

    inference_dataset = inference_dataset.map(
        partial(tokenize_batch_flan_fn_inference, tokenizer),
        batched=True
    )
    assert 'input_ids' in inference_dataset.column_names
    inference_dataset = inference_dataset.map(partial(generate_predictions, flan_t5, tokenizer),
                                    batched=True,
                                    batch_size=2)
    inference_dataset.to_json(f"data/distillation_data/{input_file_prefix}_coref_inference_with_predictions.json")
    logger.info(f"Wrote inference dataset with predictions to data/distillation_data/{input_file_prefix}_coref_inference_with_predictions.json")

def construct_length_limited_prompt(victim_name: str, 
                                    coref_entity_obj: CorefEntityMetadata,
                                    all_paragraphs: List[str], 
                                    coref_auto_indices: List[int], 
                                    annotation_indices: List[int]) -> Dict:
    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    #### Constructing the input
    coref_auto_paragraphs = "\n".join([f"{i+1}. {all_paragraphs[index - 1]}" for i, index in enumerate(coref_auto_indices)])
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 
    index_to_auto_paragraph_index = {index: i+1 for i, index in enumerate(coref_auto_indices)}
    intersection_indices = list(sorted(set(annotation_indices).intersection(set(coref_auto_indices))))

    preamble_str = f"This is an article about the killing of {victim_name} by police." 
    subset_indices = [index_to_auto_paragraph_index[index] for index in intersection_indices]
    #### Constructing the input
    task_instruction_str = preamble_str +\
        f" Here are references to a potential entity: {coref_entity_obj.cluster_strings}\n" +\
        f" Here are the paragraphs that mention them:\n" +\
        f"{coref_auto_paragraphs}\n\n" +\
        f" Parse whether there is a valid entity, and, if so, what the entity name is whether they're aligned with the police, and which paragraphs reflect their perspectives." 

    if coref_entity_obj.valid_entity: 
        # if len(intersection_indices) == 0: # TODO: this is more likely to happen, since you're not doing the split...
        # output_str = json.dumps({'entity_name': coref_entity_obj.entity_name, 'police_aligned': coref_entity_obj.police_aligned,'perspective_paragraphs': subset_indices})
        # output_str = json.dumps({'entity_name': coref_entity_obj.entity_name, 'police_aligned': coref_entity_obj.police_aligned,'perspective_paragraphs': subset_indices})
        # TODO: try a natural language output
        output_str = f"The entity name is {coref_entity_obj.entity_name}. They are {'aligned with the police' if coref_entity_obj.police_aligned == 'yes' else 'not aligned with the police'}. The paragraphs that reflect their perspectives are: {', '.join(map(str, subset_indices)) if len(subset_indices) > 0 else 'none'}."
    else:
        output_str = f"There is no valid entity providing a perspective here."
    return {'prompt': task_instruction_str, 'completion': output_str, 'entity_name': coref_entity_obj.entity_name, 'valid_entity': coref_entity_obj.valid_entity, victim_name: victim_name}

def _create_training_instance(victim_name: str, 
                              all_paragraphs: List[str], 
                              coref_metadata_obj: CorefEntityMetadata, 
                              training_annotation_entity: Dict) -> List[Dict]:
    # TODO: need to return multiple instances, containing at most 5 paragraphs.

    MAX_PARAGRAPHS = 5
    # TODO: get the intersection of the paragraphs
    ## TODO: watch out for multiple paragraphs
    unique_auto_indices = list(sorted(list(set(coref_metadata_obj.auto_paragraph_indices))))
    num_prompts_required = len(unique_auto_indices) // MAX_PARAGRAPHS + (1 if len(unique_auto_indices) % MAX_PARAGRAPHS > 0 else 0)
    training_instances = []
    all_perspective_paragraphs_empty = True
    for i in range(num_prompts_required):
        coref_indices = unique_auto_indices[i*MAX_PARAGRAPHS:(i+1)*MAX_PARAGRAPHS]
        training_instances.append(construct_length_limited_prompt(
            victim_name=victim_name,
            coref_entity_obj=coref_metadata_obj,
            all_paragraphs=all_paragraphs,
            coref_auto_indices=coref_indices,
            annotation_indices=get_manual_annotation_occurrences(training_annotation_entity, coref_metadata_obj.entity_name) if coref_metadata_obj.valid_entity else []
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

def create_hf_dataset_from_articles(articles: List[str], 
                                    repaired_set: Optional[List[str]] = []) -> Dataset:
    training_set = []
    victim_names = []
    outlets = []
    article_indices = []
    for article in articles:
        # load the coref object and the annotation object
        article_index = article.split('_')[0]
        person_name = article.split('_')[1]
        outlet = article.split('_')[2]
        annotations = load_training_data_annotations_for_person(person_name, outlet, identifier=article_index)
        # coref_metadata_objects = [CorefEntityMetadata(**obj) for obj in json.load(open(os.path.join("data/coref_metadata", article)))]
        paragraphs = load_article_paragraphs(article)

        if article not in repaired_set:
            coref_metadata_objects = [CorefEntityMetadata(**obj) for obj in json.load(open(os.path.join("data/linked_coref_annotations", article)))]
        # TODO: load the repaired articles here, separately.
        else:
            coref_metadata_objects= [CorefEntityMetadata(**obj) for obj in json.load(open(os.path.join("data/repaired_coref_annotations", article.replace('.json', '_repaired.json'))))]
        for coref_obj in coref_metadata_objects:
            training_instances = _create_training_instance(
                victim_name=person_name,
                all_paragraphs=paragraphs,
                coref_metadata_obj=coref_obj, # just do the first one for now
                training_annotation_entity=annotations
            )
            training_set.extend(training_instances)
            victim_names.extend([person_name] * len(training_instances))
            outlets.extend([outlet] * len(training_instances))
            article_indices.extend([article_index] * len(training_instances))
    dataset = Dataset.from_dict({
        'prompt': [instance['prompt'] for instance in training_set],
        'completion': [instance['completion'] for instance in training_set],
        'entity_name': [instance['entity_name'] for instance in training_set],
        'valid_entity': [instance['valid_entity'] for instance in training_set],
        'outlet': outlets,
        'article_index': article_indices,
        'victim_name': victim_names
    })
    return dataset

    
@click.command()
def create_coref_training_dataset():

    def _remove_repaired_suffix(article_name: str) -> str:
        assert '_repaired' in article_name, f"Article name {article_name} does not contain '_repaired'"
        return article_name.replace('_repaired', '')
    # write a function to create a coreference resolution training dataset.

    split_to_article = obtain_train_eval_test_split()
    ipdb.set_trace()
    train_articles = split_to_article['train']
    dev_articles = split_to_article['dev']
    test_articles = split_to_article['test']
    repaired_articles = split_to_article['repaired (train)']

    # log the number of valid entities relative to the total
    train_dataset = create_hf_dataset_from_articles(train_articles, repaired_set=[_remove_repaired_suffix(article) for article in repaired_articles]) 
    dev_dataset = create_hf_dataset_from_articles(dev_articles)
    test_dataset = create_hf_dataset_from_articles(test_articles)
    train_dataset.to_json("data/distillation_data/coref_training_dataset.json")
    dev_dataset.to_json("data/distillation_data/coref_dev_dataset.json")
    test_dataset.to_json("data/distillation_data/coref_test_dataset.json")
    return

def obtain_train_eval_test_split():
    repaired_articles = load_repaired_articles()
    perfect_ratio_articles = ['23_Charles Qirngnirq_CBC.json', '47_Raymond Alliman_York Region.json',
                              '79_Buck E Evans_Edmonton Journal.json', '40_Raymond Alliman_Ottawa Citizen.json',
                              '51_Ralph Stephens_Calgary Sun.json', '61_Abisay Cruz_CityNews Montreal.json',
                              '8_Vitaly Savin_CBC.json', '77_Tommy Ningiuk_Nunatsiaq News.json',
                              '63_Pierre Charron_Ottawa Citizen.json', '84_Buck E Evans_CBC.json',
                              '86_David-Huges Lacour_CityNews Ottawa.json']
    all_articles = load_all_articles()
    # the train set will be all of these, unless they are about a person in {DEV_SUBJECTS}
    train_set = []
    train_repaired_articles = []
    for article in repaired_articles + perfect_ratio_articles:
        if not any(dev_subject in article for dev_subject in DEV_SUBJECTS):
            train_set.append(
                article.replace('_repaired', '')
            )
            if '_repaired' in article:
                train_repaired_articles.append(article.replace('_repaired', ''))
    print(f"{len(train_set)} Train set articles: {train_set}\n=========")
    # development articles
    development_articles = []
    for article in all_articles:
        for dev_subject in DEV_SUBJECTS:
            if dev_subject in article:
                development_articles.append(article)
                break
    # select 25 - len(development_articles) articles to also add into the development set. 
    # Then, the rest will be the test set.
    remaining_articles = set(all_articles) - set(train_set) - set(development_articles)
    num_additional_dev = 25 - len(development_articles)
    additional_dev_articles = random.sample(remaining_articles, num_additional_dev)
    development_articles.extend(additional_dev_articles)
    print(f"{len(development_articles)} Development set articles: {development_articles}\n=======")

    test_set = set(all_articles) - set(development_articles) - set(train_set)
    print(f"{len(test_set)} Test set articles: {test_set}")

    # compute the total number of paragraphs in each set
    for set_name, article_list in [('train', train_set), ('dev', development_articles), ('test', list(test_set))]:
        total_paragraphs = 0
        for article in article_list:
            paragraphs = load_article_paragraphs(article)
            total_paragraphs += len(paragraphs)
        print(f"{set_name} set has {total_paragraphs} paragraphs across {len(article_list)} articles.")
    ipdb.set_trace()
    return {
        'train': train_set,
        'dev': development_articles,
        'test': list(test_set),
        'repaired (train)': repaired_articles
    }

@click.command()
def check_agreement():
    with open("data/02_annotations.json", 'r') as f:
        articles = list(json.load(f).keys())
    print(articles)
    all_gt_annotations = []
    all_indep_annotations = []
    for article in articles:
        person_name = article.split('_')[1]
        outlet = article.split('_')[2].replace('.json', '')
        identifier = int(article.split('_')[0])
        gt_annotations = load_training_data_annotations_for_person(person_name=person_name,
                                                                outlet=outlet, 
                                                                identifier=identifier
                                                                )
        indep_annotations = load_training_data_annotations_for_person(person_name=person_name,
                                                                    outlet=outlet, 
                                                                    identifier=identifier,
                                                                    fname="data/02_annotations.json"
                                                                    )                                                

        paragraphs = load_article_paragraphs(article)                                                            
        gt_paragraph_to_affinities = compute_paragraph_to_affinities(gt_annotations)
        indep_paragraph_to_affinities = compute_paragraph_to_affinities(indep_annotations)
    # load paragraphs for this article
        y_one, y_two = convert_to_ternary_label_list(gt_paragraph_to_affinities, indep_paragraph_to_affinities, len(paragraphs))
        all_gt_annotations.extend(y_one)
        all_indep_annotations.extend(y_two)
    score = cohen_kappa_score(all_gt_annotations, all_indep_annotations, labels=['police-aligned', 'victim-aligned', 'no entity'])
    # Compute the randomized agreement, by shufflign all_indep_annotations
    shuffled_gt_annotations = all_gt_annotations.copy()
    shuffled_indep_annotations = all_indep_annotations.copy()
    random_scores = []
    for i in range(1000):
        random.shuffle(shuffled_indep_annotations)
        random.shuffle(shuffled_gt_annotations)
        random_score = cohen_kappa_score(shuffled_gt_annotations, shuffled_indep_annotations, labels=['police-aligned', 'victim-aligned', 'no entity'])
        random_scores.append(random_score)
    avg_random_score = sum(random_scores) / len(random_scores)
    print(f"Average randomized Cohen's kappa score: {avg_random_score}")


    logger.info(f"Cohen's kappa score: {score}")
    print(len(all_gt_annotations))
    # compute the f1 score for police-aligned vs not police-aligned
    report = classification_report(all_indep_annotations, all_gt_annotations, labels=['police-aligned', 'victim-aligned', 'no entity'])
    print(report)





    # are there any cases where all_gt_annotations[i] is 'police' and all_indep_annotations[i] is 'victim' or vice versa?
    # polar_disagreements = 0
    # for i in range(len(all_gt_annotations)):
    #     if (all_gt_annotations[i] == 'police-aligned' and all_indep_annotations[i] == 'victim-aligned') or \
    #        (all_gt_annotations[i] == 'victim-aligned' and all_indep_annotations[i] == 'police-aligned'):
    #         logger.warning(f"Disagreement at index {i}: GT={all_gt_annotations[i]}, Indep={all_indep_annotations[i]}")
    #         polar_disagreements += 1
    # logger.info(f"{polar_disagreements}/{len(all_gt_annotations)} polar disagreements found.")

@click.command()
@click.argument("input_file_prefix", type=str)
def analyze_large_scale_inference(input_file_prefix):
    inference_dataset = load_dataset("json", 
                           data_files={'train': f"data/distillation_data/{input_file_prefix}_coref_inference_with_predictions.json"},
                           split='train')
    inference_dataset = pl.from_pandas(inference_dataset.to_pandas())
    # TODO: need to write a new function
    indices = set(inference_dataset['article_index'])
    failed_indices = []

    # create a folder f"data/{input_file_prefix}_inference_predictions" if it doesn't exist
    os.makedirs(f"data/{input_file_prefix}_inference_predictions", exist_ok=True)
    for index in tqdm(indices):
        article_subset = inference_dataset.filter(pl.col('article_index') == index)
        article = f"{index}_{article_subset['victim_name'][0]}_{article_subset['outlet'][0]}"
        try:
            y_pred = get_predicted_article_paragraph_mappings(article_subset, article, input_file_prefix)
            article_paragraphs = load_unsupervised_article_paragraphs(article, input_file_prefix + "_articles")
            assert len(y_pred) == len(article_paragraphs), f"Length mismatch for {article}: {len(y_pred)} vs {len(article_paragraphs)}"
            # write the predictions to "data/unsupervised_inference_predictions/{article}.json"
            with open(f"data/{input_file_prefix}_inference_predictions/{article}", 'w') as f:
                json.dump(y_pred, f)
        except ValueError as e:
            logger.error(f"Value error for article {article}: {e}")
            failed_indices.append(index)
    logger.info(f"Failed indices: {failed_indices}")

main.add_command(distill_flant5)
# main.add_command(create_training_dataset_rolling)
main.add_command(compute_required_memory)
main.add_command(assess_baseline_ner_model)
# main.add_command(create_training_dataset_rolling)
main.add_command(assess_ft_flan_model)
main.add_command(create_coref_training_dataset)
main.add_command(check_agreement)
main.add_command(run_large_scale_inference)
main.add_command(analyze_large_scale_inference) # TODO: create a new file for every article, lining up with the large dataset. Should basically be able to do a .join with it.
# main.add_command(obtain_train_eval_test_split)
# main.add_command(create_distillation_examples_task1)

if __name__ == "__main__":
# Example: generate 5 random configs
    main()
    # trials = random_hyperparams(n_trials=5, seed=42)
    # # learning_rates = " ".join([f'{trial['learning_rate']:.6f}' for trial in trials])
    # learning_rates = " ".join([f"{trial['learning_rate']:.6f}" for trial in trials])
    # training_steps = " ".join([str(trial['training_steps']) for trial in trials])
    # warmup_steps = " ".join([str(trial['warmup_steps']) for trial in trials])
    # weight_decays = " ".join([str(trial['weight_decay']) for trial in trials])
    # print(f"LRS=({learning_rates})")
    # print(f"TRAINING_STEPS=({training_steps})")
    # print(f"WARMUP_STEPS=({warmup_steps})")
    # print(f"WEIGHT_DECAYS=({weight_decays})")
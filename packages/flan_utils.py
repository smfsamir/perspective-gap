import loguru
import ipdb
from typing import Dict, List
import jiwer
from collections import Counter
from sklearn.metrics import classification_report

# reference = ["i can spell", "i hope"]
# hypothesis = ["i kan cpell", "i hop"]

# error = jiwer.cer(reference, hypothesis)

logger = loguru.logger
def generate_predictions(model, tokenizer, batch) -> Dict:
    inputs = tokenizer(batch['prompt'], return_tensors='pt', padding=True).to('cuda')
    outputs = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=300
    )
    batch['predicted_text'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch

def generate_predictions_tokenized_batch(model, tokenizer, batch) -> Dict:
    # the batch only has [input_ids, labels, and attention_mask]
    prediction_logits = model.generate(
                    input_ids=batch['input_ids'].to(model.device), 
                    attention_mask=batch['attention_mask'].to(model.device), 
                    max_new_tokens=300
                )  
    batch['predicted_text'] = tokenizer.batch_decode(prediction_logits, skip_special_tokens=True)
    batch['label_text'] = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    return batch

def compute_metrics_tokenized_batch(batch) -> Dict:
    no_entity_str = "there is no valid entity providing a perspective here."
    entity_present_str = "the entity name is"
    entity_correct_batch = []
    batch_cers = []
    for i, example in enumerate(batch['label_text']):
        ground_truth = example.lower()
        prediction = batch['predicted_text'][i].lower()
        if entity_present_str in ground_truth:
            batch['valid_entity'] = 1
            if entity_present_str in prediction:
                is_correct = 1
            else:
                is_correct = 0
        elif no_entity_str in ground_truth:
            if no_entity_str in prediction:
                is_correct = 1
            else:
                is_correct = 0
        else:
            logger.warning(f"Ground truth not in expected format: {ground_truth}")
            raise ValueError(f"Ground truth not in expected format: {ground_truth}")
        entity_correct_batch.append(is_correct)
        batch_cers.append(jiwer.cer(ground_truth, prediction))
    return {
        'entity_present_correct': entity_correct_batch, 
        'cer': batch_cers
    }

def generate_singleton_prediction(model, tokenizer, example) -> Dict:
    inputs = tokenizer(example['prompt'], return_tensors='pt').to('cuda')
    outputs = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=300
    )
    ipdb.set_trace()
    example['predicted_text'] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return example


def evaluate_entity_identified_single(example) -> Dict: # not batched
    no_entity_str = "there is no valid entity providing a perspective here."
    entity_present_str = "the entity name is"
    ground_truth = example['completion'].lower()
    prediction = example['predicted_text'].lower()
    if entity_present_str in ground_truth:
        example['valid_entity'] = 1
        if entity_present_str in prediction:
            is_correct = 1
        else:
            is_correct = 0
    elif no_entity_str in ground_truth:
        if no_entity_str in prediction:
            is_correct = 1
        else:
            is_correct = 0
    else:
        logger.warning(f"Ground truth not in expected format: {ground_truth}")
        raise ValueError(f"Ground truth not in expected format: {ground_truth}")
    return is_correct

def convert_text_to_entity_present_label(description_texts):
    no_entity_str = "there is no valid entity providing a perspective here"
    entity_present_str = "the entity name is"
    labels = []
    for text in description_texts:
        if no_entity_str in text.lower():
            labels.append('no entity')
        elif entity_present_str in text.lower():
            labels.append('valid entity')
        else:
            labels.append('unrecognized')
    return labels

def is_valid_entity_present(description_text):
    no_entity_str = "there is no valid entity providing a perspective here"
    entity_present_str = "the entity name is"
    if no_entity_str in description_text.lower():
        return 'no entity'
    elif entity_present_str in description_text.lower():
        return 'valid entity'

def is_police_aligned_entity(description_text):
    assert is_valid_entity_present(description_text) == 'valid entity', "Entity is not valid"
    if 'not aligned with the police' in description_text.lower():
        return False
    else:
        return True

def extract_relevant_paragraphs(description_text):
    assert is_valid_entity_present(description_text) == 'valid entity', f"Entity is not valid: {description_text}"
    # remove the period, hence the [:-1]
    paragraph_numbers = description_text.split("The paragraphs that reflect their perspectives are:")[-1].strip()[:-1]\
        .split(",")
    if 'none' in paragraph_numbers[0]:
        logger.warning('valid entity but no paragraphs?')
        return []
    else:
        return [int(num.strip()) for num in paragraph_numbers]

def reduce_affinities_to_individual_prediction(
        predicted_paragraph_to_affinity) -> Dict[int, List[str]]:
    paragraph_to_affinities = {}
    for paragraph_index, affinities in predicted_paragraph_to_affinity.items():
        # count the number of each affinity, and assign the most common one
        affinity_counts = Counter(affinities)
        most_common_affinity = affinity_counts.most_common(1)[0][0]
        paragraph_to_affinities[paragraph_index] = most_common_affinity
    return paragraph_to_affinities


def convert_to_ternary_label_list(gt_paragraph_to_affinity, 
               predicted_paragraph_to_affinity, 
               num_paragraphs_in_article):
    gt_paragraph_to_affinity = gt_paragraph_to_affinity.copy()
    predicted_paragraph_to_affinity = predicted_paragraph_to_affinity.copy()
    # add any missing paragraphs as 'no entity', to both dictionaries.
    for i in range(1, num_paragraphs_in_article + 1):
        if i not in gt_paragraph_to_affinity:
            gt_paragraph_to_affinity[i] = 'no entity'
        if i not in predicted_paragraph_to_affinity:
            predicted_paragraph_to_affinity[i] = 'no entity'
    assert len(gt_paragraph_to_affinity) == len(predicted_paragraph_to_affinity)
    y_true = []
    y_pred = []
    for i in range(1, num_paragraphs_in_article + 1):
        y_true.append(gt_paragraph_to_affinity[i])
        y_pred.append(predicted_paragraph_to_affinity[i])
    return y_true, y_pred

def convert_to_ternary_label_list_inference( 
               predicted_paragraph_to_affinity, 
               num_paragraphs_in_article):
    predicted_paragraph_to_affinity = predicted_paragraph_to_affinity.copy()
    # add any missing paragraphs as 'no entity', to both dictionaries.
    for i in range(1, num_paragraphs_in_article + 1):
        if i not in predicted_paragraph_to_affinity:
            predicted_paragraph_to_affinity[i] = 'no entity'
    y_pred = []
    for i in range(1, num_paragraphs_in_article + 1):
        y_pred.append(predicted_paragraph_to_affinity[i])
    return y_pred
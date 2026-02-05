#!/bin/bash
ARTICLE_FOLDER_PREFIX=$1

echo "Running perspective gap analysis on articles in the ${ARTICLE_FOLDER_PREFIX}_articles folder."

python main_unsupervised.py compute-fastcoref-annotations new_articles $ARTICLE_FOLDER_PREFIX
python main_unsupervised.py compute-article-basis-coref-objects $ARTICLE_FOLDER_PREFIX 
python main_unsupervised.py create-coref-inference-dataset $ARTICLE_FOLDER_PREFIX
python main_distillation.py run-large-scale-inference $ARTICLE_FOLDER_PREFIX
python main_distillation.py analyze-large-scale-inference $ARTICLE_FOLDER_PREFIX
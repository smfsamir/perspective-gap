#!/bin/bash
ARTICLE_FOLDER_PREFIX=$1

# mkdir data if it doesn't exist
mkdir -p data

# check that an argument was provided

if [ -z "$ARTICLE_FOLDER_PREFIX" ]; then
  echo "Usage: $0 <article_folder_prefix>"
  exit 1
fi
echo "Running perspective gap analysis on articles in the ${ARTICLE_FOLDER_PREFIX}_articles folder."

python main_unsupervised.py compute-fastcoref-annotations new_articles $ARTICLE_FOLDER_PREFIX
python main_unsupervised.py compute-article-basis-coref-objects $ARTICLE_FOLDER_PREFIX 
python main_unsupervised.py create-coref-inference-dataset $ARTICLE_FOLDER_PREFIX
python main_distillation.py run-large-scale-inference $ARTICLE_FOLDER_PREFIX
python main_distillation.py analyze-large-scale-inference $ARTICLE_FOLDER_PREFIX

# log the filenames and contents of the files in data/
echo "Files in data/ after running perspective gap analysis:"
ls -l data/
for file in data/*; do
  echo "Contents of $file:"
  cat "$file"
  echo "-----------------------------------"
done
#!/bin/bash
ARTICLE_FOLDER_PREFIX=$1

# mkdir data if it doesn't exist
mkdir -p data

# Check that an argument was provided
if [ -z "$ARTICLE_FOLDER_PREFIX" ]; then
  echo "Error: No article folder prefix provided"
  echo "Usage: $0 <article_folder_prefix>"
  exit 1
fi

echo "=================================================="
echo "Perspective Gap Analysis Pipeline"
echo "Article folder: ${ARTICLE_FOLDER_PREFIX}_articles"
echo "=================================================="
echo ""

# Function to run a command and only show output on error
run_step() {
  local step_name="$1"
  shift
  local cmd=("$@")
  
  echo "→ ${step_name}..."
  
  # Run command and capture output
  if output=$("${cmd[@]}" 2>&1); then
    echo "  ✓ Complete"
    return 0
  else
    echo "  ✗ Failed"
    echo ""
    echo "Error output:"
    echo "$output"
    return 1
  fi
}

# Step 1: Coreference Resolution
echo "[Step 1/3] Running coreference resolution"
run_step "Computing FastCoref annotations" \
  python main_unsupervised.py compute-fastcoref-annotations new_articles "$ARTICLE_FOLDER_PREFIX" || exit 1

run_step "Computing article basis coref objects" \
  python main_unsupervised.py compute-article-basis-coref-objects "$ARTICLE_FOLDER_PREFIX" || exit 1

run_step "Creating coref inference dataset" \
  python main_unsupervised.py create-coref-inference-dataset "$ARTICLE_FOLDER_PREFIX" || exit 1

echo ""

# Step 2: Large-scale Inference
echo "[Step 2/3] Running large-scale inference"
run_step "Executing inference" \
  python main_distillation.py run-large-scale-inference "$ARTICLE_FOLDER_PREFIX" || exit 1

echo ""

# Step 3: Analysis
echo "[Step 3/3] Parsing and analyzing inference results"
run_step "Analyzing results" \
  python main_distillation.py analyze-large-scale-inference "$ARTICLE_FOLDER_PREFIX" || exit 1

echo ""
echo "=================================================="
echo "Analysis Complete!"
echo "=================================================="
echo ""

# Log the filenames and contents of the files in the inference predictions folder
echo "Files in data/${ARTICLE_FOLDER_PREFIX}_inference_predictions/ after running perspective gap analysis:"
ls -l "data/${ARTICLE_FOLDER_PREFIX}_inference_predictions/"
echo ""
for file in data/${ARTICLE_FOLDER_PREFIX}_inference_predictions/*; do
  echo "Contents of $file:"
  cat "$file"
  echo "-----------------------------------"
done
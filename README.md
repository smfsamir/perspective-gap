# Installation
```
python3.12 -m venv perspective_gap_venv
pip install -r requirements.txt
```

# Instructions


You can run the `perspective-gap` model on news articles on deadly force events by executing the `./run_perspective_gap.sh` script. It takes one command-line argument, for the name of the directory containing the news articles. The directory should be named as  `{PREFIX}_articles`, and you supply `PREFIX` to the script. For example, you could create a directory called `news_articles`, place your articles in there, and call `./run_perspective_gap.sh news`.  

## Formatting of articles
The articles in `{PREFIX}_articles` should be of the format `{index}_{victim_name}_{outlet}.json`, for example, `1_Chris Amyotte_The Tyee.json'. The format of the file should be:
```
[
  "The family of a man who died after....",
  "Moments after being hit in the back...",
]
```
There should be one **paragraph** per line, as this is how the model was trained. The model is available here: `https://huggingface.co/smfsamir/perspective-gap`

## Perspective-gap output
The output will be printed by the script, using the artifacts in `data/{PREFIX}_inference_predictions/`. There will be a file with the same name as the news articles, e.g., `data/{PREFIX}_inference_predictions/1_Chris Amyotte_The Tyee.json`. The file will contain, for example, 

```
["victim-aligned", "no entity", "victim-aligned", "no entity", "no entity", "police-aligned", "police-aligned", "police-aligned", "no entity", "no entity", "no entity", "victim-aligned", "victim-aligned", "victim-aligned", "police-aligned", "police-aligned", "police-aligned", "no entity", "no entity", "victim-aligned", "police-aligned", "police-aligned", "victim-aligned", "victim-aligned", "police-aligned", "police-aligned", "no entity", "no entity", "police-aligned", "no entity", "police-aligned", "victim-aligned", "victim-aligned", "police-aligned", "police-aligned", "police-aligned", "no entity", "police-aligned", "police-aligned", "police-aligned", "victim-aligned", "victim-aligned", "victim-aligned", "victim-aligned", "victim-aligned", "no entity"]
```
There is one label per paragraph in the article, so the number of elements in this list should be the same as the number of lines in the article JSON file. 




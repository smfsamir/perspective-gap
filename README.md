# Instructions

You can run the `perspective-gap` model on news articles on deadly force events by executing the `./run_perspective_gap.sh` script. It takes one command-line argument, for the name of the directory containing the news articles. The directory should be named as  `{PREFIX}_articles`, and you supply `PREFIX` to the script. For example, you could create a directory called `news_articles`, place your articles in there, and call `./run_perspective_gap.sh news`.  

## Formatting of articles
The articles should be of the format `{index}_{victim_name}_{outlet}`, for example, `1_Chris Amyotte_The Tyee.json'. The format of the file should be:
```
[
  "The family of a man who died after....",
  "Moments after being hit in the back...",
]
```

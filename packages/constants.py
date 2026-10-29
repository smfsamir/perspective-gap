import textwrap

TUNING_INDICES = set([6, 7, 10, 13, 15, 17, 22, 24, 27, 28, 29, 33, 36, 38, 40, 42, 44, 46, 47, 48,
50, 51, 54, 55, 58, 61, 63, 66, 67, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82,
84, 85, 86, 87, 89, 90, 92, 95, 98, 99])

OLD_TO_NEW_LABEL_MAP = {'police-aligned': 'state-aligned', 'victim-aligned': 'civilian-aligned', 'no entity': 'no entity'}
LABELS = ['state-aligned', 'civilian-aligned', 'no entity']

QWEN_INSTRUCTIONS = textwrap.dedent("""
Each numbered line above is a paragraph from a news article about a death involving police.
For every line that expresses the perspective of a specific actor, say whose perspective it is
and whether that actor is state-aligned or civilian-aligned.

What counts as a perspective:
- A quote, indirect speech, or paraphrase of an actor's statement, belief, perception, judgement,
decision, intention or emotion.
- Statements of not knowing or not disclosing ("Police could not say whether...",
"X declined to comment") count as that actor's perspective.
- Institutional decisions count ("The SIU has designated the officer as the subject of its probe").
- Attribution can carry over: a line with no attribution of its own continues the previous
speaker's perspective until a new source is introduced or the text returns to neutral narration.
For example, a paragraph that is only a quote, right after a paragraph quoting the victim's
sister, is the sister's perspective.
- Statements from a report produced by a state institution is also considered a state-aligned perspective.
- NOT a perspective: neutral narration of events, and things the news outlet itself reports
("Postmedia has learned...").

State-aligned: government and policing actors, whether organizations or the individuals working
for them, including police services, officers, police chiefs, police unions, mayors and other
officials, and police oversight bodies (e.g. SIU, IIO, ASIRT, BEI). They are state-aligned even
when their statements sound sympathetic.

Civilian-aligned: the victim, and the victim's family, friends, lawyer, and community members
who knew them; civil rights or advocacy organizations. Include family members even if they are
not named (e.g. "his sister").

Bystanders and witnesses: decide from what they say. If they side more with the police, they are
state-aligned; if they side more with the victim, they are civilian-aligned.

Output a JSON object containing ONLY the lines that express a perspective. Every line you leave
out is treated as having no perspective. If a line shows more than one perspective, use the label and entity that is more prominent. 
Use the line number as the key, for example:
{"3": "state-aligned", "7": "label": "civilian-aligned"}
Output only the JSON object.
""").strip()
TASK_1_PROMPT = """
### Instruction:
Read the provided article on an incident of police violence to complete the following task. Task: Identify each entity mentioned in the article and determine their role and affiliation. The entity may be a person or an instituion, like the ACLU or the Special Investigations Unit (SIU). Only include entities that make a statement on the case.

Affiliation: An entity can either be on the side of the police, or the civilian victim.

Police-aligned entities:
- Government and policing entities (both organizations and individuals working for them) are always police-aligned. Even if their statements are ostensibly sympathetic, they should always be considered police-aligned. This includes police chiefs, mayors, police oversight bodies, police unions, etc. 
- For individual police officers, only include them if they are mentioned by name. Do not include generic references to a police officer (e.g., "the police officer", "the officer", "the cop", etc.).
- Attempt to identify the specific police department, if it is explicity named. Otherwise use the generic term "police"

Victim-aligned entities:
- Entities who side with the civilian victim, on the other hand, tend to be, of course, the victim's family or friends, their attorney, community members who knew the victim, witnesses, civil rights organization, and more.
- If a person is clearly identified as a family member or friend (e.g., “my dad,” “his sister,” “the victim’s mother”), but their name is not mentioned, you should still include them.

Output format:
{
"Victim-aligned": ["entity1 (role)", "entity2 (role)", "entity3 (role)"],
"Police-aligned": ["entity4 (role)", "entity5 (role)"],
}

Here is the article:
"""


# Assume we have 100 cleanly annotated articles. 

# Model distillation: 
# Take 80 of the articles and their labels and train our own model.
# This model we can then apply on thousands of articles.

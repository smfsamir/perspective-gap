TASK_1_PROMPT = """
Read the provided article on an incident of police violence to complete the following task.

Task: Identify each entity mentioned in the article and determine their role and affiliation. The entity may be a person or an instituion, like the ACLU or the Special Investigations Unit (SIU). Only include entities that make a statement on the case.

Restrictions on who/what to identify:
- Only entities that are identified by either a name (e.g.,idenitfy "Matt Dumas", "Matt", "American Civil Liberties Union" "ACLU", and do not identify "man", "an organization") or relationship should be included (e.g. "victim's brother", "victim's neighbour", "officer's supervisor", etc.)
- Do not include media outlets as entities
- Avoid including entities which are only mentioned generically (e.g. "the officer", "the man") unless it is absolutely necessary to understand the article. If the generic, unnamed entity is the victim mark their name as "victim". 
- Only include civilians if they 1. are mentioned by name or relationship and 2. have been quoted or directly paraphrased in the article
- A person may represent or be a spokeperson for an organization. Identify both the person and the organization as separate entities.

How to label entities:
- Avoid abbreviations when the full name is mentioned (say "American Civil Liberties Union" and not ACLU)
- The same entity could be mentioned many times by different referring expressions. For instance, the terms "he", "Bob Dylan", "the singer", and "Dylan" all refer to the same person. 
Use the article context to recognize these mentions as co-references and list the entity only once using their most full name or most descriptive name (e.g. "Bob Dylan"). 
Do not create separate entities for pronouns, mentions, and names (e.g. "he", "the singer", "Dylan", etc). 
- Every entity has a role. Following each entity's name, indicate their role in parentheses, such as (police chief), (witness), (mayor), (victim's mother), (legal nonprofit), (police oversight body) (victim's attorney), (police officer), (police union), etc. 
    - For example, if the article mentions "mayor john", "the victim's mother jane" and "police chief bob", you would return "john (mayor)", "jane (victim's mother)", "bob (police chief)"
    - Never include the role in the name, e.g. tag "const. jerry" as "jerry (const.)" and not "const. jerry (const.)" or "const. jerry"
- If the person is unnamed but is described by a relationship, include the full name of the entity with whom they have a relationship. For example, if an article about Bob Dylan mentions "dylan's brother", tag this person as "bob dylan's brother". 

Affiliation: An entity can either be on the side of the police, or the civilian victim.

Police-aligned entities:
- Government and policing entities (both organizations and individuals working for them) are always police-aligned. Even if their statements are ostensibly sympathetic, they should always be considered police-aligned. This includes police chiefs, mayors, police oversight bodies, police unions, etc. 
- For individual police officers, only include them if they are mentioned by name. Do not include generic references to a police officer (e.g., "the police officer", "the officer", "the cop", etc.).
- Attempt to identify the specific police department, if it is explicity named. Otherwise use the generic term "police"
- There may also be bystanders who take the side of the police. For these bystanders, it  is important to take their quotes into account to determine their affiliation. If their opinion sympathizes with the police more than the victim, assign them police-aligned. 

Victim-aligned entities:
- Entities who side with the civilian victim, on the other hand, tend to be, of course, the victim's family or friends, their attorney, community members who knew the victim, witnesses, civil rights organization, and more.
- If a person is clearly identified as a family member or friend (e.g., “my dad,” “his sister,” “the victim’s mother”), but their name is not mentioned, you should still include them.
- Bystanders may take the side of the victim.  For these bystanders, it  is important to take their quotes into account to determine their affiliation. If their opinion sympathizes with the victim more than the police, assign them victim-aligned. 

People mentioned from previous cases (whether officers or victims) should also be recorded, even if they are not part of the current case itself.

Output format:
- Use all lowercase for the names and roles
- Use the following format exactly: 
{
"Victim-aligned": ["entity1 (role)", "entity2 (role)", "entity3 (role)"],
"Police-aligned": ["entity4 (role)", "entity5 (role)"],
}

Here is the article:
"""
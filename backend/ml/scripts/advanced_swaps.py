import spacy

"""
This module demonstrates an ADVANCED approach to demographic swapping using spaCy.
It uses Named Entity Recognition (NER) to detect PERSON entities and a token-level
approach to handle pronouns.

You can import and call the `swap_demographics_spacy` function in data_pipeline.py
or anywhere else in your code.

INSTALLATION:
    pip install spacy
    python -m spacy download en_core_web_sm
"""


nlp = spacy.load("en_core_web_sm")

# Example dictionary of male-to-female name swaps
NAME_SWAPS_MALE_FEMALE = {
    "John": "Maria",
    "Michael": "Jessica",
    "Robert": "Laura",
    "James": "Sarah",
    "David": "Emily",
    # etc. Expand as needed.
}

# Example dictionary for pronoun swaps (male->female)
PRONOUN_SWAPS_M2F = {
    "he": "she",
    "He": "She",
    "him": "her",
    "Him": "Her",
    "his": "her",
    "His": "Her",
    "father": "mother",
    "Father": "Mother",
    # etc.
}

# Example dictionary for last names or any other relevant words
# Use as needed.
LASTNAME_SWAPS = {
    "Smith": "Rodriguez",
    "Miller": "Johnson",
}


def swap_demographics_spacy(
    text: str,
    name_swaps: dict = None,
    pronoun_swaps: dict = None,
    last_name_swaps: dict = None,
) -> str:
    """
    Use spaCy to parse text, detect PERSON entities, and then build a new text.
    - name_swaps: dictionary for swapping first names
    - pronoun_swaps: dictionary for swapping pronouns (e.g., male->female)
    - last_name_swaps: dictionary for swapping last names

    Returns: A new string with swapped entities & tokens.
    """
    if not name_swaps:
        name_swaps = {}
    if not pronoun_swaps:
        pronoun_swaps = {}
    if not last_name_swaps:
        last_name_swaps = {}

    doc = nlp(text)

    # We'll build a list of tokens we can modify
    new_tokens = []

    # We'll keep track of entity boundaries to avoid duplicating replacements
    entity_indexes = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            for i in range(ent.start, ent.end):
                entity_indexes.add(i)

    for i, token in enumerate(doc):
        # If this token is part of a PERSON entity, handle name swaps
        if i in entity_indexes:
            # Check if the token text is in name_swaps or last_name_swaps
            # We do a .strip() or .lower() comparison if needed
            base_form = token.text
            if base_form in name_swaps:
                new_tokens.append(name_swaps[base_form])
            elif base_form in last_name_swaps:
                new_tokens.append(last_name_swaps[base_form])
            else:
                # If no direct match, leave it as is
                new_tokens.append(base_form)
        else:
            # Not in a PERSON entity, handle pronoun swaps or else keep token
            if token.text in pronoun_swaps:
                new_tokens.append(pronoun_swaps[token.text])
            else:
                new_tokens.append(token.text)

    # Reconstruct sentence
    return " ".join(new_tokens)


def demo_swap():
    text = "John Smith said he would help his father, Michael Miller, soon."  # example
    swapped_text = swap_demographics_spacy(
        text,
        name_swaps=NAME_SWAPS_MALE_FEMALE,
        pronoun_swaps=PRONOUN_SWAPS_M2F,
        last_name_swaps=LASTNAME_SWAPS
    )

    print("Original:", text)
    print("Swapped :", swapped_text)

if __name__ == "__main__":
    demo_swap()

import spacy

# Try to load the spaCy English model. If not found, prompt the user to install it.
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

############################################
#      Demographic Swap Dictionaries       #
############################################

# Mapping for male-to-female name swaps.
NAME_SWAPS_MALE_FEMALE = {
    "John": "Maria",
    "Michael": "Jessica",
    "Robert": "Laura",
    "James": "Sarah",
    "David": "Emily",
}

# Mapping for male-to-female pronoun swaps.
PRONOUN_SWAPS_M2F = {
    "he": "she", "He": "She",
    "him": "her", "Him": "Her",
    "his": "her", "His": "Her",
    "father": "mother", "Father": "Mother",
}

# Mapping for last name swaps.
LASTNAME_SWAPS = {
    "Smith": "Rodriguez",
    "Miller": "Johnson",
}

############################################
#      Function: swap_demographics_spacy   #
############################################

def swap_demographics_spacy(
    text: str,
    name_swaps: dict = None,
    pronoun_swaps: dict = None,
    last_name_swaps: dict = None,
) -> str:
    """
    Swap demographic information in the text using spaCy NER.

    This function uses spaCy's named entity recognition (NER) to find PERSON entities
    within the text. For tokens that are part of a PERSON entity, it attempts to replace
    them using provided name or last name swap dictionaries. For tokens outside of a PERSON
    entity, it checks for pronoun swaps.

    Parameters:
        text (str): The original text to modify.
        name_swaps (dict, optional): Replacement mapping for first names.
                                     Defaults to an empty dict if not provided.
        pronoun_swaps (dict, optional): Replacement mapping for pronouns.
                                        Defaults to an empty dict if not provided.
        last_name_swaps (dict, optional): Replacement mapping for last names.
                                          Defaults to an empty dict if not provided.

    Returns:
        str: The modified text after applying the demographic swaps.
    """
    # Initialize swap dictionaries if they are not provided.
    if not name_swaps:
        name_swaps = {}
    if not pronoun_swaps:
        pronoun_swaps = {}
    if not last_name_swaps:
        last_name_swaps = {}

    # Process the input text using spaCy.
    doc = NLP(text)
    new_tokens = []  # List to hold the new token texts.
    entity_indexes = set()  # Set to hold indices of tokens that are part of PERSON entities.

    # Identify tokens belonging to PERSON entities.
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            for i in range(ent.start, ent.end):
                entity_indexes.add(i)

    # Iterate over each token to apply the swaps.
    for i, token in enumerate(doc):
        # If the token is part of a PERSON entity, attempt to swap using first name or last name mappings.
        if i in entity_indexes:
            word = token.text
            if word in name_swaps:
                new_tokens.append(name_swaps[word])
            elif word in last_name_swaps:
                new_tokens.append(last_name_swaps[word])
            else:
                new_tokens.append(word)
        else:
            # Outside PERSON entities, check for pronoun swaps.
            if token.text in pronoun_swaps:
                new_tokens.append(pronoun_swaps[token.text])
            else:
                new_tokens.append(token.text)

    # Reconstruct the text, preserving the original whitespace.
    reconstructed_text = "".join([t + doc[i].whitespace_ for i, t in enumerate(new_tokens)])
    return reconstructed_text

############################################
#             Demo Function                #
############################################

def demo_swap():
    """
    Demonstrates the demographic swapping by applying the swaps
    to an example sentence and printing the before and after.
    """
    text = "John Smith said he would help his father, Michael Miller, soon."
    swapped = swap_demographics_spacy(
        text,
        name_swaps=NAME_SWAPS_MALE_FEMALE,
        pronoun_swaps=PRONOUN_SWAPS_M2F,
        last_name_swaps=LASTNAME_SWAPS,
    )
    print("Original:", text)
    print("Swapped :", swapped)

############################################
#               Main Section               #
############################################

if __name__ == "__main__":
    # Run the demo if the script is executed directly.
    demo_swap()

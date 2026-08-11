import sys
import os
import re # regular expressions

def parse(tokens, pos):
    """
    This helper function will parse the tokens into a tree, starting at index pos.
    It returns (tree, new_pos)
    A tree is either a plain string (a word) or a tuple (label, [children]).
    """
    if tokens[pos] != '(':
        return tokens[pos], pos + 1 # First check if it is just a word and return it if it is

    # otherwise we have a tree, so we need to parse it
    pos += 1 # skips the opening paren
    label = tokens[pos] # Grabs the first thing after the opening paren, which is the label
    pos += 1 # Moves to the next token after the label
    children = [] # Empty list to hold the children

    while tokens[pos] != ')': # Keeps reading the children until the end paren
        child, pos = parse(tokens,pos) # Recursively parse the child and update the position, trees are inherently recursive (I used this StackOverflow post to help me: https://stackoverflow.com/questions/44708743/parse-a-penn-syntax-tree-to-extract-its-grammar-rules)
        children.append(child) # Add the child to the list of children

    pos += 1 # skips the end paren

    return (label, children), pos # Return the parsed tree and the new position


def count(tree, counts):
    """
    This function will count the number of times each label appears in the tree
    It recursively goes over the tree and updates the dictionary with the counts of each label
    """

    if isinstance(tree, str): # If the tree is a string (a word), we don't need to do anything
        return

    label, children = tree # Split the tree into its parts

    if label == 'S': # If the label is 'S', we add onto the Sentences count tracker
        counts['Sentences'] += 1

    if label == 'NP': # If the label is 'NP', we add onto the Noun Phrases count tracker
        counts['Noun Phrases'] += 1

    if label == 'VP': # If the label is 'VP', we add onto the Verb Phrases count tracker
        counts['Verb Phrases'] += 1

        np_count = sum(1 for c in children if isinstance(c, tuple) and c[0] == 'NP') # Count the number of NP children in the VP

        if np_count == 0:
            counts['Intransitive Verb Phrases'] += 1 # If there are no NP children, we add onto the Intransitive Verb Phrases count tracker

        elif np_count == 2:
            counts['Ditransitive Verb Phrases'] += 1 # If there are exactly two NP children, we add onto the Ditransitive Verb Phrases count tracker

    for child in children: # Recursively call count on each child
        count(child, counts)


# ------ main program ---------

directory = sys.argv[1] # Get the directory from the command line arguments

counts = {
    'Sentences': 0,
    'Noun Phrases': 0,
    'Verb Phrases': 0,
    'Intransitive Verb Phrases': 0,
    'Ditransitive Verb Phrases': 0
} # Initialize counts at 0 for each category

for filename in sorted(os.listdir(directory)): # Go through each file in the given directory

    filepath = os.path.join(directory, filename) # Get the full path to the file

    if not os.path.isfile(filepath): # Skip if it is not a file
        continue
    with open(filepath) as f:
        text = f.read() # Read the file

    tokens = re.findall(r'\(|\)|[^\s()]+', text) # Use regex to find all parens and words (with help from StackOverflow)
    pos = 0 # Start at the beginning of the tokens
    while pos < len(tokens): # Go through tokens until we reach the end
        tree, pos = parse(tokens, pos) # Parse using our parse function, which will return the tree and the new position
        count(tree, counts) # Count the tree using our count function, which will update the counts dictionary

for name in counts: # Print out the counts for each category
    print(f"{name}: {counts[name]}")

import sys
import os
import re

def generate_unigram_model(path):
    """
    This function will generate a unigram model from the text files in the given directory
    It will return a dictionary with the counts of each word in the text files
    """

    counts = {} # Initialize an empty dictionary to hold the counts

    for filename in os.listdir(path): # Go through each file in the given directory

        filepath = os.path.join(path, filename) # Get the full path to the file

        if not os.path.isfile(filepath): # Skip if it is not a file
            continue

        with open(filepath) as f:
            text = f.read() # Read the file

        #1. Remove SGML tags
        text = re.sub(r'<[^>]*>', ' ', text) # Use regex to remove SGML tags (with help from StackOverflow: https://stackoverflow.com/questions/11229831/regular-expression-to-remove-html-tags-from-a-string)

        #2. Split by spaces
        tokens = text.split(' ') # Split the text by spaces to get the words (tokens because there might be punctuation)

        #3. Keep only tokens formed entirely of A-Z, a-z, and straight apostrophe
        for token in tokens:
            if re.fullmatch(r"[A-Za-z\x27]+", token): # Use regex to check if the token is formed entirely of A-Z, a-z, and straight apostrophe using the ASCII code to be explicit

                #4a. Take out words that start or end with the apostrophe being explicit about the ASCII character
                if token.startswith("\x27") or token.endswith("\x27"):
                    continue

                #4b. Convert to lowercase for all remaining tokens
                word = token.lower()

                # Add the word to the counts dictionary, incrementing its count
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1

    #5. OUTSIDE OF THE LOOP OF FILES so that we loop through all files before performing this operation, sort by descending frequency then alphabetically

    # first sort alphabetically
    sorted_words = sorted(counts.items())

    # Then sort by frequency in descneding order. The alphabetic sorting will remain through this step
    # Use a lambda function to sort by the second item in the tuple (the count) in reverse order
    sorted_words_final = sorted(sorted_words, key=lambda x: x[1], reverse=True)

    # Finally print the results in the proper way
    for word, count in sorted_words_final:
        print(f"{word}\t{count}")

if __name__ == "__main__":
    generate_unigram_model(sys.argv[1]) # Call the function with the directory from the command line arguments
    
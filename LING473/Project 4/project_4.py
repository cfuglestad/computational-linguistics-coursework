import sys
import os
import math

def load_language_model(filepath):
    """
    Reads one of the unigram language model files and returns a dictionary mapping words
    to its count and a total count of all words including <UNK>
    """

    word_counts = {}
    total_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() # strip whitespace from both ends of the line

            if not line:
                continue # skip empty lines

            parts = line.split() #split by whitespace (defaulted by split())

            word = parts[0] # word is the first part
            count = int(parts[-1]) # count is the last part, convert to int

            word_counts[word] = count # add to dictionary
            total_count += count # add to total count

    return word_counts, total_count

def load_all_models(model_directory):
    """
    Loads all 15 language models from the given directory and returns
    a dictionary with language codes being the keys and tuple of 
    (word_counts, total_count) as the values
    """

    models = {}

    for filename in os.listdir(model_directory):
        if filename.endswith('unigram-lm'): # only process files with the specified ending
            language_code = filename.split('.')[0] # get the language code from the filename

            filepath = os.path.join(model_directory, filename) # attach the filename to the directory path
            word_counts, total_count = load_language_model(filepath) # load the model
            models[language_code] = (word_counts, total_count) # add to the models dictionary

    return models

def compute_log_probability(text, word_counts, total_count):
    """
    Computes the log probabilty of a text given a language model
    We will sum up the log-probs across all words in the text
    """

    words = text.split() # split the text into words

    unk_count = word_counts.get('<UNK>', 0) # get the count of <UNK> from the model, default to 0 if not found

    log_prob_sum = 0.0 # initialize log probability sum

    for word in words:
        if word in word_counts:
            count = word_counts[word] # get the count of the word from the model
        else:
            count = unk_count # if the word is not found, use the <UNK> count

    log_prob_sum += math.log10(count) - math.log10(total_count) # compute log probability and add to sum

    return log_prob_sum

def classify_text(text, models):
    """
    Classifies a text fragment by computing its log-prob under
    each language model and returns the language with the highest score

    Also returns a dictionary with language codes as keys and log_probs as values
    """

    scores = {} # initialize scores dictionary

    for language_code, (word_counts, total_count) in models.items():
        score = compute_log_probability(text, word_counts, total_count) # compute log probability for the text
        scores[language_code] = score # add to scores dictionary

    best_language = max(scores, key=scores.get) # find the language with the highest score

    return scores, best_language

def main():
    """
    Main function which reads args, loads models, classifies each text sample
    and writes the output file
    """

    if len(sys.argv) != 4:
        print("Usage: python project_4.py <model_directory> <input_file> <output_file>")
        sys.exit(1)

    model_directory = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    models = load_all_models(model_directory) # load all language models

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            text = line.strip() # strip whitespace from both ends of the line
            if not text:
                continue # skip empty lines

            parts = line.split('\t', 1) # split the line by ONLY the first tab
            identifier = parts[0] # the first part is the identifier
            text = parts[1] if len(parts) > 1 else '' # the second part is the text, if it exists

            scores, best_language = classify_text(text, models) # classify the text

            outfile.write(f"{identifier}\t{text}") # write the identifier and text to the output file

            for lang_code in sorted(scores.keys()): # sort the language codes for consistent output
                outfile.write(f"{lang_code}\t{scores[lang_code]:.6f}\n") # write the language code and its score to the output file

            outfile.write(f"result\t{best_language}\n") # write the result line to the output file

            outfile.write("\n") # write a newline to separate entries

if __name__ == "__main__":
    main() # call the main function
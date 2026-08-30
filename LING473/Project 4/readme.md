# Project 4 - Naive Bayes Classifier
## Connor Fuglestad - LING473 - Summer Quarter 2026

### Project Description

This is a python program that reads a directory of files and classifies them into one of 15 languages the text is most likely written in by comparing how probable each word in the text is under each language's statistical model.

### Results

The results will be found in the "output" file in my tar ball. Use "vim output" to see it.

### Approach

1. Load each of the 15 language's unigram model files
2. Estimate word probabilities for all the words in that "language"
3. If a word is not in the model's vocabulary stubstitue in the `<UNK>` word
4. Score each language and predict the best language, aka the least negative score in the prediction
5. Finally, write to the output file

The hardest part of this one was honestly just making sure I was parsing the files with the correct thing (tabs, spaces) and making sure the log prob was correctly translated over with the math library. That took a quick google search.

### Limitations

- Each model contains only 1,500 words and this can cause two problems. First, that isn't exactly a huge N and second, any word outside of the set gets treated as `<UNK>` which could be discarding potentially useful signal.
- Naive Bayes has no word-order information. As language speakers, we know that word order, grammar, etc are also useful signal.
- Highly specialized text will struggle in this model because of the size of the models.
- Lastly, as we know from lecture, Naive Bayes assumes all input features are independent and I don't think that is fair in the case of languages. This is kind of discussed above regarding the useful signals we discard.

### Goals completed

I was able to complete all stated goals of the project. The program takes the necessary input as well as the output file, categorizes according to the Naive Bayes algorithm, predicts the best language, and returns the output file with some nice formatting.
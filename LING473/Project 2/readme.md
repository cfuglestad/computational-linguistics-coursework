# Project 2 - Unigram Language Model
## Connor Fuglestad - LING473 - Summer Quarter 2026

### Project Description

This is a python program that reads a corpus directory, cleans all the files in the directory of SGML tags, extracts valid words, and outputs a table with the frequency of each word (a unigram language model) sorted by descending count then alphabetically

### Results
The results are very large, so I am not including them in my readme file. The table below is supposed to be the output of running my program. In my tar ball, if you "vim output" you should see this table.

| Word                      | Count |
| ------------------------- | ----- |
| *run program to populate* |       |

### Approach

1. First read all the files in the given corpus path
2. Remove all SGML tags which consist of anything between (and including) the less-than and greater-than symbols
3. Split text by white spaces
4. Keep only the pieces of the split strings (called tokens) that are composed of only upper case letters, lower case letters, and the straight apostrophe (ASCII character given in assignment).
5. Filter out tokens that begin or end with the straight apostrophe
6. Lowercase all the tokens (now safely called words in the program)
7. Sort the dictionary first alphabetically and then again by descending count since the alphabetic sorting will persist

Again I had to do a fair bit of research and trial and error with the regex I used and I needed help with the sort.

### Limitations

- I used the default .read() method and I'm not sure what encoding that assumes. From the docs online it looks like it assumes UTF-8 so if a corpus file is not encoded that way it would error I believe.
- Assumes no errors in SGML tags

### Goals completed

I was able to complete all stated goals of the project. The program takes one file path as its only argument and prints results to standard output in the specified format.
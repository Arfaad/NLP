import nltk
from nltk import CFG
from nltk.parse import ChartParser
# Define grammar
grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the' | 'a'
N -> 'student' | 'book'
V -> 'reads'
""")

# Create parser
parser = ChartParser(grammar)

# Input sentence
sentence = "the student reads a book"

# Tokenize
words = sentence.lower().split()

print("========== CONSTITUENCY PARSING ==========")

# Parse sentence
for tree in parser.parse(words):
    print(tree)
    tree.pretty_print()

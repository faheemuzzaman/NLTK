import nltk
import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
EXAMPLE_TEXT = '3138708556'

print(sent_tokenize(EXAMPLE_TEXT))
print(word_tokenize(EXAMPLE_TEXT))
print(nltk.bigrams(EXAMPLE_TEXT))





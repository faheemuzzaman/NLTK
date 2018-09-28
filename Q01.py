import nltk
import numpy as np # Union Plus Intersections
#nltk.download('gutenberg')
print(nltk.corpus.gutenberg.fileids())
emma_words = nltk.corpus.gutenberg.words('austen-emma.txt')

print(emma_words[0:30])
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(emma_words)
finder.apply_freq_filter(5)

emma_bigrams_pmi  = finder.nbest(bigram_measures.pmi, 30)
emma_bigrams_chi = finder.nbest(bigram_measures.chi_sq, 30)

print("\n")
print("\n")
print (emma_bigrams_pmi[0:29])
print("\n")
print("\n")
print (emma_bigrams_chi[0:29])
print("\n")
print("\n")
print(np.union1d(emma_bigrams_pmi, emma_bigrams_chi)) # Input union/intersection(arr,arr)
print("\n")
print("\n")
print(np.intersect1d(emma_bigrams_pmi, emma_bigrams_chi))

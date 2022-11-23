import random
from collections import namedtuple

class Aeda_Augmenter:
	def __init__(self,pct_words_to_swap=0.1, transformations_per_example=4):

		self.transformations_per_example = transformations_per_example
		self.pct_words_to_swap = pct_words_to_swap
        

		

	def insert_punctuation_marks(self,sentence):
		PUNCTUATIONS = ('.', ',', '!', '?', ';', ':')
		words = sentence.split(' ')
		new_line = []
		q = random.randint(1, int(self.pct_words_to_swap * len(words) + 1))
		qs = random.sample(range(0, len(words)), q)

		for j, word in enumerate(words):
			if j in qs:
				new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
				new_line.append(word)
			else:
				new_line.append(word)
		new_line = ' '.join(new_line)
		return new_line

	def augment(self,line):
		res = []
		for i in range(self.transformations_per_example):		
			new_line = self.insert_punctuation_marks(line)
			res.append(new_line)
		return res




import nltk
from nltk.stem import PorterStemmer

# create a Porter stemmer object
stemmer = PorterStemmer()


with open('word_all.txt', 'r', encoding='utf-8') as f:  # 得到词表
        word_list = f.read().split('\n')
f_out = open("negative_words_stem.txt", "w",encoding='utf-8')
for words in word_list:
       a=nltk.word_tokenize(words)
       for single_word in a:
            stemmed_word=stemmer.stem(single_word)
            f_out.write(stemmed_word+"\n")

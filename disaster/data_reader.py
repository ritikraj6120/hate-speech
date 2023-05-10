from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import pad_sequences
import functools
import pickle
import text_preprocess as tp
import codecs
import logging
import pickle as pk
import re

import nltk
import numpy as np
import pandas as pd
from nltk.text import TextCollection

# Stopwords list
sw_list = ['hai', 'vry', 'ha', 'bhi', 'i', 'me', 'my', 'sss', 'sdv', 'hh', 'xbh', 'bh', 'sj', 'myself', 'and',
           'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because', 'as', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 'can', 'will', 'just', 'should', "should've", 'now', 'll', 're']


def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in text_data:
        # Removing punctuations
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Converting lowercase and removing stopwords
        preprocessed_text.append(' '.join(token.lower()
                                          for token in nltk.word_tokenize(sentence)
                                          if token.lower() not in sw_list))

    return preprocessed_text

# from sklearn.feature_extraction.text import TfidfTransformer


def text_preprocessing(df2):
    df2['final'] = df2['tweet'].apply(str)
    import re
    # Remove HTTP tags
    df2['final'] = df2['final'].map(lambda x: ' '.join(
        re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))
    # Lower Case
    df2['final'] = df2['final'].map(lambda x: x.lower())
    # it will remove the old style retweet text "RT"
    df2['final'] = df2['final'].map(lambda x: re.sub(r'^RT[\s]+', '', x))
    # it will remove hashtags. We have to be careful here not to remove
    # the whole hashtag because text of hashtags contains huge information.
    # only removing the hash # sign from the word
    df2['final'] = df2['final'].map(lambda x: re.sub(r'#', '', x))
    # it will remove single numeric terms in the tweet.
    df2['final'] = df2['final'].map(lambda x: re.sub(r'[0-9]', '', x))
    # Remove punctuations
    df2['final'] = df2['final'].map(lambda x: re.sub(r'[^\w\s]', '', x))
    # Remove unicodes
    df2['final'] = df2['final'].map(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
    preprocessed_review = preprocess_text(df2['tweet'].values)
    df2['tweet'] = preprocessed_review
    return df2


logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'


def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(vocab_path):
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    # for index, token in enumerate(tokens):
    #     if token == '@' and (index + 1) < len(tokens):
    #         tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
    #         tokens.pop(index)
    return tokens


def get_dict():  # 字符词典 character vocabulary
    dict = {}
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    for i, c in enumerate(alphabet):
        dict[c] = i + 1
    return dict


def strToIndexs2(s, length=300):
    s = s.lower()
    m = len(s)
    n = min(m, length)
    str2idx = np.zeros(length, dtype='int64')
    dict = get_dict()
    for i in range(n):
        c = s[i]
        if c in dict:
            str2idx[i] = dict[c]
    return str2idx


def create_vocab(tweets, vocab_size=0):
    logger.info('Creating vocabulary.........')
    total_words, unique_words = 0, 0
    word_freqs = {}

    # next(input_file)
    for i in range(len(tweets)):

        tweet = tp.strip_hashtags(tweets[i])
        # print(tweet)
        content = tokenize(tweet)
        for word in content:
            try:
                word_freqs[word] += 1
            except KeyError:
                unique_words += 1
                word_freqs[word] = 1
            total_words += 1
    logger.info('  %i total words, %i unique words' %
                (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(
        list(word_freqs.items()), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq >= 1:
                vocab_size += 1
    # The number 2 means that it contains dirty words, 3 means don't contain.
    vocab = {'<pad>': 0, '<unk>': 1, '<word>': 2, '<no_word>': 3}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab


def get_indices(tweets, vocab, word_list_path):
    from sklearn import preprocessing
    import enchant
    d = enchant.Dict('en-US')

    with open(word_list_path, 'r', encoding='utf-8') as f:  # 得到词表
        word_list = f.read().split('\n')
    word_list = [s.lower() for s in word_list]

    data_x, char_x, ruling_embedding, category_embedding = [], [], [], []
    unk_hit, total = 0., 0.
    for i in range(len(tweets)):
        tweet = tp.strip_hashtags(tweets[i])
        # tweet = tweets[i]

        indices_char = []
        indices_char = strToIndexs2(tweet)
        content = tokenize(tweet)
        n = len(content)
        t = False
        indices = []
        category_indeics = []  # 脏词为0，错词为1，其他为2
        # for word in content:
        for j in range(n):
            # if is_number(word):
            #     indices.append(vocab['<num>'])
            #     num_hit += 1
            word = content[j]
            if j < n-1:
                word_2 = ' '.join(content[j:j+2])
            if j < n-2:
                word_3 = ' '.join(content[j:j+3])

            if word in word_list or word_2 in word_list or word_3 in word_list:  # 3-gram
                t = True

            if word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1

            # if word in word_list:
            #     category_indeics.append(0)
            # elif (not d.check(word)) and ('@' not in word) and (word not in
            #                                                     [':', ',', "''", '``', '!', "'s", '?', 'facebook', "n't", "'re", "'", "'ve", 'everytime']):
            #     category_indeics.append(1)
            # else:
            #     category_indeics.append(2)
            # total += 1
            if word in word_list:
                category_indeics.append(2)
            else:
                category_indeics.append(3)
            total += 1

        if t:
            ruling = [2]*50
        else:
            ruling = [3]*50
        # It corresponds to category embedding in the paper
        ruling_embedding.append(ruling)
        data_x.append(indices)
        char_x.append(indices_char)
        # It's just a category of words, it don't use now.
        category_embedding.append(category_indeics)
    logger.info('<unk> hit rate: %.2f%%' % (100 * unk_hit / total))
    return data_x, char_x, ruling_embedding, category_embedding


def turn2(Y):
    for i in range(len(Y)):
        if Y[i] == 2:
            Y[i] -= 1
    return Y


def read_dataset(args, vocab_path, MAX_SEQUENCE_LENGTH):
    from keras.utils import np_utils
    from sklearn.utils import shuffle
    df_task = pd.read_csv(args.data_path, encoding="latin-1")
    # Replace the labels with 1 for Explosion and 0 for NONE
    df_task['label'] = df_task['label'].replace({'Explosion': 1, 'NONE': 0})
    df_task = text_preprocessing(df_task)
    # df_task_test = pd.read_csv(args.trial_data_path, encoding="latin-1")
    df_task, df_task_test = train_test_split(df_task, test_size=0.2)
    # df_task_test=text_preprocessing(df_task_test)
    # df_task_test = df_task_test.head(1000)
    df_task['task_idx'] = [0]*len(df_task)
    df_task_test['task_idx'] = [0]*len(df_task_test)
    data_all = df_task[['tweet', 'label', 'task_idx']]
    # data_all = pd.concat([data_all, df_task[['tweet', 'label', 'task_idx']]], ignore_index=True)
    df_task_test = df_task_test[['tweet', 'label', 'task_idx']]
    print("test_task size>>>", len(df_task_test))
    if args.sentiment_data_path:
        df_sentiment = pd.read_csv(args.sentiment_data_path)
        df_sentiment = text_preprocessing(df_sentiment)
        # df_sentiment = df_sentiment.head(1000)
        df_sentiment['task_idx'] = [1]*len(df_sentiment)
        df_sentiment.sample(frac=1)  # shuffle data
        # df_sentiment['sequences_char_unorganized'] = get_indices(df_sentiment.tweet, vocab)
        df_sentiment_train, df_sentiment_test = df_sentiment.iloc[:int(
            0.5*len(df_task)), :], df_sentiment.iloc[int(0.99*len(df_sentiment)):, :]
        df_task_test = pd.concat([df_task_test, df_sentiment_test[[
                                 'tweet', 'label', 'task_idx']]], ignore_index=True)
        data_all = pd.concat(
            [data_all, df_sentiment_train[['tweet', 'label', 'task_idx']]], ignore_index=True)
    print("test_task size>>>", len(df_task_test))

    data_all.sample(frac=1)  # shuffle data
    if not vocab_path:
        data = data_all
        tweets = pd.concat([data, df_task_test], ignore_index=True).tweet
        vocab = create_vocab(tweets)
    else:
        vocab = load_vocab(vocab_path)
    logger.info('  Vocab size: %i' % (len(vocab)))

    data_tokens, train_chars, ruling_embedding_train, category_embedding_train = get_indices(
        data_all.tweet, vocab, args.word_list_path)  # 得到词、字符的索引
    X_test_data, test_chars, ruling_embedding_test, category_embedding_test = get_indices(
        df_task_test.tweet, vocab, args.word_list_path)
    Y = data_all['label']
    Y = turn2(Y)
    y_test = df_task_test['label']
    y_test = np_utils.to_categorical(y_test)
    dummy_y = np_utils.to_categorical(Y)
    X_train_data, y_train = data_tokens, dummy_y
    task_idx = np.array(list(data_all.task_idx), dtype='int32')
    task_idx_train = np_utils.to_categorical(task_idx)
    task_idx_test = np.array(list(df_task_test.task_idx), dtype='int32')
    task_idx_test = np_utils.to_categorical(task_idx_test)

    # X_train_data, X_test_data, y_train, y_test = train_test_split(X_train_data, y_train,
    #                                                               test_size=0.15,
    #                                                               random_state=20)

    # train_chars, test_chars, task_idx_train, task_idx_test = train_test_split(train_chars, task_idx_train, test_size=0.15, random_state=20)

    # ruling_embedding_train, ruling_embedding_test, category_embedding_train, category_embedding_test = train_test_split(ruling_embedding_train,
    #                                                                                     category_embedding_train, test_size=0.15, random_state=20)
    X_train_data = pad_sequences(
        X_train_data, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_test_data = pad_sequences(
        X_test_data, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    category_embedding_train = pad_sequences(
        category_embedding_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    category_embedding_test = pad_sequences(
        category_embedding_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    return X_train_data, X_test_data, y_train, y_test, np.array(train_chars), np.array(test_chars), task_idx_train, task_idx_test, np.array(ruling_embedding_train, dtype=np.single), \
        np.array(ruling_embedding_test,
                 dtype=np.single), category_embedding_train, category_embedding_test, vocab


def get_data(args):
    # data_path = args.data_path
    X_train_data, X_test_data, y_train, y_test, train_chars, test_chars, task_idx_train, task_idx_test, train_ruling_embedding, test_ruling_embedding, category_embedding_train, category_embedding_test, vocab = read_dataset(
        args, args.vocab_path, args.maxlen)

    return X_train_data, X_test_data, y_train, y_test, train_chars, test_chars, task_idx_train, task_idx_test, train_ruling_embedding, test_ruling_embedding, category_embedding_train, category_embedding_test, vocab


def hate_word_statistics(tweet_file_path, hate_word_file_path):
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    raw_data = pd.read_csv(tweet_file_path, sep=',', encoding="latin-1")
    with open(hate_word_file_path, "r") as f:
        hate_word_list = f.read().split('\n')
    # print(hate_word_list, type(hate_word_list))
    # raw_data.drop(raw_data.index[0], inplace=True)
    tweets = raw_data.tweet.values
    print(len(tweets))
    labels = raw_data['class'].values
    special_hate_word = ['jungle bunny',
                         'pissed off', 'porch monkey', 'blow job']
    # statistics = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5+', 'num_totals'], index=[0, 1, 2])
    statistics = np.zeros((3, 7), dtype=int)
    ruling_embedding = []
    print(statistics)
    for i in range(len(tweets)):
        statistics[int(labels[i]), 6] += 1  # 统计各个标签的数量
        tweet = tp.strip_hashtags(tweets[i])
        num_hate = 0
        for hate_word in special_hate_word:
            # print(tweet.find(hate_word))
            if tweet.find(hate_word) != -1:
                num_hate += 1
        content = tokenize(tweet)
        for word in content:
            word = wnl.lemmatize(word, pos=get_pos(word))
            if word in hate_word_list:
                # print(word)
                num_hate += 1
        # if num_hate != 0 and labels[i] == 2:
            # print(content)
        if num_hate < 6:
            statistics[labels[i], num_hate] += 1
        else:
            statistics[labels[i], 5] += 1
        if num_hate > 0:
            ruling_embedding.append([1, 1, 1, 1, 1])
        else:
            ruling_embedding.append([0, 0, 0, 0, 0])
    pd.DataFrame(ruling_embedding).to_csv('ruling_embedding.csv', index=False)
    statistics_df = pd.DataFrame(statistics, columns=[
                                 '0', '1', '2', '3', '4', '5+', 'num_totals'], index=[0, 1, 2])
    statistics_df.to_csv('tweet_statistics.csv')
    print(statistics_df)


if __name__ == '__main__':
    tweet_file_path = '../data/labeled_data.csv'
    hate_word_file_path = '../data/new_hate_word.txt'
    hate_word_statistics(tweet_file_path, hate_word_file_path)

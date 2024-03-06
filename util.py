import numpy as np
from sklearn.preprocessing import OneHotEncoder


def preprocess(text:str):
    corpus = []
    word_to_id = {}
    id_to_word = {}
    n = 0

    for word in text.split(' '):
        if word_to_id.get(word)==None:
            corpus.append(n)
            word_to_id[word]=n
            id_to_word[n]=word
            n+=1
        else:
            corpus.append(word_to_id[word])

    return corpus, word_to_id, id_to_word


def create_contexts_target(corpus, window_size=1):
    """
    맥락 : 
        [[0,2],
        [1,3]
        ...
        [1,6]
        ]
    타깃 : 
    [1,2,3,4,1,5]
    맥락과 타겟을 반환한다 
    """
    contexts = []
    target = []

    for i in range(len(corpus)-window_size-1):
        contexts.append([corpus[i], corpus[i+window_size+1]])
        target.append(corpus[i+window_size])

    return np.array(contexts), np.array(target)

def convert_one_hot(array, vocab_size):
    """
    array에 있는 값을 원 핫 인코딩으로 변환. 길이는 vocab_size

    return: encoded_array 반환
    """
    encoded_array = []
    for i in range(array.shape[0]):
        value = array[i]
        print(value)
        if isinstance(value, np.int64):
            one_hot_encoding = np.zeros(vocab_size)  # vocab_size의 길이를 갖는 one-hot 벡터 생성
            one_hot_encoding[value] = 1  # 해당 인덱스의 위치를 1로 설정하여 one-hot 벡터 생성
            print(one_hot_encoding)
            encoded_array.append(one_hot_encoding)  # 결과를 저장

        else:  # value가 리스트인 경우
            values = []
            for val in value:
                one_hot_encoding = np.zeros(vocab_size)  # vocab_size의 길이를 갖는 one-hot 벡터 생성
                one_hot_encoding[val] = 1  # 해당 인덱스의 위치를 1로 설정하여 one-hot 벡터 생성
                values.append(one_hot_encoding)
            
            encoded_array.append(values) 

    return np.array(encoded_array)

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


import warnings
warnings.filterwarnings('ignore')
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import numpy as np
nlp.utils.check_version('0.7.0')

EMB_PATH = 'glove.6B.'

class Embedder():
    def __init__(self, dimensions=100): # takes 50, 100, 2000, 300
        
        self.__d = dimensions # size of the embeddings of each word
        self.__emb_path = EMB_PATH + str(self.__d)  + 'd'

        temp = nlp.embedding.create('glove', source= self.__emb_path)
        embedder = nlp.Vocab(nlp.data.Counter(temp.idx_to_token))
        embedder.set_embedding(temp)
        
        self.__embedder = embedder  # embedder object
        self.__emb_mapper = self.__embedder.embedding # maps words to the embeddings


    def __norm_vecs_by_row(self, x):

        return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

    
    def __cos_sim(self, x, y):

            return nd.dot(x, y) / (nd.norm(x) * nd.norm(y)) 


    def get_embedder(self):
        '''
        Returns an embedder dictionary that returns embeddings of words.
        Eg.
        emb = Embedder(dimensions = 50)
        embedding = emb.get_embedder()
        embedding['beautiful'] 

        Returns...
            50 dimension embedding for the word 'beautiful' in the form of NDArray
        '''
        return self.__emb_mapper
    

    def most_similar_to(self,word, k=5):
        '''
        Returns top k words similar to the argument.
        Eg.
        emb = Embedder(dimensions = 50)
        print(emb.most_similar_to('baby'))
        Returns...
            ['babies', 'boy', 'girl', 'newborn', 'pregnant']
        '''        
        vec = self.__emb_mapper[word].reshape((-1,1))
        emb_vecs = self.__norm_vecs_by_row(self.__emb_mapper.idx_to_vec)
        dot_product = nd.dot(emb_vecs, vec)
        indices = nd.topk(dot_product.reshape((len(self.__embedder), )), k = k+1, ret_typ='indices')
        indices = [int(i.asscalar()) for i in indices]
        # Remove unknown and input tokens.

        return self.__embedder.to_tokens(indices[1:])


    def get_top_k_by_analogy(self, word1, word2, word3, k=1):
        '''
        Returns analogical word for the set of 3 words that are passed as arguments.
        Analogy refers to:
            king->queen ; man->woman
            good->better ; bad->worse
            do->did ; go->went
        Eg.
        emb= Embedder(dimensions=50)
        print(emb.get_top_k_analogy('good','best','bad'))
        Returns...
            ['worst'] 
        Returns a list because you can have top k analogies as result
        '''
        word_vecs = self.__emb_mapper[word1, word2, word3]
        word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
        vocab_vecs = self.__norm_vecs_by_row(self.__emb_mapper.idx_to_vec)
        dot_product = nd.dot(vocab_vecs, word_diff)
        indices = nd.topk(dot_product.reshape((len(self.__embedder), )), k=k, ret_typ='indices')
        indices = [int(i.asscalar()) for i in indices]

        return self.__embedder.to_tokens(indices)
    

    def cosine_similarity(self, word1, word2):
        '''
        Returns cosine of the angle between the embeddings of the two words.
        Eg.
        emb = Embedder(dimensions = 50)
        print(emb.cosine_similarity('good', 'bad'))
            0.79648924
        print(emb.cosine_similarity('good', 'bad'))
            0.8546279
        '''
        x = self.__emb_mapper[word1]
        y = self.__emb_mapper[word2]

        return np.squeeze(self.__cos_sim(x,y).asnumpy())


    def cosine_sim_analogy(self, word1, word2, word3, word4):
        '''
        Returns correctness of an analogy.
        Scores closer to 1 indicate better analogy.
        Eg.
        emb = Embedder(dimensions=50)
        print(emb.cosine_sim_analogy('man', 'woman', 'son', 'daughter'))
        Returns...
            0.9658341
        '''
        words = [word1, word2, word3, word4]
        vecs = self.__emb_mapper[words]

        return np.squeeze(self.__cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3]).asnumpy())


    def get_embedding_matrix(self, vocab, num_words, oov_token='<oov>'):
        '''
        Parameters:
            vocab (list): top {num_words} words in the tokenized dataset

            num_words (int): size of the vocabulary

        Returns:
            embedding_matrix, idx_to_token (dict), token_to_idx (dict)

        When using pre-trained word embeddings, the Embedding Layer from tensorflow.keras.layers demands for a parameter 'weights'. 
        We pass: weights = [embedding_matrix] for the same...
        This function returns embedding_matrix for your vocabulary
        Eg.
        emb = Embedder(dimensions=50)
        embedding_matrix, id_to_token, token_to_id = emb.get_embedding_matrix(vocabulary, num_words = 5000)
        '''
        token_to_idx = {}
        idx_to_token = {}
        embedding_matrix = []

        if oov_token not in vocab:
            embedding_matrix.append(self.__emb_mapper[oov_token].asnumpy())
            token_to_idx[oov_token] = 0
            idx_to_token[0] = oov_token

            for i in range(num_words-1):
                word = vocab[i].lower()
                token_to_idx[word] = i+1
                idx_to_token[i+1] = word
                embedding_matrix.append(self.__emb_mapper[word].asnumpy())

        else:
            for i in range(num_words):
                word = vocab[i].lower()
                token_to_idx[word] = i
                idx_to_token[i] = word
                embedding_matrix.append(self.__emb_mapper[word].asnumpy())
        
        embedding_matrix = np.asarray(embedding_matrix)

        return (
            embedding_matrix,
            idx_to_token,
            token_to_idx
        )


def generate_vocabulary(tokenized_sentences):
    '''
    Parameters:
        tokenized_sentences : list of tokenised sentences of the dataset
    Returns:
        vocab -> list of words in the dataset sorted from highest occurance to lowest
        vocab_dict -> dictionary that maps words to the count of their occurance in the dataset
    '''
    vocab_dict = dict()
    
    # getting vocab dictionary
    for sentence in tokenized_sentences:
        for word in sentence:
            vocab_dict[word] = vocab_dict.get(word,0)+1
    
    # sorting vocab_dict from highest frequency to lowest
    vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)}

    vocab = [word for word in vocab_dict]

    return vocab, vocab_dict
    

def token_seq_to_num_seq(token_sequences, token_to_idx, oov_token):
    '''
    Paramters:
        token_sequences -> list of word sequences generated by tokenizing the word sentences
        token_to_idx -> dictionary that maps tokens to ID's
        oov_token -> token used for out of vocabulary words

    Returns:
        num_sequences -> list of number sequences generated by mapping tokens to their ID's

    Eg.
    emb = Embedder(dimensions=50)
    embedding_matrix, id2token, token2id = emb.get_embedding_matrix(vocab, num_words)
    token_seq_to_num_seq(token_sequences, token2id, oov_token = '<oov>')
    '''
    num_sequences = []
    for sequence in token_sequences:
        num_sequences.append(np.asarray([ token_to_idx.get(token.lower(), token_to_idx.get(oov_token,0)) for token in sequence]))
    
    return num_sequences


def num_seq_to_token_seq(num_sequences, idx_to_token):
    '''
    Parameters: 
        num_sequences -> list of sequences represented using ID's of the tokens
        idx_to_token -> dictionary that maps ID's -> tokens
    
    Returns:
        token_sequences -> list of token sequences for the corresponding number sequences
    
    Eg.
    X = token_seq_to_num_seq(token_sequences, token2id, oov_token = '<oov>')
    num_seq_to_num_seq(X, id2token)
    '''
    token_sequences = []
    for sequence in num_sequences:
        token_sequences.append([ idx_to_token[i] for i in sequence])
    
    return token_sequences
import numpy as np
import pickle
def load_w2v(fin, type, vector_size):
    """
    Load word vector file.
    :param fin: input word vector file name.
    :param type: word vector type, "Google" or "Glove" or "Company".
    :param vector_size: vector length.
    :return: Output Gensim word2vector model.
    """
    model = {}
    if type == "Google" or type == "Glove":
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fin, binary=True)
    elif type == "Company":
        model["PADDING"] = np.zeros(vector_size)
        model["UNKNOWN"] = np.random.uniform(-0.25, 0.25, vector_size)
        with open(fin, "r", encoding="utf-8") as fread:
            for line in fread.readlines():
                line_list = line.strip().split(" ")
                word = line_list[0]
                word_vec = np.fromstring(" ".join(line_list[1:]),
                                         dtype=float, sep=" ")
                model[word] = word_vec
    else:
        print("type must be Glove or Google or Company.")
        sys.exit(1)
    print(type)
    return model
if __name__ == "__main__":
    W2V = load_w2v("../data/word_embedding/w2v-zh.model", "Company", 200)
    W2V = dict((k, W2V[k]) for k in W2V.keys() if len(W2V[k]) == 200)
    #W2V = OrderedDict(W2V)
    with open('../data/word_embedding/w2v-zh.model.pkl', 'wb') as handle:
        pickle.dump(W2V, handle, protocol=pickle.HIGHEST_PROTOCOL)
    word_to_ix = {word: i for i, word in enumerate(W2V)}
    MAX_LEN = 200  # maximum text length for each vertex
    embed_size = len(list(W2V.values())[0])
    print("W2V loaded! \nVocab size: %d, Embedding size: %d" % (len(word_to_ix), embed_size))
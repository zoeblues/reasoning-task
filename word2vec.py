import numpy as np
import pandas as pd
import re
from collections import Counter

from patsy.state import center


class Word2VecSGNS:
    """
    Skip-gram with Negative Sampling implemented using only NumPy.
    """

    def __init__(self, embedding_dim=50, window_size=5, neg_samples=5, lr=0.025):
        self.eta = lr # learning rate
        self.dim = embedding_dim # size of the vector
        self.window = window_size
        self.k = neg_samples # number of noise samples in SGNs
        self.word2idx = {} # mapping word to index
        self.idx2word = [] # mapping index to word
        self.vocab_size = 0 # unique words after preprocessing
        self.W_center = None  # center word embeddings
        self.W_context = None  # context word embeddings
        self.unigram_table = None

    # convert alphabetic tokens to lowercase
    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    # build vocabulary and perform subsampling
    def build_vocab(self, raw_text, subsample_threshold=1e-3):
        tokens = self._tokenize(" ".join(raw_text))
        counts = Counter(tokens)
        total_count = len(tokens)

        # subsampling formula: P(w) = 1 - sqrt(t / f(w))
        kept_tokens = []
        for word in tokens:
            freq = counts[word] / total_count
            prob_keep = np.sqrt(subsample_threshold / freq) # Mikolov's formula
            if np.random.random() < prob_keep:
                kept_tokens.append(word)

        self.idx2word = list(set(kept_tokens))
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}
        self.vocab_size = len(self.idx2word)

        # initialize weights
        limit = np.sqrt(6 / (self.vocab_size + self.dim))
        self.W_center = np.random.uniform(-limit, limit, (self.vocab_size, self.dim))
        self.W_context = np.zeros((self.vocab_size, self.dim))

        # build unigram table for negative sampling
        word_pow = np.array([counts[w] ** 0.75 for w in self.idx2word])
        self.unigram_table = word_pow / word_pow.sum()

        return kept_tokens

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def train_step(self, center_idx, context_idx, current_lr):
        center_vector = self.W_center[center_idx]
        context_vector = self.W_context[context_idx]

        # negative sampling
        neg_indices = np.random.choice(self.vocab_size, size=self.k, p=self.unigram_table)
        u_neg = self.W_context[neg_indices]

        # forward pass
        score_pos = np.dot(context_vector, center_vector) # dot product for similarity score
        prob_pos = self._sigmoid(score_pos) # turn score into probability

        score_neg = np.dot(u_neg, center_vector) # dot products for the k fake pairs
        prob_neg = self._sigmoid(score_neg) # probability for fake pairs

        # backward pass - gradients
        grad_context_word = (prob_pos - 1) * center_vector # error on positive paris times the target vector
        grad_negative_word = np.outer(prob_neg, center_vector) # error on negative pairs times the target vector
        grad_center_word = (prob_pos - 1) * context_vector + np.dot(prob_neg, u_neg) # total error across all pairs

        # Stochastic Gradient Descent updates
        self.W_context[context_idx] -= current_lr * grad_context_word
        self.W_context[neg_indices] -= current_lr * grad_negative_word
        self.W_center[center_idx] -= current_lr * grad_center_word

        # return loss
        loss = -np.log(prob_pos + 1e-9) - np.sum(np.log(1 - prob_neg + 1e-9))
        return loss

    def train(self, tokens, epochs=10):
        for epoch in range(epochs):
            # linearly decay the learning rate
            current_lr = self.eta * (1 - epoch / epochs)
            total_loss = 0

            for i, word in enumerate(tokens):
                if word not in self.word2idx:
                    continue

                center_idx = self.word2idx[word]

                # context window
                start = max(0, i - self.window)
                end = min(len(tokens), i + self.window + 1)

                for j in range(start, end):
                    if i == j or tokens[j] not in self.word2idx:
                        continue

                    context_idx = self.word2idx[tokens[j]]
                    total_loss += self.train_step(center_idx, context_idx, current_lr)

            avg_loss = total_loss / len(tokens)
            print(f"epoch {epoch + 1}/{epochs} | loss: {avg_loss:.4f}")

    def get_most_similar(self, word, n=5):
        if word not in self.word2idx: return "Word not in vocabulary."
        v = self.W_center[self.word2idx[word]]

        # Cosine Similarity using vectorization
        norm_v = np.linalg.norm(v)
        norm_all = np.linalg.norm(self.W_center, axis=1)
        similarities = np.dot(self.W_center, v) / (norm_all * norm_v + 1e-9)

        top_indices = np.argsort(similarities)[::-1][1:n + 1]
        return [(self.idx2word[i], similarities[i]) for i in top_indices]

def load_real_data(file_path):
    df = pd.read_csv(file_path)
    # fill missing values to prevent str() conversion issues
    df['description'] = df['description'].fillna('')
    df['transcription'] = df['transcription'].fillna('')

    text_data = df['description'] + " " + df['transcription']
    return text_data.tolist()


if __name__ == "__main__":
    medical_data = load_real_data("demo.csv")

    model = Word2VecSGNS(embedding_dim=20, window_size=3, neg_samples=5)
    processed_tokens = model.build_vocab(medical_data)
    model.train(processed_tokens, epochs=100)

    # example usage
    print("\nSimilarity search:")
    print(f"Heart: {model.get_most_similar('heart')}")
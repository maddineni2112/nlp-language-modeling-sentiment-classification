# -*- coding: utf-8 -*-
# Assignment 2 - Task 1
# N-gram Language Model (Bigram) + Laplace Smoothing
# Stages:
# 0: load corpus text
# 1: tokenization (basic tokenize + sentence markers)
# 2: gram counting (unigram + bigram)
# 3: Maxium Likelihood Estimation (MLE) bigram probability + unseen pair demo
# 4: Laplace (add-one) smoothing demo
# 5: Testing.


# How to use STAGE:
# ------------------------------------------------------------
# 1. Start with STAGE = 0.
# 2. Implement only the function(s) required for that stage.
# 3. Run the script and inspect the printed output.
# 4. When the stage works correctly, increase STAGE by 1.
# 5. Do NOT skip stages.
#
# Important:
# - You may ONLY modify:
#     (a) STAGE
#     (b) Functions marked with TODO
# - You must NOT modify:
#     Runner
#     Printing helpers
#     Test logic
#
# Corpus:
# ------------------------------------------------------------
# 1. Download the text (.txt) file from Canvas.
# 2. Place it in the same directory as this script.
# 3. Set CORPUS_PATH to the filename below.
#
# Example:
#     CORPUS_PATH = "task1Data.txt"
import os
STAGE = 9
CORPUS_PATH = "task1Data.txt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH_ = os.path.join(SCRIPT_DIR, CORPUS_PATH)
import os
########################################
# Stage 0: Load corpus
########################################
def load_corpus(path: str) -> str:
    """
    Read txt file and return a cleaned lowercase string.
    Requirements:
    - Read entire file. (you can use open function or any NLP Lib)
    - Convert to lowercase.
    - Return as a single string.
    """
    # TODO (Stage 0)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.lower()
########################################
# Stage 1: Tokenization

# IMPORTANT NOTE:
# ------------------------------------------------------------
# This tokenization is intentionally minimal.
#
# In Assignment 1 (BPE), tokenization actively constructed
# a vocabulary by merging symbol pairs.
#
# Here, we do NOT construct or modify the vocabulary.
# We simply split text into tokens using whitespace,
# then insert sentence boundary markers.
#
# Specifically, we DO NOT:
# - build a dictionary
# - merge small tokens into large ones
# - deal with punctuation
# - apply subword rules
#
# This stage treats the corpus as a sequence of observed words.
# The vocabulary will later be derived from counts,
# not constructed during tokenization.

########################################

def tokenize(text: str):
    """
    Tokenization = basic whitespace tokenization + sentence boundary markers.
    Rules:
    - Split on whitespace (text.split()).
    - Treat tokens ending with '.' as sentence boundary.
    - Insert <s> at the start of the corpus and after each boundary.
    - Insert </s> after each boundary token.
    - Keep the original token that ends with '.' as-is (do not strip '.').
    Return:
    - tokens_with_markers: list[str]
    """
    # TODO (Stage 1)
    raw_tokens = text.split()
    tokens_with_markers = ["<s>"]

    for token in raw_tokens:
        tokens_with_markers.append(token)
        if token.endswith("."):
            tokens_with_markers.append("</s>")
            tokens_with_markers.append("<s>")

    if len(tokens_with_markers) > 0 and tokens_with_markers[-1] == "<s>":
        tokens_with_markers.pop()

    return tokens_with_markers
########################################
# Stage 2: Gram counting
########################################

def count_ngrams(tokens, n: int):
    """
    Count n-grams from a token list.
    Requirements:
    - Return a dict mapping tuple(tokens[i:i+n]) -> count
    - Example: for bigram, key is (w1, w2)
    """
    # TODO (Stage 2)
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts
def build_vocabulary(unigram_counts: dict):
    """
    Build a vocabulary list from unigram counts.
    Return:
    - vocab: sorted list of tokens
    """
    # TODO (Stage 2)
    vocab = [gram[0] for gram in unigram_counts.keys()]
    return sorted(vocab)
########################################
# Stage 3: MLE probability (unsmoothed)

# IMPORTANT NOTE:
# ------------------------------------------------------------
# In class, we derived that the Maximum Likelihood Estimate (MLE)
# of a bigram conditional probability is:
#
#     P(w2 | w1) = count(w1, w2) / count(w1)
#
# This solution comes from:
#   - Maximizing the likelihood of the observed corpus
#   - Under the constraint that probabilities sum to 1
#
# Therefore, the count you compute here is not arbitrary.
# It is the closed-form solution of a constrained optimization problem.
#
# However, two important conceptual points:
#
# (1) The bigram model does NOT construct a full distribution
#     over all possible vocabulary items.
#     It only assigns non-zero probability to observed pairs.
#     Unseen pairs implicitly receive probability 0.
#
#     So P(w2 | w1) is empirical frequency,
#     not a learned distribution over all possible words.
#
# (2) The model depends only on relative position (adjacency),
#     not absolute position in the corpus.
#
#     If the word "dog" appears at position 10 or 5000,
#     the model treats both occurrences identically.
#
#     This is the Markov assumption:
#     The next word depends only on the previous word,
#     not on the full history or absolute index.
#
# These two assumptions define what an N-gram model is.
########################################
def bigram_prob_mle(w2: str, w1: str, unigram_counts: dict, bigram_counts: dict) -> float:
    """
    Unsmoothed MLE:
    P(w2 | w1) = count(w1,w2) / count(w1)
    Requirements:
    - If count(w1) == 0, return 0.0
    - If (w1,w2) not in bigram_counts, treat count as 0
    """
    # TODO (Stage 3)
    count_w1 = unigram_counts.get((w1,), 0)
    if count_w1 == 0:
        return 0.0

    count_w1_w2 = bigram_counts.get((w1, w2), 0)
    return count_w1_w2 / count_w1
# Stage 4: Laplace smoothing (add-one)
########################################
def bigram_prob_laplace(w2: str, w1: str, unigram_counts: dict, bigram_counts: dict, vocab_size: int) -> float:
    """
    Laplace (add-one) smoothing:
    P(w2 | w1) = (count(w1,w2) + 1) / (count(w1) + V)
    Requirements:
    - If count(w1) == 0, use denominator (0 + V)
    - Works even for unseen (w1,w2)
    """
    # TODO (Stage 4)
    count_w1 = unigram_counts.get((w1,), 0)
    count_w1_w2 = bigram_counts.get((w1, w2), 0)

    return (count_w1_w2 + 1) / (count_w1 + vocab_size)

########################################
# Stage 5: Perplexity test
#
# IMPORTANT NOTE:
# ------------------------------------------------------------
# Perplexity is a standard evaluation metric for language models.
# It measures how well a model predicts held-out (unseen) text.
#
# In principle, perplexity must be computed on data that was NOT
# used for training. This is called a hold-out evaluation.
#
# In this assignment:
#   - The runner already splits the corpus into train and test.
#   - You only need to implement the perplexity computation.
#
# Bigram perplexity over tokens w_0, w_1, ..., w_T is:
#
#     PP = exp( - (1/N) * sum_{t=1..T} log P(w_t | w_{t-1}) )
#
# where:
#   - N = number of predicted tokens = len(tokens) - 1
#   - log is natural logarithm
#
# Conceptual observations:
#
# (1) If an unsmoothed MLE model assigns probability 0
#     to any bigram in the test set,
#     perplexity becomes infinite.
#
# (2) Laplace smoothing avoids zero probabilities,
#     but redistributes probability mass.
#
# (3) Perplexity is meaningful only when evaluated
#     on data not used for parameter estimation.
#
# Your task here is ONLY to compute perplexity correctly.
########################################
import math
def compute_perplexity(tokens, unigram_counts, bigram_counts, vocab_size, use_laplace=True):
    """
    Compute bigram perplexity on a token list.
    Requirements:
    - Use natural log (math.log).
    - If any bigram probability is 0.0, return float("inf").
    - N should be the number of predicted tokens = len(tokens)-1.
    - If len(tokens) < 2, return float("inf").
    """
    # TODO (Stage 5)
    if len(tokens) < 2:
        return float("inf")

    log_prob_sum = 0.0
    N = len(tokens) - 1

    for i in range(1, len(tokens)):
        w1 = tokens[i - 1]
        w2 = tokens[i]

        if use_laplace:
            prob = bigram_prob_laplace(w2, w1, unigram_counts, bigram_counts, vocab_size)
        else:
            prob = bigram_prob_mle(w2, w1, unigram_counts, bigram_counts)

        if prob == 0.0:
            return float("inf")

        log_prob_sum += math.log(prob)

    return math.exp(-log_prob_sum / N)


# Printing helpers (DO NOT MODIFY)
########################################
def top_k(counter_dict: dict, k=20):
    items = list(counter_dict.items())
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:k]
def print_top_ngrams(ngram_counts: dict, k=20, title="Top n-grams"):
    print(title)
    for key, c in top_k(ngram_counts, k=k):
        if isinstance(key, tuple):
            print("  " + " ".join(key) + " -> " + str(c))
        else:
            print("  " + str(key) + " -> " + str(c))
    print()




# Assignment 2 - Task 2
# Logistic Regression for Sentiment Analysis (from scratch)
#
# You only change STAGE and functions marked with TODO.
# Do NOT modify runner or printing helpers.
#
# Stages (continuing after Task 1 Stage 5):
# 6: load supervised data (CSV)
# 7: preprocess (tokenization + BoW representation)
# 8: train logistic regression (gradient descent)
# 9: test + evaluation

import os
import csv
import math
import random

DATA_FILENAME = "task2Data.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, DATA_FILENAME)

SEED = 42
TRAIN_RATIO = 0.8


########################################
# Stage 6: Load supervised data

# IMPORTANT NOTE:
# ------------------------------------------------------------
# This task uses supervised data: each example consists of
# a text document and a binary label (0 or 1).
#
# In supervised learning, evaluation must be performed on
# data that is not used for training.
#
# In this assignment:
#   - The runner will split the dataset into train and test.
#   - You do NOT need to implement the split.
#   - Your only responsibility in this stage is to
#     correctly load all (text, label) pairs from the CSV file.
#
# Make sure:
#   - Texts are read as strings.
#   - Labels are converted to integers (0 or 1).
########################################

def load_supervised_csv(path: str):
    """
    IMPORTANT NOTE:
    - This task uses supervised data (text, label).
    - The runner will create a train/test split for you.
    - Your implementation only needs to load all examples from the CSV.

    Return:
    - texts: list[str]
    - labels: list[int] (0 or 1)

    CSV format expected:
    - header row exists
    - columns include: "text" and "label"
    """
    # TODO (Stage 6)
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            text = row["text"]
            label = int(row["label"])  # ensure integer (0 or 1)

            texts.append(text)
            labels.append(label)

    return texts, labels

########################################
# Stage 7: Preprocess (Tokenization + BoW)
########################################

def tokenize_simple(text: str):
    """
    IMPORTANT NOTE:
    - Logistic regression does not define "features" by itself.
      We must choose a representation.
    - Here we use Bag-of-Words (BoW): data-driven but NOT learned.
    - Vocabulary is built ONLY from the training set.
    - BoW ignores word order (contrast with Task 1 n-gram).

    Minimal tokenization:
    - lowercase
    - whitespace split
    """
    # TODO (Stage 7)
    return text.lower().split()


def build_vocab(texts, min_freq=1):
    """
    Build vocabulary from training texts only.

    Return:
    - vocab: dict token -> index (0..V-1)

    Notes:
    - Only include tokens with frequency >= min_freq.
    """
    # TODO (Stage 7)
    from collections import Counter

    counter = Counter()

    for text in texts:
        tokens = tokenize_simple(text)
        counter.update(tokens)

    vocab = {}
    idx = 0

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab


def vectorize_bow(text: str, vocab: dict):
    """
    Convert one document to a BoW vector (counts).

    Return:
    - x: list[float] length V

    Notes:
    - OOV tokens are ignored (treated as zero count).
    """
    # TODO (Stage 7)
    x = [0.0] * len(vocab)

    tokens = tokenize_simple(text)

    for token in tokens:
        if token in vocab:
            x[vocab[token]] += 1.0

    return x


########################################
# Stage 8: Train Logistic Regression
#
# IMPORTANT NOTE:
# ------------------------------------------------------------
# Logistic regression models:
#
#   p(y=1 | x) = sigmoid(w · x + b)
#
# where:
#   - w is the weight vector
#   - b is the bias
#   - sigmoid(z) = 1 / (1 + exp(-z))
#
# We train the model by maximizing likelihood,
# which is equivalent to minimizing negative log-likelihood
# (binary cross-entropy loss):
#
#   L = - [ y log(p) + (1-y) log(1-p) ]
#
# From this objective, the gradient can be derived.
#
# For one training example:
#
#   ∂L/∂w = (p - y) * x
#   ∂L/∂b = (p - y)
#
# This leads to the gradient descent update rule:
#
#   w := w - lr * (p - y) * x
#   b := b - lr * (p - y)
#
# Key ideas:
#   - The term (p - y) measures prediction error.
#   - If prediction is correct, update is small.
#   - If prediction is wrong, update is large.
#   - This is not a heuristic: it comes directly from
#     maximum likelihood estimation.
#
# Do not use external libraries (sklearn/torch).
########################################


def sigmoid(z: float) -> float:
    # stable-ish sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def train_logreg(X, y, lr=0.1, epochs=10):
    """
    IMPORTANT NOTE:
    - Binary logistic regression models:
        p(y=1|x) = sigmoid(w·x + b)
    - Train by minimizing negative log-likelihood (cross-entropy).
    - Use gradient descent (no sklearn/torch).

    Inputs:
    - X: list[list[float]]  (N vectors)
    - y: list[int]         (N labels, 0/1)

    Return:
    - w: list[float]
    - b: float
    """
    # TODO (Stage 8)
    if len(X) == 0:
        return [], 0.0

    num_features = len(X[0])
    w = [0.0] * num_features
    b = 0.0

    for epoch in range(epochs):
        total_loss = 0.0

        for xi, yi in zip(X, y):
            z = sum(wj * xj for wj, xj in zip(w, xi)) + b
            p = sigmoid(z)

            error = p - yi

            for j in range(num_features):
                w[j] -= lr * error * xi[j]

            b -= lr * error

            eps = 1e-12
            total_loss += -(yi * math.log(p + eps) + (1 - yi) * math.log(1 - p + eps))

        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return w, b


########################################
# Stage 9: Test + Evaluation

# IMPORTANT NOTE:
# ------------------------------------------------------------
# There are multiple ways to evaluate a binary classifier.
#
# In this assignment, the runner reports accuracy by default.
# You do NOT need to modify the accuracy implementation.
#
# However, accuracy alone is often insufficient.
# According to the textbook definitions, you must compute
# and report the following metrics on the TEST set:
#
#   - Precision
#   - Recall
#
# Definitions (binary classification):
#
#   True Positive (TP): predicted 1, true label 1
#   False Positive (FP): predicted 1, true label 0
#   False Negative (FN): predicted 0, true label 1
#
#   Precision = TP / (TP + FP)
#   Recall    = TP / (TP + FN)
#
# You are expected to:
#   1. Manually compute TP, FP, FN from predictions.
#   2. Report precision and recall in your submission.
#
# Do NOT change the runner. Perform your calculation
# separately after observing predictions.
########################################
def predict_proba(x, w, b) -> float:
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    return sigmoid(z)


def predict_label(x, w, b, threshold=0.5) -> int:
    return 1 if predict_proba(x, w, b) >= threshold else 0


def evaluate_accuracy(X, y, w, b):
    correct = 0
    for xi, yi in zip(X, y):
        if predict_label(xi, w, b) == yi:
            correct += 1
    return correct / max(1, len(y))

def compute_precision_recall(X, y, w, b):
    """
    Return:
    - precision
    - recall
    - f1
    """
    # TODO (Stage 9)
    TP = 0
    FP = 0
    FN = 0

    for xi, yi in zip(X, y):
        pred = predict_label(xi, w, b)

        if pred == 1 and yi == 1:
            TP += 1
        elif pred == 1 and yi == 0:
            FP += 1
        elif pred == 0 and yi == 1:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1









########################################
# Runner (DO NOT MODIFY)
########################################
def run():
    print("STAGE =", STAGE)
    print()
    if STAGE >= 0:
        print("Stage 0: Load corpus (edit load_corpus).")
        text = load_corpus(CORPUS_PATH_)
        print("Corpus length (characters):", len(text))
        print("First 250 characters:")
        print(text[:250])
        print()
    if STAGE >= 1:
        print("Stage 1: Tokenization (edit tokenize).")
        tokens = tokenize(text)
        print("Total tokens (with markers):", len(tokens))
        print("First 80 tokens:")
        print(tokens[:80])
        print()
        print("Concept check: markers are inserted by your code, not present in the text.")
        print("If you move a sentence to another place in the corpus, bigram counting still only sees adjacency, not absolute position.")
        print()
    if STAGE >= 2:
        print("Stage 2: Gram counting (edit count_ngrams, build_vocabulary).")
        unigram_counts = count_ngrams(tokens, 1)
        bigram_counts = count_ngrams(tokens, 2)
        vocab = build_vocabulary(unigram_counts)
        print("Vocabulary size:", len(vocab))
        print_top_ngrams(unigram_counts, k=20, title="Top 20 unigrams:")
        print_top_ngrams(bigram_counts, k=20, title="Top 20 bigrams:")
    if STAGE >= 3:
        print("Stage 3: MLE bigram probability (edit bigram_prob_mle).")
        unigram_counts = count_ngrams(tokens, 1)
        bigram_counts = count_ngrams(tokens, 2)
        demo_pairs = [("the", "of"), ("the", "unicorn"), ("</s>", "<s>"), ("<s>", "the")]
        print("MLE demos: P(w2 | w1)")
        for w1, w2 in demo_pairs:
            p = bigram_prob_mle(w2, w1, unigram_counts, bigram_counts)
            print(f"  P({w2} | {w1}) = {p}")
        print()
        print("Key idea: unsmoothed MLE only assigns non-zero probability to observed pairs; unseen pairs get 0.")
        print("This is not a full conditional distribution learned from first principles, it is empirical frequency with zeros for unseen events.")
        print()
    if STAGE >= 4:
        print("Stage 4: Laplace smoothing (edit bigram_prob_laplace).")
        unigram_counts = count_ngrams(tokens, 1)
        bigram_counts = count_ngrams(tokens, 2)
        vocab = build_vocabulary(unigram_counts)
        V = len(vocab)
        demo_pairs = [("the", "of"), ("the", "unicorn"), ("</s>", "<s>"), ("<s>", "the")]
        print("Laplace demos: P(w2 | w1)")
        for w1, w2 in demo_pairs:
            p = bigram_prob_laplace(w2, w1, unigram_counts, bigram_counts, V)
            print(f"  P({w2} | {w1}) = {p}")
        print()
        print("Compare Stage 3 vs Stage 4: Laplace makes unseen pairs non-zero, at the cost of shifting mass from seen pairs.")
        print()
    if STAGE >= 5:
        print("Stage 5: Perplexity test (edit compute_perplexity).")

        # Split the SAME corpus into train/test by position.
        # This keeps the assignment self-contained (no extra files).
        split = int(0.9 * len(tokens))
        train_tokens = tokens[:split]
        test_tokens = tokens[split:]

        unigram_train = count_ngrams(train_tokens, 1)
        bigram_train = count_ngrams(train_tokens, 2)
        vocab = build_vocabulary(unigram_train)
        V = len(vocab)

        pp_mle = compute_perplexity(
            test_tokens, unigram_train, bigram_train, V, use_laplace=False
        )
        pp_lap = compute_perplexity(
            test_tokens, unigram_train, bigram_train, V, use_laplace=True
        )

        print("Train tokens:", len(train_tokens))
        print("Test tokens:", len(test_tokens))
        print("Vocabulary size (train):", V)
        print("Perplexity (MLE, unsmoothed):", pp_mle)
        print("Perplexity (Laplace):", pp_lap)
    if STAGE >= 6:
        print("Stage 6: Load supervised data (edit load_supervised_csv).")
        texts, labels = load_supervised_csv(DATA_PATH)
        print("Total examples:", len(texts))
        print("First example:")
        print("  label =", labels[0])
        print("  text  =", texts[0][:120])
        print()

        random.seed(SEED)
        idx = list(range(len(texts)))
        random.shuffle(idx)

        split = int(TRAIN_RATIO * len(idx))
        train_idx = idx[:split]
        test_idx = idx[split:]

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        print("Train size:", len(train_texts))
        print("Test size :", len(test_texts))
        print()

    if STAGE >= 7:
        print("Stage 7: Preprocess (edit tokenize_simple, build_vocab, vectorize_bow).")
        vocab = build_vocab(train_texts, min_freq=1)
        V = len(vocab)
        print("Vocabulary size (train):", V)

        X_train = [vectorize_bow(t, vocab) for t in train_texts]
        X_test = [vectorize_bow(t, vocab) for t in test_texts]

        print("Vector length check:", len(X_train[0]), "(should equal V)")
        print()

    if STAGE >= 8:
        print("Stage 8: Train logistic regression (edit train_logreg).")
        w, b = train_logreg(X_train, train_labels, lr=0.1, epochs=10)
        print("Training finished.")
        print()

    if STAGE >= 9:
        print("Stage 9: Test + evaluation.")
        acc_train = evaluate_accuracy(X_train, train_labels, w, b)
        acc_test = evaluate_accuracy(X_test, test_labels, w, b)
        print("Train accuracy:", acc_train)
        print("Test accuracy :", acc_test)
        print()

        print("IMPORTANT NOTE:")
        print("- Testing must use the same vocabulary as training.")
        print("- OOV tokens are ignored (zero count).")
        print("- Accuracy is a simple metric. Later we will discuss calibration and confidence.")
        
        precision, recall, f1 = compute_precision_recall(X_test, test_labels, w, b)
        print("Test Precision:", precision)
        print("Test Recall   :", recall)
        print("Test F1       :", f1)


if __name__ == "__main__":
    run()

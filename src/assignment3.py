# -*- coding: utf-8 -*-
# Assignment 3
#
# Task 1: Simple Neural Network for Sentiment Analysis
# Task 2: Tiny Transformer for Sentiment Analysis
#

# Important rules:
# ------------------------------------------------------------
# 1. Start with STAGE = 0.
# 2. Implement only the function(s) required for that stage.
# 3. Run the script and inspect the printed output.
# 4. When the stage works correctly, increase STAGE by 1.
# 5. Do NOT skip stages.
#
# You may ONLY modify:
#   (a) STAGE
#   (b) Functions or classes marked with TODO
#
# You must NOT modify:
#   - Runner
#   - Printing helpers
#   - Test logic
#
# Recommended package:
# ------------------------------------------------------------
# This assignment uses PyTorch for the neural-network parts.
#
# The goal is NOT to learn the full PyTorch ecosystem.
# The goal is to understand the modeling steps:
#   text -> token ids -> embeddings -> pooled/contextualized representation -> classifier

import os
import csv
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Global configuration
# ============================================================
STAGE = 10

DATA_FILENAME = "task2Data.csv"   # same sentiment dataset as Assignment 2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, DATA_FILENAME)

SEED = 42
TRAIN_RATIO = 0.8
MAX_LEN = 24
MIN_FREQ = 1
BATCH_SIZE = 16
EMBED_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 5
LR = 1e-3

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Shared utilities
# ============================================================
def print_header(title: str):
    print("=" * 60)
    print(title)
    print("=" * 60)


def train_test_split(texts, labels, ratio=0.8, seed=42):
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    split = int(ratio * len(idx))
    train_idx = idx[:split]
    test_idx = idx[split:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_texts, train_labels, test_texts, test_labels


def batchify(X, y, batch_size=16, shuffle=True):
    indices = list(range(len(X)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        xb = torch.tensor([X[i] for i in batch_idx], dtype=torch.long)
        yb = torch.tensor([y[i] for i in batch_idx], dtype=torch.float32)
        yield xb, yb


def evaluate_accuracy(model, X, y, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in batchify(X, y, batch_size=batch_size, shuffle=False):
            logits = model(xb).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += len(yb)
    return correct / max(total, 1)


# ============================================================
# Task 1: Simple Neural Network for Sentiment Analysis
# Stages 0-5
# ============================================================

# ------------------------------------------------------------
# Stage 0: Load the same sentiment data as Assignment 2
# ------------------------------------------------------------
def load_supervised_csv(path: str):
    """
    Return:
    - texts: list[str]
    - labels: list[int]

    CSV format expected:
    - header row exists
    - columns include: "text" and "label"
    
    # NOTE:
    # - File reading here is the same as in Assignment 2.
    # - Always cast labels to int explicitly. This is a good habit to avoid downstream bugs caused by inconsistent data types.
    """

    # TODO (Stage 0)
    texts = []
    labels = []
    
    # Open the file and parse lines using csv.DictReader
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(int(row['label']))  # explicitly cast to integer
            
    return texts, labels


# ------------------------------------------------------------
# Stage 1: Tokenization + vocabulary + encoding
#
#   Note:
# - Logistic regression in Assignment 2 used Bag-of-Words tokens(order of the token does not matter).
# - Here, we keep token order in a simple token sequence.
# - Tokens are mapped to integer ids.
# - Vocabulary is built from training data only.

# ------------------------------------------------------------
def tokenize_text(text: str):
    """
    Minimal tokenization:
    - lowercase
    - whitespace split

    Return:
    - list[str]
    """
    # TODO (Stage 1)
    # Lowercase the entire text and then split on whitespaces
    return text.lower().split()
   


def build_vocab(texts, min_freq=1):
    """
    Build token -> id dictionary from training texts only.

    Requirements:
    - Include PAD_TOKEN with index 0
    - Include UNK_TOKEN with index 1
    - Add tokens whose frequency >= min_freq

    Return:
    - vocab: dict[str, int]

        # NOTE:
    # - PAD_TOKEN and UNK_TOKEN are defined as global constants.
    # - You should assign them fixed ids: PAD -> 0, UNK -> 1.
    # - PAD is used for padding shorter sequences to a fixed length.
    # - UNK(unknown) is used to represent out-of-vocabulary (OOV) tokens at test time.

    """
    # Initialize the vocab with predefined IDs for special tokens
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    
    # Count frequency of each token across all texts
    token_counts = Counter()
    for text in texts:
        tokens = tokenize_text(text)
        token_counts.update(tokens)
        
    # Start assigning numerical IDs starting from 2
    current_idx = 2
    for token, freq in token_counts.items():
        if freq >= min_freq:
            vocab[token] = current_idx
            current_idx += 1
            
    return vocab



def encode_text(text: str, vocab: dict):
    """
    Convert one text into a list of token ids.

    out of vocabulary tokens should map to vocab[UNK_TOKEN].
    """
    # TODO (Stage 1)
    tokens = tokenize_text(text)
    unk_id = vocab[UNK_TOKEN]
    
    # Get the ID for each token, defaulting to unk_id if not found in vocab
    return [vocab.get(token, unk_id) for token in tokens]


# ------------------------------------------------------------
# Stage 2: Padding and fixed-length input representation
#
# Concept note:
# - Neural models usually expect tensors with consistent shape.
# - Texts have different lengths.
# - Padding is an engineering step that makes batch computation possible.
# ------------------------------------------------------------
def pad_sequence(token_ids, max_len, pad_idx=0):
    """
    If token_ids is shorter than max_len, pad on the right.
    If longer than max_len, truncate on the right.

    Return:
    - padded_ids: list[int] of length max_len
    """
    # TODO (Stage 2)
    if len(token_ids) >= max_len:
        # Truncate to max_len
        return token_ids[:max_len]
    else:
        # Pad with pad_idx up to max_len
        return token_ids + [pad_idx] * (max_len - len(token_ids))



def prepare_dataset(texts, labels, vocab, max_len):
    """
    Convert a list of texts into padded token-id sequences.
    - For each (text, label) pair:
        1. Use encode_text() to convert the text into token ids.
        2. Use pad_sequence() to make the sequence length = max_len.
        3. Store the padded sequence and its label.

    - Do NOT implement padding logic here. Always call pad_sequence().
    - All sequences must have the same length for batch processing.
    - Do NOT change the order of tokens.
    - PAD tokens are only for alignment, not part of the original text.
    Return:
    - X: list[list[int]]
    - y: list[int]
    """
    X = []
    y = []
    
    pad_idx = vocab[PAD_TOKEN]
    for text, label in zip(texts, labels):
        # 1. Encode into token ids
        token_ids = encode_text(text, vocab)
        
        # 2. Pad to identical lengths
        padded_ids = pad_sequence(token_ids, max_len, pad_idx=pad_idx)
        
        X.append(padded_ids)
        y.append(label)
        
    return X, y


# ------------------------------------------------------------
# Stage 3: Embedding lookup demo
#
# Concept note:
# - Token ids are discrete symbols.
# - Embeddings turn each token id into a learnable dense vector.
# - This is the first step from symbolic input to learned representation.
# ------------------------------------------------------------
class SimpleEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        """
        TODO:
        Step 1:
        - Create an embedding layer using nn.Embedding

        Step 2:
        - Assign it to self.embedding

        IMPORTANT:
        - You MUST write it as self.embedding = ...
        Why:
        - Only layers stored in self.* are tracked by PyTorch
        - Otherwise the model will not learn (parameters won't be updated)

        Required arguments:
        - num_embeddings = vocab_size
        - embedding_dim = embed_dim
        - padding_idx = pad_idx

        What this layer does:
        - Input: token ids (integers)
        - Output: vectors of size embed_dim
        """
        # TODO (Stage 3)
        # PyTorch Embedding layer maps token IDs to dense learnable vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )


    def forward(self, x):
        """
        TODO:
        Step 1:
        - Call the embedding layer on x

        Step 2:
        - Return the result

        IMPORTANT:
        - Use self.embedding(x), not something else

        Input:
        - x: [batch_size, seq_len]

        Output:
        - [batch_size, seq_len, embed_dim]
        """
        # TODO (Stage 3)
        # Compute embeddings for sequence x
        return self.embedding(x)



# ------------------------------------------------------------
# Stage 4: Mean pooling + MLP classifier
#
# Concept note:
# - A sentence is a sequence of token vectors.
# - We need one vector for sentence-level classification.
# - Here we use mean pooling as the simplest aggregation.
# - Then an MLP maps the pooled sentence vector to a sentiment logit.
# ------------------------------------------------------------
class SentimentMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        """
        TODO:
        Step 1:
        - Create a SimpleEmbedder and store it as self.embedder

        Step 2:
        - Create the first linear layer and store it as self.fc1
          This layer should map from embed_dim -> hidden_dim

        Step 3:
        - Create the second linear layer and store it as self.fc2
          This layer should map from hidden_dim -> 1

        IMPORTANT:
        - You MUST store all layers as self.*
        - Do NOT create local variables like fc1 = ... or fc2 = ...

        Why:
        - Only layers stored in self.* are tracked by PyTorch
        - Otherwise their parameters will not be updated during training

        Model structure:
        - token ids
        - embedding
        - mean pooling over sequence length
        - linear layer
        - ReLU
        - linear layer
        - output logit
        """
        super().__init__()
        # TODO (Stage 4)
        # Step 1: Embedder instance mapping IDs to dense representations
        self.embedder = SimpleEmbedder(vocab_size, embed_dim, pad_idx=pad_idx)
        
        # Step 2: Linear layer from embed_dim to hidden_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        
        # Step 3: Linear layer mapping hidden representations to 1 logit output
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        TODO:
        Step 1:
        - Convert token ids into embeddings using self.embedder(x)

        Step 2:
        - Average the embeddings across the sequence dimension
          Hint: use mean(dim=1)
          After this step, the shape should be [batch_size, embed_dim]

        Step 3:
        - Pass the pooled vector through self.fc1

        Step 4:
        - Apply ReLU activation

        Step 5:
        - Pass the result through self.fc2

        Step 6:
        - Return the final logits

        IMPORTANT:
        - Do NOT apply sigmoid here
        - The training function will use BCEWithLogitsLoss, which expects raw logits

        Input:
        - x: [batch_size, seq_len]

        Output:
        - logits: [batch_size, 1]
        """
        # TODO (Stage 4)                                                                            
        # Convert IDs to embeddings
        emb = self.embedder(x)
        
        # Average embeddings across the sequence length (dim=1)
        pooled = emb.mean(dim=1)
        
        # Pass to the first layer, apply ReLU activation
        hidden = F.relu(self.fc1(pooled))
        
        # Final linear layer to output the raw un-normalized logit
        logits = self.fc2(hidden)
        
        return logits


# ------------------------------------------------------------
# Stage 5: Training loop for Task 1
# ------------------------------------------------------------
def train_model(model, X_train, y_train, X_test, y_test, epochs=5, lr=1e-3, batch_size=16):
    """
    Train a binary classifier using BCEWithLogitsLoss.

    TODO:
    Step 1:
    - Create a loss function and store it as criterion
    - Use nn.BCEWithLogitsLoss()

    Step 2:
    - Create an optimizer and store it as optimizer
    - Use torch.optim.Adam(model.parameters(), lr=lr)

    Step 3:
    - Repeat training for the given number of epochs

    Inside each epoch:
    - Switch the model to training mode using model.train()
    - Loop over mini-batches from batchify(...)
    - For each batch:
        1. Clear old gradients using optimizer.zero_grad()
        2. Run the model forward on xb
        3. Remove the last dimension using squeeze(1)
        4. Compute loss between logits and yb
        5. Run backpropagation using loss.backward()
        6. Update parameters using optimizer.step()

    After each epoch:
    - Compute the average training loss
    - Evaluate test accuracy
    - Print both so you can observe whether training is working

    IMPORTANT:
    - Do NOT apply sigmoid before BCEWithLogitsLoss
    - BCEWithLogitsLoss expects raw logits, not probabilities
    - Use model.parameters() in the optimizer so all learnable parameters are updated
    - Call model.train() before training batches
    - The evaluation function will handle model.eval() separately

    Return:
    - trained model
    """
    # TODO (Stage 5)
    # Use BCEWithLogitsLoss because the model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer updates the model's learnable parameters using backpropagation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Switch model to training mode (enables gradient tracking)
        model.train()
        
        total_loss = 0.0
        batches = 0
        
        for xb, yb in batchify(X_train, y_train, batch_size=batch_size, shuffle=True):
            # 1. Clear previous gradients
            optimizer.zero_grad()
            
            # 2. Forward pass and remove the singleton dimension
            logits = model(xb).squeeze(1)
            
            # 3. Compute loss
            loss = criterion(logits, yb)
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / max(batches, 1)
        
        # Evaluate model after epoch
        test_acc = evaluate_accuracy(model, X_test, y_test)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")
        
    return model



# ============================================================
# Task 2: Tiny Transformer for Sentiment Analysis
# Stages 6-10
# ============================================================

# ------------------------------------------------------------
# Stage 6: Reuse the same encoded and padded data
#
# Concept note:
# - We keep the task and preprocessing pipeline fixed.
# - Only the model architecture changes.
# - This makes the comparison with Task 1 easier to interpret.
# ------------------------------------------------------------
def build_position_ids(batch_size, seq_len):
    """
    IMPORTANT:
    - Each row should be identical
      Example (seq_len = 5):
        [0, 1, 2, 3, 4]

    Why we need this:
    - The model currently only sees token ids
    - It does NOT know the order of tokens
    - Position ids will later be converted into position embeddings
      and added to token embeddings
    # - This function creates position indices (0,1,2,...).
    # - These indices themselves are NOT learnable.
    # - They will later be passed into an embedding layer to produce
    #   learnable position vectors.
    # - This is different from sinusoidal positional encoding,
    #   where position vectors are fixed (not learned).
    # - Here: index -> embedding -> learned position representation.

    Hint:
    - You can use list(range(seq_len))
    - Then replicate it using a loop or list multiplication
    - Finally convert using torch.tensor(...)
    """
    # TODO (Stage 6)
     # Create one sequence of range seq_len
    pos_ids = list(range(seq_len))
    
    # Replicate it for the whole batch
    batch_pos_ids = [pos_ids for _ in range(batch_size)]
    
    # Convert and return as a PyTorch tensor
    return torch.tensor(batch_pos_ids, dtype=torch.long)


# ------------------------------------------------------------
# Stage 7: Token embedding + positional embedding
#
# Concept note:
# - Self-attention alone does not know token order.
# - Positional information is added so the model can distinguish
#   between early and late token positions.
# ------------------------------------------------------------
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, pad_idx=0):
        super().__init__()
        """
        TODO:
        Step 1:
        - Create a token embedding layer and store it as self.token_embedding
        - Use nn.Embedding(...)
        - This layer maps each token id to a learnable word vector

        Step 2:
        - Create a position embedding layer and store it as self.position_embedding
        - Use nn.Embedding(...)
        - This layer maps each position id (0,1,2,...) to a learnable position vector

        IMPORTANT:
        - You MUST store both layers as self.*
        - Do NOT create local variables like token_embedding = ... or position_embedding = ...
        - Only layers stored in self.* will be tracked and updated by PyTorch

        Required arguments for token embedding:
        - num_embeddings = vocab_size
        - embedding_dim = embed_dim
        - padding_idx = pad_idx

        Required arguments for position embedding:
        - num_embeddings = max_len
        - embedding_dim = embed_dim

        Why max_len for position embedding:
        - Position ids range from 0 to max_len - 1
        - So we need one learnable vector for each possible position

        Concept:
        - token embedding tells the model what the word is
        - position embedding tells the model where the word is
        - later we add them together to get the final input representation
        """
        # TODO (Stage 7)
         # Token Embeddings layer (maps identity to vector)
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim, 
            padding_idx=pad_idx
        )

        # Position Embedding layer (maps positional index to vector)
        self.position_embedding = nn.Embedding(
            num_embeddings=max_len, 
            embedding_dim=embed_dim
        )
    def forward(self, x):
        """
        TODO:
        Step 1:
        - Get the batch size and sequence length from x

        Step 2:
        - Build position ids by calling build_position_ids(batch_size, seq_len)

        Step 3:
        - Pass x into self.token_embedding to get token embeddings

        Step 4:
        - Pass the position ids into self.position_embedding to get position embeddings

        Step 5:
        - Add the two embedding tensors together

        Step 6:
        - Return the final result

        IMPORTANT:
        - x contains token ids, so use self.token_embedding(x)
        - position_ids contains position indices, so use self.position_embedding(position_ids)
        - The two tensors must have the same shape before addition
        - Addition works because both should have shape [batch_size, seq_len, embed_dim]

        Input:
        - x: [batch_size, seq_len]

        Output:
        - h: [batch_size, seq_len, embed_dim]

        Concept:
        - After this step, each token representation contains both:
            1. token identity
            2. position information
        """
        # TODO (Stage 7)
        batch_size, seq_len = x.shape
        
        # Get position indices. Same device needed if using GPUs. 
        # (Though not required per instructions, best practice is to cast to x.device)
        pos_ids = build_position_ids(batch_size, seq_len).to(x.device)
        
        # Extract embeddings
        t_emb = self.token_embedding(x)
        p_emb = self.position_embedding(pos_ids)
        
        # Return summed representations indicating what & where the token is
        return t_emb + p_emb

# ------------------------------------------------------------
# Stage 8: Single-head self-attention
#
# Concept note:
# - Each token produces a query, key, and value vector.
# - Attention weights tell us how much one token attends to others.
# - This allows context-dependent interaction inside the sentence.
# ------------------------------------------------------------
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # TODO (Stage 8)
        """
        TODO:
        Step 1:
        - Create a linear layer for queries and store it as self.W_q
        - Use nn.Linear(embed_dim, embed_dim)

        Step 2:
        - Create a linear layer for keys and store it as self.W_k
        - Use nn.Linear(embed_dim, embed_dim)

        Step 3:
        - Create a linear layer for values and store it as self.W_v
        - Use nn.Linear(embed_dim, embed_dim)

        IMPORTANT:
        - You MUST store all three layers as self.*
        - Do NOT create local variables like W_q = ... or W_k = ...
        - Only layers stored in self.* will be tracked and updated by PyTorch

        What these layers do:
        - self.W_q maps each token vector to a query vector
        - self.W_k maps each token vector to a key vector
        - self.W_v maps each token vector to a value vector

        Why we need three different projections:
        - Query: what this token is looking for
        - Key: what this token offers to other tokens
        - Value: the actual information passed forward after attention
        """
        # Linear projections mapping combined embedded vector -> Query, Key, and Value
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        TODO:
        Step 1:
        - Pass x through self.W_q to get Q
        - Pass x through self.W_k to get K
        - Pass x through self.W_v to get V

        Step 2:
        - Compute attention scores using matrix multiplication:
              scores = Q @ K^T
        - You need to transpose the last two dimensions of K
        - Use K.transpose(1, 2)

        Step 3:
        - Scale the scores by sqrt(embed_dim)
        - This keeps the values from becoming too large
        - Hint: use math.sqrt(x.size(-1))

        Step 4:
        - Apply softmax over the last dimension to get attention weights
        - Use F.softmax(scores, dim=-1)

        Step 5:
        - Multiply attention weights by V to get the final output

        Step 6:
        - Return both:
            1. output
            2. attention weights

        IMPORTANT:
        - x has shape [batch_size, seq_len, embed_dim]
        - Q, K, V should all have shape [batch_size, seq_len, embed_dim]
        - scores should have shape [batch_size, seq_len, seq_len]
        - attention weights should also have shape [batch_size, seq_len, seq_len]
        - output should have shape [batch_size, seq_len, embed_dim]

        What the score matrix means:
        - scores[i, j] measures how much token i attends to token j
        - each row corresponds to one token looking at all tokens in the sequence

        Why softmax is applied:
        - It turns raw scores into normalized weights
        - Each row will sum to 1
        - This lets the model compute a weighted combination of all value vectors
        """
        # TODO (Stage 8)
        # 1. Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2 & 3. Compute attention scores: Q @ K.T scaled by sqrt of the embedding dim
        # Transpose K's last two dimensions: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        K_t = K.transpose(1, 2)
        scores = torch.matmul(Q, K_t) / math.sqrt(x.size(-1))
        
        # 4. Turn raw scores into probabilities
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Multiply attention probabilities with V
        output = torch.matmul(attn_weights, V)
        
        # Return output alongside attention weights for later inspection
        return output, attn_weights


# ------------------------------------------------------------
# Stage 9: Tiny Transformer block + classifier
#
# Concept note:
# - A transformer block usually contains:
#   self-attention -> residual -> normalization -> feed-forward
# - We keep the block intentionally small.
# - The final sentence representation is obtained by mean pooling.
# ------------------------------------------------------------
class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len, pad_idx=0):
        super().__init__()
        """
        This module implements a simplified Transformer-based classifier.
        The model follows a standard Transformer block structure:
        1. Token embedding + positional embedding
        2. Self-attention to allow interaction between tokens
        3. Residual connection + LayerNorm
        4. Feed-forward network applied to each token
        5. Another residual connection + LayerNorm
        6. Mean pooling to obtain a sentence-level representation
        7. Final linear layer for classification

        You need to construct the following components:

        - self.embedding:
          A TransformerEmbedding module that converts token ids into
          vectors that include both token identity and position information.

        - self.attention:
          A SingleHeadSelfAttention module that enables each token to
          attend to all other tokens in the sequence.

        - self.norm1:
          A LayerNorm applied after the attention block.

        - self.ff1 and self.ff2:
          A two-layer feed-forward network:
              embed_dim → hidden_dim → embed_dim

        - self.norm2:
          A second LayerNorm applied after the feed-forward block.

        - self.classifier:
          A final linear layer mapping from embed_dim → 1

        IMPORTANT:
        - All components must be stored as self.*
        - Do NOT create layers inside forward()
        - Otherwise parameters will not be tracked or updated
        """
        # TODO (Stage 9)
        # Data preparation mapping
        self.embedding = TransformerEmbedding(vocab_size, embed_dim, max_len, pad_idx)
        
        # The self-attention block
        self.attention = SingleHeadSelfAttention(embed_dim)
        
        # First LayerNorm component
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # The Feed-forward network structure
        self.ff1 = nn.Linear(embed_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, embed_dim)
        
        # Second LayerNorm component
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Output classifier block mapping vector to 1 target class
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, return_attention=False):
        """
        The forward pass follows the Transformer computation pipeline.

        First, token ids are converted into embeddings that contain both
        word identity and position information.

        Then, self-attention is applied so that each token representation
        becomes context-dependent, incorporating information from other tokens.

        A residual connection adds the original representation back to the
        attention output, followed by LayerNorm for stabilization.

        Next, a feed-forward network is applied independently to each token,
        followed again by a residual connection and LayerNorm.

        After these transformations, we still have token-level representations.
        To perform sentence classification, we apply mean pooling across the
        sequence dimension to obtain a single vector per example.

        Finally, this vector is passed through a linear layer to produce a logit.

        IMPORTANT:
        - Do NOT apply sigmoid here.
        - BCEWithLogitsLoss expects raw logits.

        NOTE:
        - This simplified implementation does NOT mask PAD tokens.
        - Real Transformer models usually include an attention mask.

        Input:
        - x: [batch_size, seq_len]

        Output:
        - logits: [batch_size, 1]
        - optionally attention weights: [batch_size, seq_len, seq_len]
        """
        # TODO (Stage 9)
        # Produce contextual embeddings
        h = self.embedding(x)
        
        # Process self-attention mechanism capturing word relations
        attn_out, attn_weights = self.attention(h)
        
        # Residual connection + LayerNorm1 
        h = self.norm1(h + attn_out)
        
        # Apply Feed Forward logic to tokens
        ff_out = self.ff2(F.relu(self.ff1(h)))
        
        # Residual connection + LayerNorm2
        h = self.norm2(h + ff_out)
        
        # Mean pooling to single sentence-level representation
        sentence_rep = h.mean(dim=1)
        
        # Classify pooled presentation
        logits = self.classifier(sentence_rep)
        
        if return_attention:
            return logits, attn_weights
            
        return logits


# ------------------------------------------------------------
# Stage 10: Train and inspect attention
# ------------------------------------------------------------
def inspect_attention(model, x_example):
    """
    Inspect attention behavior of the trained transformer.

    In previous stages, we focused on building and training the model.
    Here, we move from "using the model" to "understanding the model".

    This function runs a single example through the model and extracts
    the attention weights produced by the self-attention layer.

    Each attention matrix tells us:
    - For each token (row), how much it attends to every other token (columns)

    This allows us to analyze questions such as:
    - Which words influence each other?
    - Does the model focus on meaningful parts of the sentence?

    Implementation details:
    - We call the model with return_attention=True
    - We use model.eval() and torch.no_grad() since no training is needed

    Input:
    - x_example: tensor of shape [1, seq_len]

    Return:
    - attn_weights: [1, seq_len, seq_len]
    
    Once you implement this function you can use it to understand the attention behavior of the model.
    For example, you can pick one sentence and inspect its attention weights and ask:
    Which words does the model focus on?
    Do these attention patterns seem meaningful?
    """
     # Toggle model to evaluation mode
    model.eval()
    
    # Disable gradient tracking as model is predicting, not training
    with torch.no_grad():
        # Get logits and the returned attention matrix weights
        logits, attn_weights = model(x_example, return_attention=True)
        
    return attn_weights


# ============================================================
# Runner (DO NOT MODIFY)
# ============================================================
def run():
    print("STAGE =", STAGE)
    print()

    texts = None
    labels = None
    train_texts = None
    train_labels = None
    test_texts = None
    test_labels = None
    vocab = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if STAGE >= 0:
        print_header("Stage 0: Load supervised sentiment data")
        texts, labels = load_supervised_csv(DATA_PATH)
        print("Total examples:", len(texts))
        print("First example label:", labels[0])
        print("First example text:", texts[0][:150])
        print()

        train_texts, train_labels, test_texts, test_labels = train_test_split(
            texts, labels, ratio=TRAIN_RATIO, seed=SEED
        )
        print("Train size:", len(train_texts))
        print("Test size :", len(test_texts))
        print()

    if STAGE >= 1:
        print_header("Stage 1: Tokenization + vocabulary + encoding")
        vocab = build_vocab(train_texts, min_freq=MIN_FREQ)
        print("Vocabulary size:", len(vocab))
        print("PAD index:", vocab[PAD_TOKEN])
        print("UNK index:", vocab[UNK_TOKEN])
        print()

        sample_tokens = tokenize_text(train_texts[0])
        sample_ids = encode_text(train_texts[0], vocab)
        print("Sample tokens:", sample_tokens[:20])
        print("Sample token ids:", sample_ids[:20])
        print()

    if STAGE >= 2:
        print_header("Stage 2: Padding and dataset preparation")
        X_train, y_train = prepare_dataset(train_texts, train_labels, vocab, MAX_LEN)
        X_test, y_test = prepare_dataset(test_texts, test_labels, vocab, MAX_LEN)

        print("Train examples:", len(X_train))
        print("Test examples :", len(X_test))
        print("Sequence length:", len(X_train[0]))
        print("First padded sequence:", X_train[0])
        print()

    if STAGE >= 3:
        print_header("Stage 3: Embedding lookup demo")
        embedder = SimpleEmbedder(len(vocab), EMBED_DIM, pad_idx=vocab[PAD_TOKEN])
        xb = torch.tensor(X_train[:2], dtype=torch.long)
        h = embedder(xb)
        print("Input shape:", tuple(xb.shape))
        print("Embedding output shape:", tuple(h.shape))
        print("First token embedding (first 8 dims):")
        print(h[0, 0, :8])
        print()

    if STAGE >= 4:
        print_header("Stage 4: Mean pooling + MLP classifier")
        model = SentimentMLP(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            pad_idx=vocab[PAD_TOKEN],
        )
        xb = torch.tensor(X_train[:4], dtype=torch.long)
        logits = model(xb)
        print("Input shape:", tuple(xb.shape))
        print("Logits shape:", tuple(logits.shape))
        print("Sample logits:")
        print(logits.squeeze(1))
        print()

    if STAGE >= 5:
        print_header("Stage 5: Train Task 1 model")
        model = SentimentMLP(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            pad_idx=vocab[PAD_TOKEN],
        )
        model = train_model(model, X_train, y_train, X_test, y_test,
                            epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)
        acc_train = evaluate_accuracy(model, X_train, y_train)
        acc_test = evaluate_accuracy(model, X_test, y_test)
        print("Final Task 1 train accuracy:", acc_train)
        print("Final Task 1 test accuracy :", acc_test)
        print()

    if STAGE >= 6:
        print_header("Stage 6: Position ids for transformer input")
        xb = torch.tensor(X_train[:2], dtype=torch.long)
        pos_ids = build_position_ids(batch_size=xb.size(0), seq_len=xb.size(1))
        print("Input shape:", tuple(xb.shape))
        print("Position id shape:", tuple(pos_ids.shape))
        print("First row of position ids:", pos_ids[0])
        print()

    if STAGE >= 7:
        print_header("Stage 7: Token embedding + positional embedding")
        transformer_embed = TransformerEmbedding(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            max_len=MAX_LEN,
            pad_idx=vocab[PAD_TOKEN],
        )
        xb = torch.tensor(X_train[:2], dtype=torch.long)
        h = transformer_embed(xb)
        print("Input shape:", tuple(xb.shape))
        print("Embedded shape:", tuple(h.shape))
        print("Example embedded vector (first token, first 8 dims):")
        print(h[0, 0, :8])
        print()

    if STAGE >= 8:
        print_header("Stage 8: Single-head self-attention")
        transformer_embed = TransformerEmbedding(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            max_len=MAX_LEN,
            pad_idx=vocab[PAD_TOKEN],
        )
        attention = SingleHeadSelfAttention(EMBED_DIM)
        xb = torch.tensor(X_train[:2], dtype=torch.long)
        h = transformer_embed(xb)
        out, attn = attention(h)
        print("Input hidden shape:", tuple(h.shape))
        print("Attention output shape:", tuple(out.shape))
        print("Attention weight shape:", tuple(attn.shape))
        print("Attention weights for sample 0, token 0, first 8 positions:")
        print(attn[0, 0, :8])
        print()

    if STAGE >= 9:
        print_header("Stage 9: Tiny transformer classifier")
        transformer_model = TinyTransformerClassifier(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            max_len=MAX_LEN,
            pad_idx=vocab[PAD_TOKEN],
        )
        xb = torch.tensor(X_train[:4], dtype=torch.long)
        logits = transformer_model(xb)
        print("Input shape:", tuple(xb.shape))
        print("Logits shape:", tuple(logits.shape))
        print("Sample logits:")
        print(logits.squeeze(1))
        print()

    if STAGE >= 10:
        print_header("Stage 10: Train tiny transformer + inspect attention")
        transformer_model = TinyTransformerClassifier(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            max_len=MAX_LEN,
            pad_idx=vocab[PAD_TOKEN],
        )
        transformer_model = train_model(
            transformer_model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
        )
        acc_train = evaluate_accuracy(transformer_model, X_train, y_train)
        acc_test = evaluate_accuracy(transformer_model, X_test, y_test)
        print("Final Task 2 train accuracy:", acc_train)
        print("Final Task 2 test accuracy :", acc_test)
        print()

        x_example = torch.tensor([X_test[0]], dtype=torch.long)
        attn = inspect_attention(transformer_model, x_example)
        print("Attention shape:", tuple(attn.shape))
        print("Attention row for token 0 (first 10 positions):")
        print(attn[0, 0, :10])
        print()


if __name__ == "__main__":
    run()

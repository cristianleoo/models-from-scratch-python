import numpy as np

class MultiHeadAttention:
    """
    Multi-head attention.

    Parameters:
    num_hiddens: int
        Number of hidden units.
    num_heads: int
        Number of attention heads.
    dropout: float
        Dropout rate.
    bias: bool
        Whether to include bias parameters in the model.
    """
    def __init__(self, num_hiddens, num_heads, dropout=0.0, bias=False):
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.d_k = self.d_v = num_hiddens // num_heads
        
        self.W_q = np.random.rand(num_hiddens, num_hiddens)
        self.W_k = np.random.rand(num_hiddens, num_hiddens)
        self.W_v = np.random.rand(num_hiddens, num_hiddens)
        self.W_o = np.random.rand(num_hiddens, num_hiddens)
        
        if bias:
            self.b_q = np.random.rand(num_hiddens)
            self.b_k = np.random.rand(num_hiddens)
            self.b_v = np.random.rand(num_hiddens)
            self.b_o = np.random.rand(num_hiddens)
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = np.zeros(num_hiddens)

    def transpose_qkv(self, X):
        """
        Transposition for batch processing
        
        Parameters:
        X: np.ndarray
            Input tensor

        Returns:
        np
            Transposed tensor
        """
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.transpose(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def transpose_output(self, X):
        """
        Transposition for output

        Parameters:
        X: np.ndarray
            Input tensor
        
        Returns:
        np
            Transposed tensor
        """
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.transpose(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    
    def scaled_dot_product_attention(self, Q, K, V, valid_lens):
        """
        Scaled dot product attention

        Parameters:
        Q: np.ndarray
            Query tensor
        K: np.ndarray
            Key tensor
        V: np.ndarray
            Value tensor
        valid_lens: np.ndarray
            Valid lengths for the query
        
        Returns:
        np
            Output tensor
        """
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
        if valid_lens is not None:
            mask = np.arange(scores.shape[-1]) < valid_lens[:, None]
            scores = np.where(mask[:, None, :], scores, -np.inf)
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights /= attention_weights.sum(axis=-1, keepdims=True)
        return np.matmul(attention_weights, V)
    
    def forward(self, queries, keys, values, valid_lens):
        """
        Forward pass

        Parameters:
        queries: np.ndarray
            Query tensor
        keys: np.ndarray
            Key tensor
        values: np.ndarray
            Value tensor
        valid_lens: np.ndarray
            Valid lengths for the query

        Returns:
        np
            Output tensor
        """
        queries = self.transpose_qkv(np.dot(queries, self.W_q) + self.b_q)
        keys = self.transpose_qkv(np.dot(keys, self.W_k) + self.b_k)
        values = self.transpose_qkv(np.dot(values, self.W_v) + self.b_v)
        
        if valid_lens is not None:
            valid_lens = np.repeat(valid_lens, self.num_heads, axis=0)
        
        output = self.scaled_dot_product_attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)
        return np.dot(output_concat, self.W_o) + self.b_o

# Define dimensions and initialize multi-head attention
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, dropout=0.5, bias=False)

# Define sample data
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = np.array([3, 2])
X = np.random.rand(batch_size, num_queries, num_hiddens)  # Use random data to simulate input queries
Y = np.random.rand(batch_size, num_kvpairs, num_hiddens)  # Use random data to simulate key-value pairs

# Apply multi-head attention
output = attention.forward(X, Y, Y, valid_lens)
print("Output shape:", output.shape)  # Expected shape: (batch_size, num_queries, num_hiddens)

# Display the output for inspection
print("Output data:", output)
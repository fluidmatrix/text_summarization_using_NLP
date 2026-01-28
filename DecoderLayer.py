import tensorflow as tf
from helper import FullyConnected

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.ffn = FullyConnected(
            embedding_dim=embedding_dim,
            fully_connected_dim=fully_connected_dim
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        batch_size = tf.shape(x)[0]
        target_seq_len = tf.shape(x)[1]
        input_seq_len = tf.shape(enc_output)[1]
        num_heads = self.mha1.num_heads

        # --- BLOCK 1: Self-Attention (with look-ahead mask) ---
        if look_ahead_mask is not None:
            look_ahead_mask = tf.cast(look_ahead_mask, tf.float32)
            if len(look_ahead_mask.shape) == 3:  # (1, target_seq_len, target_seq_len)
                look_ahead_mask = tf.expand_dims(look_ahead_mask, axis=0)
            look_ahead_mask = tf.tile(look_ahead_mask, [batch_size, 1, 1, 1])  # match batch
            look_ahead_mask = look_ahead_mask[:, :num_heads, :, :] if look_ahead_mask.shape[1] == num_heads else look_ahead_mask

        attn1, attn_weights_block1 = self.mha1(
            query=x,
            value=x,
            key=x,
            attention_mask=look_ahead_mask,
            training=training,
            return_attention_scores=True
        )
        x1 = self.layernorm1(x + attn1)

        # --- BLOCK 2: Encoder-Decoder Attention (with padding mask) ---
        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask, tf.float32)
            if len(padding_mask.shape) == 3:  # (batch_size, 1, input_seq_len)
                padding_mask = tf.expand_dims(padding_mask, axis=1)
            padding_mask = tf.tile(padding_mask, [1, num_heads, target_seq_len, 1])  # match shape

        attn2, attn_weights_block2 = self.mha2(
            query=x1,
            value=enc_output,
            key=enc_output,
            attention_mask=padding_mask,
            training=training,
            return_attention_scores=True
        )
        x2 = self.layernorm2(x1 + attn2)

        # --- BLOCK 3: Feed-Forward Network ---
        ffn_output = self.ffn(x2)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out3 = self.layernorm3(x2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2

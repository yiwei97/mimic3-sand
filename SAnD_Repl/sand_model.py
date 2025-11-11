import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_window,
        dropout_rate
    ):
        super(MaskedMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_window = attn_window
        self.dropout_rate = dropout_rate
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.dropout = nn.Dropout(dropout_rate)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # shape x: (BS, Seq Len, d_model) 
        bs, seq_len, d_model = x.shape
        
        # Pass x into 3 Linear Layers (In here we combine it into 1) and then split each qkv into (BS, Seq Len, Num Heads, Head Dim)
        qkv = self.qkv_proj(x).view(bs, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # b = batch size
        # t, T = seq len
        # h = head
        # d = head dim
        # For each batch and head calculate do product between seq_len, head_dim to get (Batch, Head, Seq Len, Seq Len)
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale

        # Create a non-look ahead mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # We do this part specified in the paper:
        # Additionally, we mask the sequence to specify how far the attention models 
        # can look into the past for obtaining the representation for each position. 
        if self.attn_window is not None:
            local_mask = torch.zeros(seq_len, seq_len, device=x.device)
            for i in range(seq_len):
                local_mask[i, max(0, i - self.attn_window):i + 1] = 1
            mask = causal_mask * local_mask
        else:
            mask = causal_mask

        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # Calculate the softmax(q*k.T / scale) * v
        out = torch.einsum("bhts,bshd->bthd", attn_weights, v)
        out = out.reshape(bs, seq_len, d_model)

        # pass through another linear layer for final projection
        out = self.dropout(self.out_proj(out))
        return out

class SAnDAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_window,
        dropout_rate
    ):
        super(SAnDAttention, self).__init__()

        self.masked_attn = MaskedMultiHeadAttention(d_model, num_heads, attn_window, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.masked_linear = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.ReLU(),
            nn.Conv1d(d_model * 2, d_model, 1),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.norm1(x + self.masked_attn(x))
        x = self.norm2(x + self.masked_linear(x))
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        input_features,
        seq_len,
        num_heads,
        n_layers,
        d_model,
        dropout_rate,
        attn_window
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.attention_layers = nn.ModuleList([
            SAnDAttention(d_model, num_heads, attn_window, dropout_rate) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        bs, seq_len, _ = x.shape

        # turn (BS, seq_len, emb_dim) -> (BS, seq_len, d_model)
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embed = self.position_embedding(positions) # Shape (1, seq_len, d_model)
        x = self.dropout(x + pos_embed)

        for layer in self.attention_layers:
            x = layer(x)

        return x
        

class DenseInterpolation(nn.Module):
    def __init__(
        self,
        seq_len,
        factor
    ):
        super(DenseInterpolation, self).__init__()
        W = np.zeros((factor, seq_len), dtype=np.float32)
        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1+m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x):
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)

        # Example: (BS, 2, seq_len) * (BS, seq_len, d_model)
        # Output: (BS, 2, d_model)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2) # (BS, d_model, 2)

class ClassificationHead(nn.Module):
    def __init__(
        self,
        d_model,
        factor,
        num_class
    ):
        super(ClassificationHead, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.fc = nn.Linear(int(d_model * factor), num_class)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, x):
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x

class RegressionHead(nn.Module):
    def __init__(
        self,
        d_model,
        factor
    ):
        self.d_model = d_model
        self.factor = factor
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(int(d_model * factor), 1)
    
    def forward(self, x):
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x


class SAnD(nn.Module):

    def __init__(
        self,
        input_features,
        seq_len,
        num_heads,
        factor,
        n_layers,
        d_model,
        dropout_rate,
        n_class,
        attn_window,
        mode='classification'
    ):
        super(SAnD, self).__init__()

        self.input_features = input_features
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.factor = factor
        self.n_layers = n_layers
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.mode = mode
        self.attn_window = attn_window

        assert mode in ['classification', 'regression', 'multiclass'], "Wrong mode should only be classification, regression or multiclass"

        self.encoder_layer = Encoder(
            input_features,
            seq_len,
            num_heads,
            n_layers,
            d_model,
            dropout_rate,
            attn_window
        )
        self.dense_interpolation_layer = DenseInterpolation(
            seq_len,
            factor
        )

        if self.mode == "classification" or self.mode == "multiclass":
            self.output_layer = ClassificationHead(d_model, factor, n_class)
        else:
            self.output_layer = RegressionHead(d_model, factor)
    
    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.dense_interpolation_layer(x)
        x = self.output_layer(x)

        return x



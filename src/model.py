import torch
import torch.nn as nn

class GeneTransformerMultiTask(nn.Module):
    def __init__(self, n_genes, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1, latent_dim=32):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.latent_dim = latent_dim

        # ----- embedding -----
        self.gene_value_proj = nn.Linear(1, d_model)
        self.gene_id_embed = nn.Embedding(n_genes, d_model)

        # ----- transformer encoder -----
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # ----- pooling -----
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ----- shared fully connected -----
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ----- heads -----
        self.head_3  = nn.Linear(d_model, 3)    # regression: 3 scores
        self.head_cls = nn.Linear(d_model, 1)   # classification
        self.head_act = nn.Linear(d_model, 1)   # scalar activation score

        # ----- multi-D latent
        self.head_latent = nn.Linear(d_model, latent_dim)


        # gene index buffer
        self.register_buffer("gene_indices",
                             torch.arange(n_genes, dtype=torch.long).unsqueeze(0))

    def forward(self, x):
        B, G = x.shape

        # embedding
        v = self.gene_value_proj(x.unsqueeze(-1))
        e = self.gene_id_embed(self.gene_indices.repeat(B, 1))
        tokens = v + e

        # transformer encoder
        h = self.encoder(tokens)
        # pooling
        h = self.pool(h.transpose(1,2)).squeeze(-1)
        # shared fc
        h_fc = self.fc(h)

        # outputs
        scores3 = self.head_3(h_fc)
        logits  = self.head_cls(h_fc).squeeze(-1)
        act     = self.head_act(h_fc).squeeze(-1)
        latent  = self.head_latent(h_fc)   # shape (B, latent_dim)

        return scores3, logits, act, latent

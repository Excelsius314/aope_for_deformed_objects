import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class CrossModalAttention(nn.Module):
    def __init__(self, d_model_b1, d_model_b2, head_dim_b1, head_dim_b2):  # num_heads):
        super().__init__()
        self.d_model_b1 = d_model_b1
        self.d_model_b2 = d_model_b2

        # self.num_heads = num_heads
        # self.head_dim_b1 = d_model_b1 // num_heads
        # self.head_dim_b2 = d_model_b2 // num_heads

        self.head_dim_b1 = head_dim_b1
        self.head_dim_b2 = head_dim_b2

        # assert (
        #    self.head_dim_b1 * num_heads == d_model_b1
        # ), "d_model_b1 must be divisible by num_heads"
        # assert (
        #    self.head_dim_b2 * num_heads == d_model_b2
        # ), "d_model_b2 must be divisible by num_heads"

        self.W_q_b1 = nn.Linear(d_model_b1, d_model_b2)
        self.W_q_b2 = nn.Linear(d_model_b2, d_model_b1)

        self.W_k_b1 = nn.Linear(d_model_b1, d_model_b1)
        self.W_k_b2 = nn.Linear(d_model_b2, d_model_b2)

        self.W_v_b1 = nn.Linear(d_model_b1, d_model_b1)
        self.W_v_b2 = nn.Linear(d_model_b2, d_model_b2)

        self.W_o_b1 = nn.Linear(d_model_b1, d_model_b1)
        self.W_o_b2 = nn.Linear(d_model_b2, d_model_b2)

    def scaled_dot_product_attention(self, Q, K, V, head_dim, mask=None):
        # Q: [batch_size, num_heads, seq_len, head_dim]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q_b1, Q_b2, K_b1, K_b2, V_b1, V_b2, mask=None):
        batch_size = Q_b1.size(0)

        assert Q_b1.size(0) == Q_b2.size(0)

        Q_b1 = self.W_q_b1(Q_b1).view(batch_size, -1, self.head_dim_b2).transpose(1, 2)
        Q_b2 = self.W_q_b2(Q_b2).view(batch_size, -1, self.head_dim_b1).transpose(1, 2)

        K_b1 = self.W_k_b1(K_b1).view(batch_size, -1, self.head_dim_b1).transpose(1, 2)
        K_b2 = self.W_k_b2(K_b2).view(batch_size, -1, self.head_dim_b2).transpose(1, 2)

        V_b1 = self.W_v_b1(V_b1).view(batch_size, -1, self.head_dim_b1).transpose(1, 2)
        V_b2 = self.W_v_b2(V_b2).view(batch_size, -1, self.head_dim_b2).transpose(1, 2)

        # Interchanges Queries between branches
        attn_output_b1 = self.scaled_dot_product_attention(
            Q_b2, K_b1, V_b1, self.head_dim_b1, mask
        )
        attn_output_b2 = self.scaled_dot_product_attention(
            Q_b1, K_b2, V_b2, self.head_dim_b2, mask
        )

        attn_output_b1 = (
            attn_output_b1.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model_b1)
        )
        attn_output_b2 = (
            attn_output_b2.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model_b2)
        )

        output_b1 = self.W_o_b1(attn_output_b1)
        output_b2 = self.W_o_b2(attn_output_b2)

        return output_b1, output_b2


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q: [batch_size, num_heads, seq_len, head_dim]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = (
            self.W_q(Q)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.W_k(K)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.W_v(V)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.W_o(attn_output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Self attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class CrossModalDecoderLayer(nn.Module):
    def __init__(
        self, d_model_b1, d_model_b2, num_heads_b1, num_heads_b2, num_heads_att
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model_b1, num_heads_b1)

        self.ffn = FeedForward(d_model_b1)
        self.norm1 = nn.LayerNorm(d_model_b1)
        self.norm2 = nn.LayerNorm(d_model_b1)
        self.norm3 = nn.LayerNorm(d_model_b1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Self attention
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross attention
        attn_output_b1, attn_output_b2 = self.cross_modal_attn(
            x_b1, x_b2, x_b1, x_b2, x_b1, x_b2
        )
        x_b1 = self.norm2_b1(x_b1 + self.dropout_b1(attn_output_b1))
        x_b2 = self.norm2_b2(x_b2 + self.dropout_b2(attn_output_b2))

        # Feed forward
        ffn_output_b1 = self.ffn_b1(x_b1)
        x_b1 = self.norm3_b1(x_b1 + self.dropout_b1(ffn_output_b1))

        ffn_output_b2 = self.ffn_b2(x_b2)
        x_b2 = self.norm3_b2(x_b2 + self.dropout_b2(ffn_output_b2))
        return x_b1, x_b2


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        num_layers=6,
    ):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.encoder_embedding(src))
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc_out(dec_output)
        return output


class LearnableUVEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, uv):
        # uv normalized to [0,1]
        return self.mlp(uv)


class LearnableXYZEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, xyz):
        return self.mlp(xyz)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, H, W, patch_size, normalize=True, device="cuda:0"):
        super().__init__()

        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.normalize = normalize
        self.device = device

        self.uv_embed = LearnableUVEmbedding(dim).to(device)
        self.xyz_embed = LearnableXYZEmbedding(dim).to(device)

        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
        ).to(device)

    def create_uv_grid(self, batch_size):
        """
        Returns:
            uv: (B, N, 2)
        """
        assert self.H % self.patch_size == 0 and self.W % self.patch_size == 0

        H_p = self.H // self.patch_size
        W_p = self.W // self.patch_size

        # Patch indices
        y = torch.arange(H_p, device=self.device)
        x = torch.arange(W_p, device=self.device)

        # Meshgrid (v = y, u = x)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H_p, W_p)

        # Convert to pixel centers
        u = (xx + 0.5) * self.patch_size
        v = (yy + 0.5) * self.patch_size

        uv = torch.stack([u, v], dim=-1)  # (H_p, W_p, 2)

        # Flatten to tokens
        uv = uv.view(-1, 2)  # (N, 2)

        # Normalize if needed
        if self.normalize:
            uv[..., 0] /= self.W
            uv[..., 1] /= self.H

        # Add batch dimension
        uv = uv.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, N, 2)

        return uv

    def compute_point_uv(self, batch_size, point_pixel_coords):
        # point_uv = (point_pixel_coords.round() // self.patch_size).view(
        #    batch_size, -1, 2
        # )  # (B, N, 2)

        patch_coords = point_pixel_coords.to(torch.float32)

        if self.normalize:
            patch_coords[:, :, 1] = patch_coords[:, :, 1] / (self.H // self.patch_size)
            patch_coords[:, :, 0] = patch_coords[:, :, 0] / (self.W // self.patch_size)

        return patch_coords

    def forward(self, batch_size, point_pixel_coords=None, xyz=None):
        if xyz is not None:
            assert point_pixel_coords is not None

            if self.normalize:
                x_min, x_max = (
                    torch.min(xyz[..., 0], dim=1, keepdim=True).values,
                    torch.max(xyz[..., 0], dim=1, keepdim=True).values,
                )
                y_min, y_max = (
                    torch.min(xyz[..., 1], dim=1, keepdim=True).values,
                    torch.max(xyz[..., 1], dim=1, keepdim=True).values,
                )
                z_min, z_max = (
                    torch.min(xyz[..., 2], dim=1, keepdim=True).values,
                    torch.max(xyz[..., 2], dim=1, keepdim=True).values,
                )

                min = torch.zeros_like(xyz)
                min[..., 0] = x_min
                min[..., 1] = y_min
                min[..., 2] = z_min
                max = torch.zeros_like(xyz)
                max[..., 0] = x_max
                max[..., 1] = y_max
                max[..., 2] = z_max

                #xyz[..., 0] = (xyz[..., 0] - x_min) / (x_max - x_min)
                #xyz[..., 1] = (xyz[..., 1] - y_min) / (y_max - y_min)
                #xyz[..., 2] = (xyz[..., 2] - z_min) / (z_max - z_min)
                xyz = (xyz - min) / (max-min)

            # Point positional embedding
            xyz_emb = self.xyz_embed(xyz)
            uv_points = self.compute_point_uv(batch_size, point_pixel_coords)
            uv_emb = self.uv_embed(uv_points)

        else:
            # Img positional encoding
            uv = self.create_uv_grid(batch_size)
            uv_emb = self.uv_embed(uv)
            xyz_emb = torch.zeros_like(uv_emb)

        pe = torch.cat([uv_emb, xyz_emb], dim=-1)
        return self.fuse(pe)


class ModalityAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalCrossAtentionTransformer(nn.Module):

    def __init__(
        self,
        visual_emb_dim,
        img_H,
        img_W,
        point_emb_dim,
        patch_size=16,
        num_heads=4,
        num_layers=3,
        out_dim=64,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.patch_size = patch_size
        self.img_W = img_W
        self.img_H = img_H

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(2 * point_emb_dim, num_heads) for _ in range(num_layers)]
        ).to(device=device)

        self.fc_out = nn.Linear(2 * point_emb_dim, out_dim).to(device)

        self.visual_emb_adapter = ModalityAdapter(visual_emb_dim, point_emb_dim).to(
            device
        )

        self.positional_encoder = LearnablePositionalEncoding(
            point_emb_dim * 2, img_H, img_W, patch_size, self.device
        )

    def forward(
        self,
        visual_features,
        point_features,
        point_pixel_coords,
        point_coords: torch.Tensor,
    ):
        # Assert equal batch size
        assert visual_features.shape[0] == point_features.shape[0]
        B = visual_features.shape[0]
        C = point_features.shape[-1]

        visual_features = self.visual_emb_adapter(visual_features)

        point_pixel_coords = (
            point_pixel_coords // self.patch_size
        )

        linear_index = (
            point_pixel_coords[..., 0] * (self.img_W // self.patch_size)
            + point_pixel_coords[..., 1]
        )

        # Concatenate downscaled img features
        fused_features = torch.cat(
            (
                point_features,
                torch.gather(
                    visual_features,
                    dim=1,
                    index=linear_index.unsqueeze(-1).expand(-1, -1, C),
                ),
            ),
            dim=-1,
        )

        # Add positional encoding
        pe = self.positional_encoder.forward(B, point_pixel_coords, point_coords)

        output = fused_features + pe

        # Encoding layers with Multihead attention
        #for layer in self.encoder_layers:
        #    output = layer(output)

        return self.fc_out(output)

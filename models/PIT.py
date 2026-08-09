import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_hgv_config():
    try:
        from . import HGVConfig
        return HGVConfig
    except ImportError:
        from models import HGVConfig
        return HGVConfig


class PITPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PITPhysicsLoss(nn.Module):
    def __init__(self, alpha=None, scaler_mean=None, scaler_std=None):
        super().__init__()
        HGVConfig = get_hgv_config()
        physics_config = HGVConfig.get_physics_config()
        physical_constraints = HGVConfig.get_physical_constraints()
        self.alpha = alpha if alpha is not None else physics_config['alpha']
        self.mse_loss = nn.MSELoss()
        self.enable_denorm = False
        if scaler_mean is not None and scaler_std is not None:
            self.register_buffer('scaler_mean', torch.as_tensor(scaler_mean, dtype=torch.float32))
            self.register_buffer('scaler_std', torch.as_tensor(scaler_std, dtype=torch.float32))
            self.enable_denorm = True
        self.min_height = physical_constraints['min_height']
        self.max_height = physical_constraints['max_height']
        self.earth_radius = physical_constraints['earth_radius']

    def forward(self, pred, target, input_data=None, dt=2.0, include_mse=True):
        if include_mse:
            mse_loss = self.mse_loss(pred, target)
        else:
            mse_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if pred.size(1) == 0:
            return mse_loss
        if self.enable_denorm:
            # 确保 scaler 与 pred 在同一设备上（register_buffer 可能未自动移动）
            scaler_std = self.scaler_std.to(pred.device)
            scaler_mean = self.scaler_mean.to(pred.device)
            pred_physical = pred * scaler_std + scaler_mean
        else:
            pred_physical = pred
        last_pred = pred_physical[:, -1, :]
        physics_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if last_pred.size(-1) >= 1:
            radius = last_pred[:, 0]
            min_r = self.earth_radius + self.min_height
            max_r = self.earth_radius + self.max_height
            radius_penalty = F.softplus(min_r - radius) + F.softplus(radius - max_r)
            physics_loss = physics_loss + radius_penalty.mean()
        total_loss = mse_loss + self.alpha * physics_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return mse_loss
        return total_loss


class TopRMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, top_r=None):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.top_r = top_r
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        if self.top_r is not None:
            top_r = min(self.top_r, seq_len)
            weights_topk, indices = torch.topk(weights, k=top_r, dim=-1)
            weights_topk = weights_topk / (weights_topk.sum(dim=-1, keepdim=True) + 1e-9)
            weights_topk = self.dropout(weights_topk)
            v_flat = v.reshape(batch_size * self.nhead, seq_len, self.head_dim)
            indices_flat = indices.reshape(batch_size * self.nhead, seq_len, top_r)
            indices_expanded = indices_flat.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            v_expanded = v_flat.unsqueeze(1).expand(-1, seq_len, -1, -1)
            v_selected = torch.gather(v_expanded, 2, indices_expanded)
            attn_output = (weights_topk.reshape(batch_size * self.nhead, seq_len, top_r).unsqueeze(-1) * v_selected).sum(dim=2)
            attn_output = attn_output.view(batch_size, self.nhead, seq_len, self.head_dim)
        else:
            weights = self.dropout(weights)
            attn_output = torch.matmul(weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class TopRTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, top_r=None):
        super().__init__()
        self.self_attn = TopRMultiheadAttention(d_model, nhead, dropout=dropout, top_r=top_r)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class PIT(nn.Module):
    def __init__(self, input_dim=6, d_model=None, nhead=None, num_encoder_layers=None, num_decoder_layers=None,
                 dim_feedforward=None, dropout=None, output_dim=3, pred_len=None, top_r=None):
        super().__init__()
        HGVConfig = get_hgv_config()
        model_config = HGVConfig.get_model_config('pit')
        train_config = HGVConfig.get_train_config()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model if d_model is not None else model_config['d_model']
        self.nhead = nhead if nhead is not None else model_config['nhead']
        self.num_encoder_layers = num_encoder_layers if num_encoder_layers is not None else model_config['num_encoder_layers']
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else model_config['num_decoder_layers']
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else model_config['dim_feedforward']
        self.dropout = dropout if dropout is not None else model_config['dropout']
        self.pred_len = pred_len if pred_len is not None else train_config['pred_len']
        self.top_r = top_r if top_r is not None else model_config.get('top_r')
        half_dim = self.d_model // 2
        self.position_embedding = nn.Linear(3, half_dim)
        self.motion_embedding = nn.Linear(3, half_dim)
        self.fusion = nn.Linear(self.d_model, self.d_model)
        self.stage_embedding = nn.Embedding(3, self.d_model)
        self.pos_encoder = PITPositionalEncoding(self.d_model, self.dropout)
        self.tgt_pos_encoder = PITPositionalEncoding(self.d_model, self.dropout)
        self.encoder_layers = nn.ModuleList([
            TopRTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                top_r=self.top_r
            )
            for _ in range(self.num_encoder_layers)
        ])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)
        self.query_embed = nn.Parameter(torch.randn(self.pred_len, self.d_model) * 0.02)
        self.output_projection = nn.Linear(self.d_model, self.output_dim)

    def _compute_stage_ids(self, gamma):
        stage_ids = torch.full_like(gamma, 2, dtype=torch.long)
        stage_ids = torch.where(gamma > 0, torch.zeros_like(stage_ids), stage_ids)
        stage_ids = torch.where(gamma < 0, torch.ones_like(stage_ids), stage_ids)
        return stage_ids

    def forward(self, src, target_length=None):
        if target_length is None:
            target_length = self.pred_len
        pos = src[:, :, :3]
        motion = src[:, :, 3:6]
        pos_embed = self.position_embedding(pos)
        motion_embed = self.motion_embedding(motion)
        fused = torch.tanh(torch.cat([pos_embed, motion_embed], dim=-1))
        fused = self.fusion(fused)
        if src.size(-1) > 4:
            gamma = src[:, :, 4]
        else:
            gamma = torch.zeros(src.size(0), src.size(1), device=src.device, dtype=src.dtype)
        stage_ids = self._compute_stage_ids(gamma)
        stage_embed = self.stage_embedding(stage_ids)
        encoder_input = self.pos_encoder(fused + stage_embed)
        memory = encoder_input
        for layer in self.encoder_layers:
            memory = layer(memory)
        queries = self.query_embed[:target_length].unsqueeze(0).repeat(src.size(0), 1, 1)
        last_gamma = gamma[:, -1]
        target_stage_ids = self._compute_stage_ids(last_gamma).unsqueeze(1).repeat(1, target_length)
        target_stage_embed = self.stage_embedding(target_stage_ids)
        decoder_input = self.tgt_pos_encoder(queries + target_stage_embed)
        decoder_output = self.decoder(decoder_input, memory)
        return self.output_projection(decoder_output)


def create_pit_model(input_dim=6, device=torch.device('cpu'), **kwargs):
    model = PIT(input_dim=input_dim, **kwargs).to(device)
    return model

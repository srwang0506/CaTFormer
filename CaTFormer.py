# -*- coding: utf-8 -*-

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18, resnet50, resnet101
from core.model import MobileFacenet
from core.utils import apply_gate, causal_effect, RMSNorm, PositionalEncoding


class selfnet(nn.Module):
    def __init__(self):
        super(selfnet, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        ])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        out = out.view(out.size(0), -1)
        return out
    

class CaTFormer(nn.Module):
    """
    CaTFormer backbone with RDF + CRE + FSN.
    Interface & public attributes strictly match the original implementation.
    """

    def __init__(
        self,
        feature_dim: int,
        nclass: int,
        hidden_dim: int,
        batch_size: int,
        outnet_name: str = "resnet18",
        innet_name: str = "selfnet",
        num_intents: int = 5,
    ):
        super().__init__()
        D = hidden_dim
        H = 4 * D
        self.model_dim, self.num_intents = D, num_intents
        self.input_proj_in = nn.Linear(64 + 1, D)
        self.input_proj_out = nn.Linear(32 + 1, D)
        
        self.pos_encoder = PositionalEncoding(D)
        enc = nn.TransformerEncoderLayer(d_model=D, nhead=8, dropout=0.1)
        self.encoder_in = nn.TransformerEncoder(enc, 2)
        self.encoder_out = nn.TransformerEncoder(enc, 2)
        self.intent_predictor = nn.Linear(D, num_intents)
        self.intent_proj = nn.Linear(num_intents, D)
        self.cross_in_out = nn.MultiheadAttention(D, 8, 0.1)
        self.cross_out_in = nn.MultiheadAttention(D, 8, 0.1)
        self.norm_in, self.norm_out = RMSNorm(D), RMSNorm(D)
        self.dropout = nn.Dropout(0.1)
        
        self.transin = nn.Sequential(nn.Linear(2 * D, H), nn.ReLU(), nn.Linear(H, D))
        self.transout = nn.Sequential(nn.Linear(2 * D + 3, H), nn.ReLU(), nn.Linear(H, D))
        self.transcat = nn.Sequential(nn.Linear(3 * D + 3, H), nn.ReLU(), nn.Linear(H, D))
        self.attfc = nn.Linear(D, 1)         
        self.in2tag = nn.Linear(D, nclass)
        self.out2tag = nn.Linear(D, nclass)
        self.hidden2tag = nn.Linear(D, nclass)

        self.outnet = self._build_backbone(outnet_name)
        self.innet = self._build_backbone(innet_name, is_in=True)

    def _build_backbone(self, name: str, is_in=False) -> nn.Module:
        """Return backbone specified by name."""
        if name == "resnet18":
            net = resnet18(pretrained=True)
        elif name == "resnet50":
            net = resnet50(pretrained=True)
        elif name == "resnet101":
            net = resnet101(pretrained=True)
        elif name == "mobilefacenet" and is_in:
            net = MobileFacenet()
            ckpt = torch.load("./core/068.ckpt")
            st = net.state_dict()
            st.update({k: v for k, v in ckpt["net_state_dict"].items() if k in st})
            net.load_state_dict(st)
        elif name == "selfnet":
            net = selfnet()
        else:
            raise ValueError(f"Unknown backbone: {name}")
        return net.cuda()

    @staticmethod
    def _shift_right(x: torch.Tensor) -> torch.Tensor:
        """Shift tensor [T,B,D] right by 1 (pad with zeros)."""
        return torch.cat((torch.zeros_like(x[:1]), x[:-1]), 0)

    def _sliding_window_average(self, speed: torch.Tensor, w: int = 5) -> torch.Tensor:
        """1-D causal smoothing: [B] → [B]."""
        if w % 2 == 0:
            w += 1
        pad = w // 2
        kernel = torch.ones(1, 1, w, device=speed.device) / w
        x = F.pad(speed.unsqueeze(1).unsqueeze(1), (pad, pad), mode="replicate")
        return F.conv1d(x, kernel).squeeze(1)

    def forward(self, train_data, targets=None):
        """
        Forward pass.

        Returns:
            res_concat, res_in, res_out, res_joint, intent_log
        """
        in_data, out_data, state_seq, *rest = train_data
        car_state = state_seq[:, -1, 1:].cuda()  
        B, T = in_data.size(0), in_data.size(1)

        infeat = self.innet(in_data.view(-1, *in_data.shape[2:]).cuda()).view(B, T, -1)
        outfeat = self.outnet(out_data.view(-1, *out_data.shape[2:]).cuda()).view(B, T, -1)

        speed = self._sliding_window_average(state_seq[:, -1, 0]) \
                    .unsqueeze(1).expand(-1, T, 1)

        proj_in = self.input_proj_in(torch.cat((infeat, speed), -1))  
        proj_out = self.input_proj_out(torch.cat((outfeat, speed), -1))
        x_in = self.pos_encoder(proj_in.permute(1, 0, 2))   
        x_out = self.pos_encoder(proj_out.permute(1, 0, 2))

        mem_in = self.encoder_in(x_in)    
        mem_out = self.encoder_out(x_out) 

        Δ_in = causal_effect(self.cross_in_out, mem_in, self._shift_right(mem_out))
        Δ_out = causal_effect(self.cross_out_in, mem_out, self._shift_right(mem_in))          

        rep_in = apply_gate(Δ_in, mem_in)
        rep_out = apply_gate(Δ_out, mem_out)

        intent_log = self.intent_predictor(rep_out)          
        itok = self.intent_proj(F.softmax(intent_log, -1))   

        hi = self.transin(torch.cat((rep_in, itok), -1)) + rep_in
        ho = self.transout(torch.cat((rep_out, car_state, itok), -1)) + rep_out
        hc = self.transcat(torch.cat((rep_in, rep_out, car_state, itok), -1)) + rep_in + rep_out

        logits = torch.stack((self.attfc(hi), self.attfc(ho), self.attfc(hc)), 1)  
        weights = F.softmax(logits, 1)                                             

        res_in = self.in2tag(hi)              
        res_out = self.out2tag(ho)            
        res_concat = self.hidden2tag(hc)       
        res_stack = torch.stack((res_in, res_out, res_concat), 2)  
        res_joint = torch.bmm(res_stack, weights).squeeze(2)        

        return res_concat, res_in, res_out, res_joint, intent_log

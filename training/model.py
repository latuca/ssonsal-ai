"""성부 배정 CNN. Cuesta & Gómez(2022)의 salience 기반 성부 배정 구조를 참고한 작은 합성곱 신경망.

입력:  salience (B, 1, T, 264) — Basic Pitch contour(반음당 3빈)
       voice_ranges (B, K, 3)  — [활성, 하한, 상한] (acappella.pipeline.voices.encode_voice_ranges)
출력:  pitch_logits (B, K, T, 88) — 슬롯별 반음 음높이 분포
       active_logits (B, K, T)    — 슬롯별 유성 여부
슬롯은 음역 높은 순(음높이 순서). 입출력 이름은 앱(voices.py)의 ONNX 계약과 같다.
"""

from __future__ import annotations

import torch
from torch import nn

from acappella.pipeline.voices import ONNX_IN_BINS, ONNX_K, ONNX_OUT_BINS


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: tuple[int, int], dil: tuple[int, int] = (1, 1)):
        pad = (dil[0] * (k[0] - 1) // 2, dil[1] * (k[1] - 1) // 2)
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=pad, dilation=dil), nn.BatchNorm2d(cout), nn.GELU()
        )


class VoiceAssignNet(nn.Module):
    def __init__(self, channels: int = 32, k: int = ONNX_K):
        super().__init__()
        c = channels
        # 앞단(파인튜닝 때 고정할 수 있는 부분): 시간·주파수 지역 패턴, 배음 간격 dilation
        self.front = nn.Sequential(
            ConvBlock(1, c, (5, 5)),
            ConvBlock(c, c, (3, 3)),
            ConvBlock(c, c, (3, 3), dil=(1, 12)),  # 4반음 간격
            ConvBlock(c, c, (3, 3), dil=(1, 36)),  # 옥타브 간격(배음)
        )
        self.down = nn.Conv2d(c, c, (1, 3), stride=(1, 3))  # 264 → 88 (반음)
        self.mid = nn.Sequential(ConvBlock(c, c, (5, 3)), ConvBlock(c, c, (9, 1), dil=(2, 1)))
        self.slot_emb = nn.Sequential(nn.Linear(3 + k, 64), nn.GELU(), nn.Linear(64, 2 * c))
        self.head = nn.Sequential(ConvBlock(c, c, (3, 3)), nn.Conv2d(c, 1, 1))
        self.active_head = nn.Linear(c, 1)
        self.k = k

    def front_parameters(self):
        return list(self.front.parameters()) + list(self.down.parameters())

    def forward(self, salience: torch.Tensor, voice_ranges: torch.Tensor):
        B, _, T, F = salience.shape
        assert F == ONNX_IN_BINS
        h = self.mid(self.down(self.front(salience)))  # (B, C, T, 88)
        C = h.shape[1]
        slot_id = torch.eye(self.k, device=salience.device, dtype=salience.dtype).expand(B, -1, -1)
        film = self.slot_emb(torch.cat([voice_ranges, slot_id], dim=-1))  # (B, K, 2C)
        gamma, beta = film[..., :C], film[..., C:]
        hk = h[:, None] * (1 + gamma[..., None, None]) + beta[..., None, None]  # (B, K, C, T, 88)
        hk = hk.reshape(B * self.k, C, T, ONNX_OUT_BINS)
        pitch = self.head(hk).reshape(B, self.k, T, ONNX_OUT_BINS)
        pooled = hk.amax(dim=-1).transpose(1, 2)  # (B*K, T, C)
        active = self.active_head(pooled).reshape(B, self.k, T)
        # 쓰지 않는 슬롯은 무음으로
        active = active + (voice_ranges[..., :1] - 1) * 20
        return pitch, active


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

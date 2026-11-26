# upgraded_model.py This file implements the Pre-trained PVTv2 Encoder, the Dynamic Channel-Wise Attention Mask Network (replacing the global parameter thresholding), and the Global Context Guidance (GCG) Block to anchor camouflaged features.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import timm


class GroupNorm2d(nn.GroupNorm):
    """Group Normalization with safe group calculation"""

    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        num_groups = min(num_groups, num_channels)
        for i in range(num_groups, 0, -1):
            if num_channels % i == 0:
                num_groups = i
                break
        else:
            num_groups = 1
        super().__init__(num_groups, num_channels, eps=eps)


# ==================== CORE NOVEL COMPONENT 1: DYNAMIC SPECTRUM MASK ====================
class DynamicFrequencyDecouplingBlock(nn.Module):
    """FDB with Dynamic Channel-Wise Attention Mask Network to mitigate domain-shift noise"""

    def __init__(self, channels):
        super().__init__()
        self.channel_adaptor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.conv_lf = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            GroupNorm2d(channels),
            nn.GELU()
        )
        self.conv_hf = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            GroupNorm2d(channels),
            nn.GELU()
        )
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        freq = torch.fft.fft2(x, norm='ortho')
        freq_shift = torch.fft.fftshift(freq)

        center_h, center_w = H // 2, W // 2
        y, x_grid = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        dist = torch.sqrt((y - center_h) ** 2 + (x_grid - center_w) ** 2)
        dist = dist.unsqueeze(0).unsqueeze(0)

        dynamic_scale = self.channel_adaptor(x)
        base_sigma = min(H, W) / 6.0
        sigma = base_sigma * (dynamic_scale * 2.0 + 0.5)

        mask = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        mask = torch.sigmoid((mask - 0.3) * 5.0)

        lf_freq = freq_shift * mask
        hf_freq = freq_shift * (1 - mask)

        lf_spatial = torch.fft.ifft2(torch.fft.ifftshift(lf_freq), norm='ortho').real
        hf_spatial = torch.fft.ifft2(torch.fft.ifftshift(hf_freq), norm='ortho').real

        refined = self.fusion(torch.cat([self.conv_lf(lf_spatial), self.conv_hf(hf_spatial)], dim=1))
        return x + refined * 0.5


# ==================== CORE NOVEL COMPONENT 2: GLOBAL CONTEXT GUIDANCE ====================
class GlobalContextGuidance(nn.Module):
    """Broadcasts deep semantic contexts directly down to early skip pathways"""

    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU()
        )
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels // 4, out_channels, 1),
                nn.Sigmoid()
            ) for out_channels in out_channels_list
        ])

    def forward(self, x3):
        g = self.global_pool(x3)
        g = self.reduction(g)
        return [proj(g) for proj in self.projections]


# ==================== SPATIAL DECISION MECHANISMS ====================
class DynamicGatingFusionModule(nn.Module):
    def __init__(self, skip_channels, up_channels):
        super().__init__()
        self.up_adjust = nn.Conv2d(up_channels, skip_channels, 1) if up_channels != skip_channels else nn.Identity()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(skip_channels * 2, skip_channels, 3, padding=1),
            GroupNorm2d(skip_channels),
            nn.GELU(),
            nn.Conv2d(skip_channels, skip_channels, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 3, padding=1),
            GroupNorm2d(skip_channels),
            nn.GELU()
        )

    def forward(self, skip_feat, up_feat):
        up_feat_adjusted = self.up_adjust(up_feat)
        if skip_feat.size()[2:] != up_feat_adjusted.size()[2:]:
            up_feat_adjusted = F.interpolate(up_feat_adjusted, size=skip_feat.shape[2:], mode='bilinear',
                                             align_corners=True)

        combined = torch.cat([skip_feat, up_feat_adjusted], dim=1)
        gate = self.gate_conv(combined)
        return self.refine(skip_feat * gate + up_feat_adjusted)


class ABGSHead(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.body_conv = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1), GroupNorm2d(base_channels),
                                       nn.GELU(), nn.Conv2d(base_channels, 1, 1))
        self.edge_conv = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, padding=1), GroupNorm2d(base_channels),
                                       nn.GELU(), nn.Conv2d(base_channels, 1, 1))
        self.selector = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1), nn.Sigmoid())
        self.refine = nn.Sequential(nn.Conv2d(2, base_channels, 3, padding=1), GroupNorm2d(base_channels), nn.GELU(),
                                    nn.Conv2d(base_channels, 1, 1))

    def forward(self, x):
        body_map = self.body_conv(x)
        edge_map = self.edge_conv(x)
        uncertainty = self.selector(torch.cat([body_map, edge_map], dim=1))
        fused = torch.cat([body_map, edge_map * uncertainty], dim=1)
        return self.refine(fused), edge_map, body_map


# ==================== PRE-TRAINED ENCODER LINK ====================
class PretrainedPVTEncoder(nn.Module):
    """Domain-Invariant PVTv2-B2 Extraction Network with Automated Key-Remapping Support"""

    def __init__(self, base_channels=32, pretrained_path=None):
        super().__init__()
        # Instantiate the baseline features-only blueprint
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True)

        if pretrained_path:
            print(f"🔄 Intercepting state dict from local asset file: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Extract internal dict if wrapped by an training engine signature
            raw_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            remapped_dict = {}

            # Define exact structural transformation mapping dictionary rules
            mapping_rules = {
                "patch_embed1.": "patch_embed.",
                "block1.": "stages_0.blocks.",
                "norm1.": "stages_0.norm.",

                "patch_embed2.": "stages_1.downsample.",
                "block2.": "stages_1.blocks.",
                "norm2.": "stages_1.norm.",

                "patch_embed3.": "stages_2.downsample.",
                "block3.": "stages_2.blocks.",
                "norm3.": "stages_2.norm.",

                "patch_embed4.": "stages_3.downsample.",
                "block4.": "stages_3.blocks.",
                "norm4.": "stages_3.norm."
            }

            for key, val in raw_dict.items():
                new_key = key
                # Apply structural stage remapping rules
                for old_prefix, new_prefix in mapping_rules.items():
                    if key.startswith(old_prefix):
                        new_key = key.replace(old_prefix, new_prefix)
                        break

                # Fix sub-layer token nesting discrepancies (.dwconv.dwconv. -> .dwconv.)
                if ".dwconv.dwconv." in new_key:
                    new_key = new_key.replace(".dwconv.dwconv.", ".dwconv.")

                remapped_dict[new_key] = val

            # Filter out head classification layers since we are performing semantic segmentation
            backbone_keys = {k: v for k, v in remapped_dict.items() if k in self.backbone.state_dict()}

            # Load with strict validation checking turned back on to guarantee alignment safety
            msg = self.backbone.load_state_dict(backbone_keys, strict=True)
            print(f"✅ Pretrained weights loaded successfully into Backbone. Status: {msg}")
        else:
            print("⚠️ Local path missing. Fetching PVTv2-B2 weights online...")
            self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)

        self.proj1 = nn.Conv2d(64, base_channels * 2, 1)
        self.proj2 = nn.Conv2d(128, base_channels * 4, 1)
        self.proj3 = nn.Conv2d(320, base_channels * 8, 1)

    def forward(self, x):
        features = self.backbone(x)
        return [self.proj1(features[0]), self.proj2(features[1]), self.proj3(features[2])]

# ==================== UNIFIED ARCHITECTURE ====================
class FDAM_Net_DomainResilient(nn.Module):
    def __init__(self, base_channels=32, pretrained_path=None):
        super().__init__()
        self.encoder = PretrainedPVTEncoder(base_channels, pretrained_path=pretrained_path)

        c1, c2, c3 = base_channels * 2, base_channels * 4, base_channels * 8

        self.fdb3 = DynamicFrequencyDecouplingBlock(c3)
        self.fdb2 = DynamicFrequencyDecouplingBlock(c2)
        self.fdb1 = DynamicFrequencyDecouplingBlock(c1)

        self.gcg = GlobalContextGuidance(c3, [c2, c1])

        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec3_conv = nn.Sequential(nn.Conv2d(c2, c2, 3, padding=1, bias=False), GroupNorm2d(c2), nn.GELU())
        self.dgfm2 = DynamicGatingFusionModule(c2, c2)

        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec2_conv = nn.Sequential(nn.Conv2d(c1, c1, 3, padding=1, bias=False), GroupNorm2d(c1), nn.GELU())
        self.dgfm1 = DynamicGatingFusionModule(c1, c1)

        self.up1 = nn.ConvTranspose2d(c1, base_channels, 2, stride=2)
        self.dec1_conv = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
                                       GroupNorm2d(base_channels), nn.GELU())
        self.final_upsample = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)

        self.abgs_head = ABGSHead(base_channels, base_channels)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)

        gcg_w2, gcg_w1 = self.gcg(x3)

        x3_refined = self.fdb3(x3)
        x_up3 = self.dec3_conv(self.up3(x3_refined))

        x2_refined = self.fdb2(x2) * gcg_w2
        x_fused2 = self.dgfm2(x2_refined, x_up3)

        x_up2 = self.dec2_conv(self.up2(x_fused2))

        x1_refined = self.fdb1(x1) * gcg_w1
        x_fused1 = self.dgfm1(x1_refined, x_up2)

        x_up1 = self.dec1_conv(self.up1(x_fused1))
        x_final = self.final_upsample(x_up1)

        final_pred, edge_pred, body_pred = self.abgs_head(x_final)
        if self.training:
            return final_pred, edge_pred
        return torch.sigmoid(final_pred)
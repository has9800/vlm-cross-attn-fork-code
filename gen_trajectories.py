import torch, sys, os, numpy as np, copy, pandas as pd
sys.path.insert(0, '/workspace/RoboticsDiffusionTransformer')
os.chdir('/workspace/RoboticsDiffusionTransformer')
from models.rdt_runner import RDTRunner

dtype = torch.bfloat16
runner = RDTRunner.from_pretrained('robotics-diffusion-transformer/rdt-170m', dtype=dtype)
runner = runner.to(device='cuda', dtype=dtype)

class ForkedFinalLayer(torch.nn.Module):
    def __init__(self, o):
        super().__init__()
        self.privileged = copy.deepcopy(o); self.whitelisted = copy.deepcopy(o)
        for p in self.privileged.parameters(): p.requires_grad = False
        for p in self.whitelisted.parameters(): p.requires_grad = True
    def forward(self, x, head='whitelisted'):
        return self.privileged(x) if head == 'privileged' else self.whitelisted(x)

class ForkedRDTBlock(torch.nn.Module):
    def __init__(self, b):
        super().__init__()
        self.norm1, self.attn, self.norm3, self.ffn = b.norm1, b.attn, b.norm3, b.ffn
        for m in [self.norm1, self.attn, self.norm3, self.ffn]:
            for p in m.parameters(): p.requires_grad = False
        self.priv_norm2, self.priv_cross_attn = b.norm2, b.cross_attn
        for m in [self.priv_norm2, self.priv_cross_attn]:
            for p in m.parameters(): p.requires_grad = False
        self.wl_norm2 = copy.deepcopy(b.norm2); self.wl_cross_attn = copy.deepcopy(b.cross_attn)
        for m in [self.wl_norm2, self.wl_cross_attn]:
            for p in m.parameters(): p.requires_grad = True
    def forward(self, x, c, mask=None, head='whitelisted'):
        x = x + self.attn(self.norm1(x))
        ox = x
        if head == 'privileged': x = self.priv_cross_attn(self.priv_norm2(x), c, mask)
        else: x = self.wl_cross_attn(self.wl_norm2(x), c, mask)
        x = x + ox; x = x + self.ffn(self.norm3(x)); return x

class CrossAttnForkedRDT(torch.nn.Module):
    def __init__(self, rdt, n_fork=2):
        super().__init__()
        self.horizon, self.hidden_size = rdt.horizon, rdt.hidden_size
        self.split_idx = len(rdt.blocks) - n_fork
        self.shared_blocks = torch.nn.ModuleList([rdt.blocks[i] for i in range(self.split_idx)])
        for b in self.shared_blocks:
            for p in b.parameters(): p.requires_grad = False
        self.forked_blocks = torch.nn.ModuleList([ForkedRDTBlock(rdt.blocks[i]) for i in range(self.split_idx, len(rdt.blocks))])
        self.forked_final = ForkedFinalLayer(rdt.final_layer)
        self.t_embedder, self.freq_embedder = rdt.t_embedder, rdt.freq_embedder
        self.x_pos_embed = rdt.x_pos_embed
        self.lang_cond_pos_embed, self.img_cond_pos_embed = rdt.lang_cond_pos_embed, rdt.img_cond_pos_embed
        for p in self.t_embedder.parameters(): p.requires_grad = False
        for p in self.freq_embedder.parameters(): p.requires_grad = False
        self.x_pos_embed.requires_grad = False; self.lang_cond_pos_embed.requires_grad = False; self.img_cond_pos_embed.requires_grad = False
    def forward(self, x, freq, t, lang_c, img_c, head='whitelisted', lang_mask=None, img_mask=None):
        t_e = self.t_embedder(t).unsqueeze(1); f_e = self.freq_embedder(freq).unsqueeze(1)
        if t_e.shape[0] == 1: t_e = t_e.expand(x.shape[0], -1, -1)
        x = torch.cat([t_e, f_e, x], dim=1) + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed
        conds, masks = [lang_c, img_c], [lang_mask, img_mask]
        for i, b in enumerate(self.shared_blocks): x = b(x, conds[i%2], masks[i%2])
        for j, b in enumerate(self.forked_blocks): x = b(x, conds[(self.split_idx+j)%2], masks[(self.split_idx+j)%2], head=head)
        x = self.forked_final(x, head=head); return x[:, -self.horizon:]

forked = CrossAttnForkedRDT(runner.model, n_fork=2).cuda()
forked.load_state_dict(torch.load('model_fork2_g1.pt', map_location='cuda'))
forked.eval()
runner.model = None
for p in runner.parameters(): p.requires_grad = False

df = pd.read_parquet('vla_data/unitreerobotics/G1_MountCameraRedGripper_Dataset/data/chunk-000/episode_000000.parquet')
raw_actions = np.stack(df['action'].values).astype(np.float32)
raw_states = np.stack(df['observation.state'].values).astype(np.float32)
actions_128 = np.zeros((len(raw_actions), 128), dtype=np.float32)
states_128 = np.zeros((len(raw_states), 128), dtype=np.float32)
actions_128[:, 64:71] = raw_actions[:, 0:7]; actions_128[:, 72] = raw_actions[:, 7]
actions_128[:, 74:81] = raw_actions[:, 8:15]; actions_128[:, 73] = raw_actions[:, 15]
states_128[:, 64:71] = raw_states[:, 0:7]; states_128[:, 72] = raw_states[:, 7]
states_128[:, 74:81] = raw_states[:, 8:15]; states_128[:, 73] = raw_states[:, 15]

chunk = actions_128[:64]; state = states_128[0:1]
mask = np.zeros((1, 128), dtype=np.float32)
mask[0, [64,65,66,67,68,69,70,72,73,74,75,76,77,78,79,80]] = 1.0
B = 1
lang = torch.randn(B, 32, 4096, device='cuda', dtype=dtype)
img = torch.randn(B, 4374, 1152, device='cuda', dtype=dtype)
st = torch.from_numpy(state).unsqueeze(0).cuda().to(dtype)
ag = torch.from_numpy(chunk).unsqueeze(0).cuda().to(dtype)
am = torch.from_numpy(mask).unsqueeze(0).cuda().to(dtype)
lm = torch.ones(B, 32, dtype=torch.bool, device='cuda')
freq = torch.tensor([25.0], device='cuda')

with torch.no_grad():
    sa = torch.cat([st, ag], dim=1); me = am.expand(-1, sa.shape[1], -1)
    sa = torch.cat([sa, me], dim=2)
    lc, ic, sac = runner.adapt_conditions(lang, img, sa)
    zt = torch.zeros(B, device='cuda').long()
    out_p = forked(sac, freq, zt, lc, ic, head='privileged', lang_mask=lm)
    out_w = forked(sac, freq, zt, lc, ic, head='whitelisted', lang_mask=lm)

def extract_g1(pred_128):
    p = pred_128[0].float().cpu().numpy()
    g1 = np.zeros((64, 16))
    g1[:, 0:7] = p[:, 64:71]; g1[:, 7] = p[:, 72]
    g1[:, 8:15] = p[:, 74:81]; g1[:, 15] = p[:, 73]
    return g1

priv_g1 = extract_g1(out_p); wl_g1 = extract_g1(out_w); gt_g1 = raw_actions[:64]
np.savez('/workspace/trajectory_comparison.npz', priv=priv_g1, wl=wl_g1, gt=gt_g1)
print('Saved trajectory_comparison.npz')
print(f'Priv range: [{priv_g1.min():.3f}, {priv_g1.max():.3f}]')
print(f'WL range: [{wl_g1.min():.3f}, {wl_g1.max():.3f}]')
print(f'GT range: [{gt_g1.min():.3f}, {gt_g1.max():.3f}]')
cos = np.dot(priv_g1.flatten(), wl_g1.flatten()) / (np.linalg.norm(priv_g1) * np.linalg.norm(wl_g1) + 1e-8)
print(f'Priv-WL cosine: {cos:.4f}')

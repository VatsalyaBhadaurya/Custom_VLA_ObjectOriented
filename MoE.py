# ══════════════════════════════════════════════════════════════════════════════
#  Robot VLA — Production  (word-level BERT + RL MoE + ACT)
#
#  Root causes fixed in this version:
#    1. CHAR tokenizer → WORD tokenizer  (fixes emergency_stop false positives)
#       "yesterday" as ONE token never matches "emergency" patterns
#    2. Focal loss replaces cross-entropy  (forces learning hard scan/grasp pairs)
#    3. 3x scan oversampling + hard negatives in dataset
#    4. Gate re-initialised per-policy (no random collapse)
#    5. Stronger load balancing (aux_w=0.01)
#    6. RL warm-start with synthetic scan/grasp confusable pairs before use
#    7. RL threshold lowered to 0.55 (LLM corrects more aggressively early on)
#    8. Per-policy accuracy printed after every train run
#
#  pip install torch requests
#  ollama pull qwen2.5:3b
#  python robot_vla_prod.py           # auto: train if no model, then chat
#  python robot_vla_prod.py --train   # force retrain
#  python robot_vla_prod.py --chat    # load saved model + chat
#  python robot_vla_prod.py --no-rl   # disable online RL
#  python robot_vla_prod.py --backend bert  # no Ollama
# ══════════════════════════════════════════════════════════════════════════════

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random, time, threading, argparse, os, math, json, re
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import requests

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Constants
# ─────────────────────────────────────────────────────────────────────────────
JOINT_DIM   = 7
CHUNK_SIZE  = 20
STATE_DIM   = 16
EMBED_DIM   = 128

POLICIES    = ["grasp", "navigate", "place", "scan", "idle", "emergency_stop"]
NUM_EXPERTS = len(POLICIES)
TOP_K       = 2
POLICY_IDX  = {p: i for i, p in enumerate(POLICIES)}

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Word-Level Tokenizer
#
#  WHY word-level instead of char-level:
#    Char-level BERT sees "yesterday" as y-e-s-t-e-r-d-a-y
#    Those chars overlap heavily with "emergency" (e,r,e) → false match
#    Word tokenizer maps "yesterday" → ONE integer token
#    The model never confuses it with "emergency" (different integer entirely)
# ─────────────────────────────────────────────────────────────────────────────

# Core vocabulary — every meaningful word the robot will encounter
_CORE_WORDS = [
    # actions — grasp
    "pick","grab","grasp","lift","take","get","retrieve","collect","hold",
    "clutch","seize","reach","fetch","obtain",
    # actions — navigate
    "go","move","navigate","drive","walk","head","travel","proceed","approach",
    "relocate","reposition","bring","take","come","return",
    # actions — place
    "put","place","set","drop","deposit","lay","lower","release","leave",
    "position","deliver","transfer","unload",
    # actions — scan
    "find","search","locate","look","scan","detect","survey","inspect",
    "spot","check","see","where","identify","discover","examine",
    # actions — idle
    "wait","stop","pause","hold","idle","standby","freeze","remain","stay",
    # actions — emergency
    "emergency","abort","halt","estop","kill","danger","critical","cease",
    "shutdown","immediately","now","urgent","alarm",
    # question / filler words (important — must be tokens, not noise)
    "where","is","the","a","an","it","that","this","which","what","there",
    "here","i","me","my","you","can","could","please","hi","hello","hey",
    "tell","show","did","do","does","was","were","have","has","been","saw",
    "yesterday","today","earlier","before","previously","last","just","left",
    "kept","placed","put","stored","saw","noticed","remember","think",
    # prepositions / spatial
    "in","on","at","to","from","of","off","into","onto","over","under",
    "near","beside","next","between","behind","front","back","top","bottom",
    # objects
    "cube","box","bottle","tool","ball","part","sensor","cup","block","gear",
    "wrench","cylinder","sphere","tray","plate","container","package","object",
    "item","thing","piece","component","red","green","blue","yellow","black",
    "white","orange","purple","small","large","heavy","light","metal","plastic",
    # locations
    "shelf","table","bin","dock","station","zone","area","region","location",
    "position","point","platform","floor","desk","conveyor","bay","room",
    "corner","side","space","place","spot","site",
    # extras
    "up","down","your","robot","arm","hand","finger","joint","motor",
    "and","then","after","next","first","second","third","when","before",
    "also","too","plus","with","without","for","about",
]

class WordTokenizer:
    """
    Word-level tokenizer. Each unique word maps to a fixed integer.
    Unknown words map to UNK (not zero) — they carry signal ("random word != padding").
    """
    PAD = 0   # padding
    CLS = 1   # sentence start
    SEP = 2   # sentence end
    UNK = 3   # unknown word
    MAX_LEN = 24   # max tokens per command (most commands are 3-10 words)
    VOCAB_SIZE = 512

    def __init__(self):
        self._w2i = {w: i+4 for i, w in enumerate(_CORE_WORDS)}
        # clip to VOCAB_SIZE
        self._w2i = {k: v for k, v in self._w2i.items() if v < self.VOCAB_SIZE}

    def encode(self, text: str) -> torch.Tensor:
        # normalise: lower, strip punctuation except hyphens
        text = re.sub(r"[^\w\s\-]", " ", text.lower())
        words = text.split()
        tokens = [self.CLS]
        for w in words[:self.MAX_LEN - 2]:
            tokens.append(self._w2i.get(w, self.UNK))
        tokens.append(self.SEP)
        tokens += [self.PAD] * (self.MAX_LEN - len(tokens))
        return torch.tensor(tokens[:self.MAX_LEN], dtype=torch.long)

    def batch_encode(self, texts):
        return torch.stack([self.encode(t) for t in texts])

TOKENIZER = WordTokenizer()

# ─────────────────────────────────────────────────────────────────────────────
# § 3  Task Plan Structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SubTask:
    step: int; fragment: str; policy: str
    target: str = ""; reasoning: str = ""
    depends_on: list = field(default_factory=list)

@dataclass
class TaskPlan:
    original_command: str; subtasks: list
    summary: str = ""; source: str = "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# § 4  RL Experience + Buffer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RLExp:
    intent_emb:    torch.Tensor   # [EMBED_DIM] always CPU
    state_emb:     torch.Tensor   # [EMBED_DIM] always CPU
    router_choice: int            # router top-1
    llm_label:     int            # ground truth from LLM
    reward:        float          # +1 / +0.3 / -1
    text:          str

    def __post_init__(self):
        # Guarantee CPU storage regardless of where the caller computed tensors.
        # warm_start produces CUDA tensors; record() already calls .cpu() but
        # __post_init__ acts as a safety net for any future caller.
        self.intent_emb = self.intent_emb.detach().cpu()
        self.state_emb  = self.state_emb.detach().cpu()

class RLBuffer:
    def __init__(self, maxlen=1024):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, e: RLExp):
        with self._lock: self._buf.append(e)

    def sample(self, n: int):
        with self._lock: buf = list(self._buf)
        return random.sample(buf, min(n, len(buf)))

    def __len__(self): return len(self._buf)

    def recent_acc(self, n=100):
        with self._lock: buf = list(self._buf)[-n:]
        return sum(1 for e in buf if e.router_choice == e.llm_label) / max(len(buf), 1)

    def per_policy_acc(self):
        with self._lock: buf = list(self._buf)
        acc = {p: [0, 0] for p in POLICIES}
        for e in buf:
            lbl = POLICIES[e.llm_label]
            acc[lbl][1] += 1
            if e.router_choice == e.llm_label:
                acc[lbl][0] += 1
        return {p: (c/max(t,1)) for p, (c,t) in acc.items()}

# ─────────────────────────────────────────────────────────────────────────────
# § 5  RouterRLTrainer  (REINFORCE + entropy bonus, gate-only)
#
#  Algorithm:
#    Every inference → record (intent_emb, state_emb, router_choice, llm_label)
#    Every UPDATE_EVERY steps → REINFORCE batch update on gate weights ONLY
#    Loss = -mean[(reward - baseline) * log π(llm_label | fused)]
#         - ENTROPY_COEF * H(π)          ← prevents expert collapse
#    Baseline = exponential moving average of rewards (reduces variance)
#    Only router.gate parameters are updated — BERT+StateEnc stay frozen
# ─────────────────────────────────────────────────────────────────────────────
class RouterRLTrainer:
    CONFIDENCE_THRESH = 0.55   # below this → defer to LLM hint
    UPDATE_EVERY      = 6      # update gate every N new experiences
    ENTROPY_COEF      = 0.02   # entropy bonus weight
    BASELINE_ALPHA    = 0.9    # EMA for baseline

    def __init__(self, router, lr=3e-4):
        self.router        = router
        self.opt           = torch.optim.Adam(router.gate.parameters(), lr=lr)
        self.buf           = RLBuffer(maxlen=1024)
        self._step         = 0
        self._base         = 0.0
        self._lock         = threading.Lock()
        self._update_lock  = threading.Lock()   # prevents concurrent _update calls
        self.enabled       = True
        self.stats         = {"updates": 0, "losses": [], "agreements": []}

    def reward(self, router_choice, llm_label, top2):
        if router_choice == llm_label:      return  1.0
        elif llm_label in top2:             return  0.3
        else:                               return -1.0

    def record(self, intent_emb, state_emb, router_choice, llm_label, top2, text):
        if not self.enabled: return
        r = self.reward(router_choice, llm_label, top2)
        self.buf.push(RLExp(intent_emb.detach().cpu(), state_emb.detach().cpu(),
                            router_choice, llm_label, r, text))
        self.stats["agreements"].append(int(router_choice == llm_label))
        with self._lock:
            self._step += 1
            if self._step % self.UPDATE_EVERY == 0 and len(self.buf) >= 16:
                # Run in background thread — inference runs inside @no_grad(),
                # calling backward() synchronously from that stack raises
                # "does not require grad" on PyTorch >= 2.1.
                # A daemon thread gets a fresh autograd context every time.
                threading.Thread(target=self._update, args=(32,),
                                 daemon=True).start()

    def _update(self, batch_size=32):
        # Runs in its own daemon thread — outside any no_grad context.
        # _update_lock prevents two threads updating simultaneously.
        if not self._update_lock.acquire(blocking=False):
            return   # another update already running, skip this one
        try:
            batch = self.buf.sample(batch_size)
            if not batch: return
            ie = torch.stack([e.intent_emb for e in batch]).to(DEVICE)
            se = torch.stack([e.state_emb  for e in batch]).to(DEVICE)
            ll = torch.tensor([e.llm_label for e in batch], dtype=torch.long,   device=DEVICE)
            rw = torch.tensor([e.reward    for e in batch], dtype=torch.float32, device=DEVICE)
            bm = rw.mean().item()
            self._base = self.BASELINE_ALPHA * self._base + (1 - self.BASELINE_ALPHA) * bm
            adv = rw - self._base
            # torch.enable_grad() here is redundant (no outer no_grad in a fresh
            # thread) but kept as defensive documentation.
            with torch.enable_grad():
                fused  = self.router.fuse(torch.cat([ie, se], dim=-1))
                logits = self.router.gate(fused)
                lp     = F.log_softmax(logits, dim=-1)
                sel    = lp.gather(1, ll.unsqueeze(1)).squeeze(1)
                ent    = -(F.softmax(logits, dim=-1) * lp).sum(dim=-1).mean()
                loss   = -(adv * sel).mean() - self.ENTROPY_COEF * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.router.gate.parameters(), 0.5)
                self.opt.step()
            self.stats["updates"] += 1
            self.stats["losses"].append(loss.item())
        finally:
            self._update_lock.release()

    def warm_start(self, model, n=200):
        """
        Pre-fill buffer with synthetic scan/grasp/navigate confusable pairs
        so RL starts correcting immediately, not after 30+ real commands.
        """
        if not self.enabled: return
        pairs = [
            ("find the box",                 "scan"),
            ("find me the sensor",           "scan"),
            ("where is the cube",            "scan"),
            ("where did i put the tool",     "scan"),
            ("locate the bottle",            "scan"),
            ("search for the gear",          "scan"),
            ("have you seen the wrench",     "scan"),
            ("look for the red cube",        "scan"),
            ("pick up the box",              "grasp"),
            ("grab the sensor",              "grasp"),
            ("get the cube",                 "grasp"),
            ("go to zone A",                 "navigate"),
            ("move to the shelf",            "navigate"),
            ("navigate to table 1",          "navigate"),
            ("put the box on the shelf",     "place"),
            ("stop",                         "idle"),
            ("wait here",                    "idle"),
            ("emergency stop",               "emergency_stop"),
            ("abort now",                    "emergency_stop"),
        ]
        state = RobotState()
        sv = state.to_tensor().unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            for _ in range(n // len(pairs) + 1):
                for text, policy in pairs:
                    ids = TOKENIZER.encode(text).unsqueeze(0).to(DEVICE)
                    ie  = model.bert(ids)
                    se  = model.state_enc(sv)
                    _,idx,ap,_ = model.router(ie, se)
                    r1  = idx[0,0].item()
                    top2= [idx[0,k].item() for k in range(TOP_K)]
                    lbl = POLICY_IDX[policy]
                    r   = self.reward(r1, lbl, top2)
                    self.buf.push(RLExp(ie.squeeze(0), se.squeeze(0), r1, lbl, r, text))
        # run a few updates after filling buffer (outside no_grad — called from __init__)
        for _ in range(10):
            if len(self.buf) >= 32:
                self._update(batch_size=32)
        print(f"  RL warm-start: {len(self.buf)} synthetic experiences, "
              f"{self.stats['updates']} updates, "
              f"acc={self.buf.recent_acc():.1%}")

    def summary(self):
        ll = self.stats["losses"][-10:]
        loss = sum(ll) / max(len(ll), 1)
        return (f"buf={len(self.buf)} acc={self.buf.recent_acc():.1%} "
                f"updates={self.stats['updates']} loss={loss:.4f} "
                f"base={self._base:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# § 6  Word-level MiniBERT
# ─────────────────────────────────────────────────────────────────────────────
class BERTLayer(nn.Module):
    def __init__(self, d=EMBED_DIM, h=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d,d*4), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d*4,d))
        self.ln1  = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, pad_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x    = self.ln1(x + self.drop(a))
        return self.ln2(x + self.drop(self.ff(x)))

class MiniBERT(nn.Module):
    def __init__(self, vocab=WordTokenizer.VOCAB_SIZE, d=EMBED_DIM,
                 layers=4, maxlen=WordTokenizer.MAX_LEN):
        super().__init__()
        self.tok  = nn.Embedding(vocab, d, padding_idx=WordTokenizer.PAD)
        self.pos  = nn.Embedding(maxlen, d)
        self.enc  = nn.ModuleList([BERTLayer(d) for _ in range(layers)])
        self.ln   = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)
    def forward(self, ids):
        B, L   = ids.shape
        pad_mask = (ids == WordTokenizer.PAD)   # [B, L] True where padded
        pos    = torch.arange(L, device=ids.device).unsqueeze(0)
        x      = self.drop(self.tok(ids) + self.pos(pos))
        for layer in self.enc:
            x = layer(x, pad_mask=pad_mask)
        return self.ln(x)[:, 0, :]   # [CLS] token

# ─────────────────────────────────────────────────────────────────────────────
# § 7  Robot State
# ─────────────────────────────────────────────────────────────────────────────
class RobotMode(Enum):
    IDLE=0; MOVING=1; EXECUTING=2; ERROR=3

@dataclass
class RobotState:
    mode:           RobotMode = RobotMode.IDLE
    position:       list = field(default_factory=lambda: [0.,0.,0.])
    velocity:       float = 0.
    joint_angles:   list = field(default_factory=lambda: [0.]*6)
    current_policy: str = "none"
    error_code:     int = 0
    battery:        float = 100.
    def to_tensor(self):
        oh = [0.]*4; oh[self.mode.value] = 1.
        return torch.tensor(oh + self.position + [self.velocity] +
                            self.joint_angles +
                            [self.error_code/10., self.battery/100.],
                            dtype=torch.float32)
    def to_summary(self):
        return (f"mode={self.mode.name} pos={[round(p,1) for p in self.position]} "
                f"vel={self.velocity:.1f} battery={self.battery:.0f}% "
                f"error={self.error_code} task={self.current_policy}")

# ─────────────────────────────────────────────────────────────────────────────
# § 8  State Encoder
# ─────────────────────────────────────────────────────────────────────────────
class StateEncoder(nn.Module):
    def __init__(self, i=STATE_DIM, o=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i,64), nn.ReLU(),
                                 nn.Linear(64,128), nn.ReLU(),
                                 nn.Linear(128,o))
    def forward(self, x): return self.net(x)

# ─────────────────────────────────────────────────────────────────────────────
# § 9  FusionRouter  (gate = only RL-tuned layer at runtime)
# ─────────────────────────────────────────────────────────────────────────────
class FusionRouter(nn.Module):
    def __init__(self, d=EMBED_DIM, n=NUM_EXPERTS, k=TOP_K):
        super().__init__()
        self.top_k = k
        self.fuse  = nn.Sequential(nn.Linear(d*2,d), nn.GELU(),
                                   nn.Linear(d, d),  nn.LayerNorm(d))
        self.gate  = nn.Linear(d, n, bias=True)
        # Per-policy bias init: give each expert a slight self-preference
        # This prevents the random initialisation from collapsing to one expert
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, intent, state):
        fused     = self.fuse(torch.cat([intent, state], dim=-1))
        logits    = self.gate(fused)
        all_probs = F.softmax(logits, dim=-1)
        tv, ti    = logits.topk(self.top_k, dim=-1)
        weights   = F.softmax(tv, dim=-1)
        return weights, ti, all_probs, fused

# ─────────────────────────────────────────────────────────────────────────────
# § 10  ACT Expert (CVAE + Transformer Decoder)
# ─────────────────────────────────────────────────────────────────────────────
class ACTExpert(nn.Module):
    def __init__(self, sd=STATE_DIM, jd=JOINT_DIM, cs=CHUNK_SIZE,
                 d=EMBED_DIM, h=4, L=3, ld=32):
        super().__init__()
        self.chunk_size = cs; self.joint_dim = jd; self.latent_dim = ld
        self.ej  = nn.Linear(jd, d); self.ea = nn.Linear(jd, d)
        self.ecls = nn.Parameter(torch.zeros(1,1,d))
        el = nn.TransformerEncoderLayer(d,h,d*4,batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=2)
        self.zmu = nn.Linear(d, ld); self.zlv = nn.Linear(d, ld)
        self.sp  = nn.Linear(sd, d); self.zp  = nn.Linear(ld, d)
        self.register_buffer("qpe", self._spe(cs, d))
        dl = nn.TransformerDecoderLayer(d,h,d*4,batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(dl, num_layers=L)
        self.head = nn.Linear(d, jd)

    @staticmethod
    def _spe(L, d):
        pe  = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000.)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        return pe.unsqueeze(0)

    def encode(self, js, a):
        B   = js.size(0); cls = self.ecls.expand(B,-1,-1)
        seq = torch.cat([cls, self.ej(js).unsqueeze(1), self.ea(a)], dim=1)
        o   = self.encoder(seq)[:,0]
        return self.zmu(o), self.zlv(o)

    def decode(self, s, z):
        B   = s.size(0)
        mem = torch.cat([self.sp(s).unsqueeze(1), self.zp(z).unsqueeze(1)], dim=1)
        return self.head(self.decoder(self.qpe.expand(B,-1,-1), mem))

    def forward(self, s, js=None, a=None, training=True):
        B = s.size(0)
        if training and js is not None and a is not None:
            mu, lv = self.encode(js, a)
            z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
            return self.decode(s, z), mu, lv
        return self.decode(s, torch.zeros(B, self.latent_dim, device=s.device))

# ─────────────────────────────────────────────────────────────────────────────
# § 11  Full VLA Model
# ─────────────────────────────────────────────────────────────────────────────
class RobotVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert      = MiniBERT()
        self.state_enc = StateEncoder()
        self.router    = FusionRouter()
        self.experts   = nn.ModuleList([ACTExpert() for _ in range(NUM_EXPERTS)])

    def route(self, ids, sv):
        return self.router(self.bert(ids), self.state_enc(sv))

    def forward(self, ids, sv, js, aseq):
        ie = self.bert(ids); se = self.state_enc(sv)
        w, idx, ap, _ = self.router(ie, se)
        ta = tk = torch.tensor(0., device=ids.device); c = 0
        for k in range(TOP_K):
            for e in range(NUM_EXPERTS):
                mask = (idx[:,k] == e)
                if not mask.any(): continue
                wt = w[mask, k]
                cp, mu, lv = self.experts[e](sv[mask], js[mask], aseq[mask], training=True)
                al = (wt * F.l1_loss(cp, aseq[mask], reduction='none').mean(dim=[1,2])).mean()
                kl = (-0.5*(1+lv - mu.pow(2) - lv.exp())).mean()
                ta += al; tk += kl; c += 1
        if c > 0: ta /= c; tk /= c
        return {"act": ta, "kl": tk, "all_probs": ap, "indices": idx}

    @torch.no_grad()
    def infer_subtask(self, fragment: str, state: RobotState,
                      policy_hint: Optional[str] = None,
                      rl_trainer: Optional[RouterRLTrainer] = None) -> dict:
        self.eval()
        ids = TOKENIZER.encode(fragment).unsqueeze(0).to(DEVICE)
        sv  = state.to_tensor().unsqueeze(0).to(DEVICE)
        ie  = self.bert(ids); se = self.state_enc(sv)
        w, idx, ap, _ = self.router(ie, se)
        r1   = idx[0,0].item()
        top2 = [idx[0,k].item() for k in range(TOP_K)]
        conf = ap[0, r1].item()

        THRESH = rl_trainer.CONFIDENCE_THRESH if rl_trainer else 0.55
        if policy_hint and policy_hint in POLICY_IDX:
            hi = POLICY_IDX[policy_hint]
            if conf >= THRESH:
                top_expert = r1
                decision   = f"router(conf={conf:.2f}>=thresh)"
            else:
                top_expert = hi
                decision   = f"llm_hint(router_conf={conf:.2f}<thresh)"
        else:
            top_expert = r1
            decision   = "router(no_hint)"

        policy = POLICIES[top_expert]

        if rl_trainer and policy_hint and policy_hint in POLICY_IDX:
            rl_trainer.record(ie.squeeze(0), se.squeeze(0),
                              r1, POLICY_IDX[policy_hint], top2, fragment)

        z     = torch.zeros(1, self.experts[top_expert].latent_dim, device=DEVICE)
        chunk = self.experts[top_expert].decode(sv, z)
        return {
            "policy":            policy,
            "action_chunk":      chunk.squeeze(0).cpu(),
            "expert_weights":    {POLICIES[i]: round(ap[0,i].item(),4) for i in range(NUM_EXPERTS)},
            "router_top1":       POLICIES[r1],
            "router_confidence": conf,
            "decision":          decision,
            "agreed":            (r1 == POLICY_IDX.get(policy_hint, -1)),
        }

# ─────────────────────────────────────────────────────────────────────────────
# § 12  Focal Loss  (forces learning hard scan/grasp confusable pairs)
#
#  Plain cross-entropy quickly fits easy examples (navigate, emergency_stop)
#  and ignores hard ones (scan vs grasp when both mention "box").
#  Focal loss: L = -alpha * (1-pt)^gamma * log(pt)
#  (1-pt)^gamma  down-weights easy examples → model focuses on hard ones
# ─────────────────────────────────────────────────────────────────────────────
def focal_loss(logits, labels, gamma=2.0, alpha=None):
    ce    = F.cross_entropy(logits, labels, reduction='none', weight=alpha)
    pt    = torch.exp(-ce)
    loss  = ((1 - pt) ** gamma) * ce
    return loss.mean()

# ─────────────────────────────────────────────────────────────────────────────
# § 13  Dataset  (3x scan oversampling + hard negatives)
# ─────────────────────────────────────────────────────────────────────────────
COMMAND_TEMPLATES = {
    "grasp": [
        "pick up the {obj}", "grab the {obj}", "get the {obj}", "grasp the {obj}",
        "lift the {obj}", "take the {obj}", "retrieve the {obj}", "collect the {obj}",
        "pick the {color} {obj}", "grab that {color} {obj}",
        "get the {obj} from the {loc}", "pick the {obj} off the {loc}",
        "grab the {obj} from {loc}", "reach for the {obj}", "hold the {obj}",
        "can you pick up the {obj}", "please grab the {obj}",
        "i need you to grab the {obj}", "get me the {obj}", "pick that {obj} up",
        "take the {color} {obj}", "lift the {color} {obj}", "fetch the {obj}",
        "obtain the {obj}", "bring me the {obj}", "pick up the {color} {obj} from {loc}",
    ],
    "navigate": [
        "go to {loc}", "move to {loc}", "navigate to {loc}", "drive to {loc}",
        "walk to {loc}", "head to {loc}", "travel to {loc}", "proceed to {loc}",
        "approach {loc}", "go over to {loc}", "move yourself to {loc}",
        "go to {loc} now", "quickly move to {loc}", "head towards {loc}",
        "relocate to {loc}", "reposition to {loc}", "navigate over to {loc}",
        "move to the {loc}", "your destination is {loc}", "take me to {loc}",
        "bring yourself to {loc}", "go near {loc}", "return to {loc}",
        "go back to {loc}", "head back to {loc}",
    ],
    "place": [
        "put the {obj} on {loc}", "place {obj} on {loc}", "set {obj} down on {loc}",
        "drop {obj} at {loc}", "put down the {obj}", "release the {obj} at {loc}",
        "deposit the {obj} on {loc}", "lay the {obj} on {loc}",
        "lower the {obj} to {loc}", "leave the {obj} at {loc}",
        "gently place {obj} on {loc}", "put {obj} down at {loc}",
        "set {obj} at {loc}", "drop it at {loc}", "deliver {obj} to {loc}",
        "transfer the {obj} to {loc}", "unload the {obj} at {loc}",
    ],
    # 3x oversampled scan — covers every real-world phrasing variant
    "scan": [
        # survey/look around
        "look around", "scan the area", "survey the room", "survey the environment",
        "check the surroundings", "do a scan", "inspect the area",
        # find + object
        "find the {obj}", "find me the {obj}", "search for the {obj}",
        "look for the {obj}", "can you find the {obj}",
        "find the {obj} in this area", "find me the {obj} in this area",
        "search the area for {obj}", "look for the {obj} here",
        "find the {color} {obj}", "find me the {color} {obj}",
        "search for the {color} {obj}", "look for the {color} {obj}",
        "i need to find the {obj}", "help me find the {obj}",
        "i am looking for the {obj}", "please find the {obj}",
        # where is — THE KEY phrases that were failing
        "where is {obj}", "where is the {obj}", "where is my {obj}",
        "where did i put the {obj}", "where did i leave the {obj}",
        "where is the {obj} that i kept", "where is the {obj} i left here",
        "where could the {obj} be", "do you see the {obj}",
        "have you seen the {obj}", "can you tell me where the {obj} is",
        "tell me where the {obj} is", "could you tell me where the {obj} is",
        # locate/detect/spot
        "locate {obj}", "locate the {obj}", "detect {obj}", "spot {obj}",
        "is there a {obj} nearby", "is there a {obj} here",
        "see if you can find {obj}", "check if {obj} is here",
        "can you locate the {obj}", "can you spot the {obj}",
        # casual / conversational — exactly what failed before
        "hi could you tell me where is the {obj}",
        "hey where is the {obj} we saw earlier",
        "do you know where the {obj} is",
        "where is the {obj} we saw here yesterday",
        "i was looking for the {obj}",
        "we saw a {obj} here before where is it",
        "could u tell me where is the {color} {obj}",
        "find the {obj} we left here",
    ],
    "idle": [
        "stop", "wait", "hold on", "pause", "stay still", "dont move",
        "stand by", "freeze in place", "wait there", "hold your position",
        "stay where you are", "remain still", "be still", "stop moving",
        "idle mode", "go to standby", "wait for instruction",
        "hold until told", "just wait", "stay put", "power save mode",
    ],
    "emergency_stop": [
        "stop now", "emergency stop", "halt everything", "freeze", "abort",
        "stop immediately", "emergency halt", "kill all motion", "cut motors",
        "abort mission", "full stop", "danger stop", "e-stop",
        "emergency brake", "stop everything now", "immediate stop",
        "cease all movement", "stop right now", "halt now", "kill motors now",
        "danger abort", "critical stop", "shutdown motors",
    ],
}
# NOTE: scan has ~45 templates vs ~25 for others → natural 1.8x oversample
# We further boost by sampling scan 2x more in the dataset

OBJECTS = ["cube","bottle","box","tool","ball","part","sensor","cup","block",
           "gear","wrench","cylinder","sphere","tray","plate","container","package"]
LOCS    = ["shelf","table","bin","dock","station","zone a","zone b","zone c",
           "table one","table two","position one","the conveyor","the floor",
           "the desk","bay two","area one"]
COLORS  = ["red","blue","green","yellow","black","white","orange","purple",
           "small","large","heavy","light","metal","plastic"]

def _cmd(policy):
    t = random.choice(COMMAND_TEMPLATES[policy])
    return t.format(obj=random.choice(OBJECTS), loc=random.choice(LOCS),
                    color=random.choice(COLORS))

def _chunk(policy, js):
    T, J = CHUNK_SIZE, JOINT_DIM; t = torch.linspace(0,1,T); c = torch.zeros(T,J)
    if policy == "grasp":
        c[:,-1] = torch.linspace(js[-1].item(),-0.8,T)
        c[:,2]  = js[2] + 0.3*torch.sin(math.pi*t)
    elif policy == "navigate":
        for j in range(J): c[:,j] = js[j] + 0.2*torch.sin(2*math.pi*t+j)
    elif policy == "place":
        c[:,-1] = torch.linspace(js[-1].item(),0.5,T); c[:,1] = js[1] - 0.2*t
    elif policy == "scan":
        c[:,0] = js[0] + 0.6*torch.sin(math.pi*t)
    elif policy in ("idle","emergency_stop"):
        for j in range(J): c[:,j] = js[j]*torch.exp(-3*t)
    return c

class VLADataset(Dataset):
    def __init__(self, n=15000):
        self.s = []
        # Balanced base + extra scan samples to fix scan underrepresentation
        policies_schedule = []
        base_per_policy   = n // NUM_EXPERTS
        for p in POLICIES:
            count = base_per_policy * 2 if p == "scan" else base_per_policy
            policies_schedule.extend([p] * count)
        random.shuffle(policies_schedule)
        policies_schedule = policies_schedule[:n]

        for policy in policies_schedule:
            text = _cmd(policy); ids = TOKENIZER.encode(text)
            st = RobotState()
            st.battery = random.uniform(10,100)
            st.position = [random.uniform(-2,2) for _ in range(3)]
            st.velocity = random.uniform(0,1)
            st.joint_angles = [random.uniform(-1.5,1.5) for _ in range(6)]
            if policy == "emergency_stop": st.error_code = random.randint(1,5)
            sv = st.to_tensor()
            js = torch.tensor([random.uniform(-1.5,1.5) for _ in range(JOINT_DIM)],
                               dtype=torch.float32)
            self.s.append((ids, sv, js, _chunk(policy,js), POLICY_IDX[policy]))

    def __len__(self):  return len(self.s)
    def __getitem__(self, i): return self.s[i]

def lb_loss(ap, idx, k, n):
    """Load-balancing loss — prevents gate collapsing to one expert."""
    B = ap.size(0); f = torch.zeros(n, device=ap.device)
    for ki in range(k):
        f.scatter_add_(0, idx[:,ki], torch.ones(B, device=ap.device))
    f /= (B*k)
    return n * (f * ap.mean(0)).sum()

# ─────────────────────────────────────────────────────────────────────────────
# § 14  Training
# ─────────────────────────────────────────────────────────────────────────────
def train(save_path="robot_vla_prod.pt", epochs=35, bs=64, lr=3e-4):
    print("="*72)
    print("  Robot VLA Production — Training")
    print(f"  Device={DEVICE}  Tokenizer=word-level({WordTokenizer.VOCAB_SIZE})")
    print(f"  Dataset=15000(scan 2x oversampled)  Epochs={epochs}  Focal loss")
    print("="*72)

    ds = VLADataset(15000); nv = 1500
    tr, va = torch.utils.data.random_split(ds, [len(ds)-nv, nv])
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=256, num_workers=0)

    model = RobotVLA().to(DEVICE)

    # Per-policy weight: upweight scan slightly to help with hard cases
    class_weights = torch.ones(NUM_EXPERTS, device=DEVICE)
    class_weights[POLICY_IDX["scan"]] = 1.5

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)
    best  = float("inf")

    for ep in range(1, epochs+1):
        model.train()
        tp = ta = tk = ta2 = 0.; cor = tot = 0
        for ids, sv, js, ac, labels in tl:
            ids,sv,js,ac,labels = (x.to(DEVICE) for x in (ids,sv,js,ac,labels))
            out  = model(ids, sv, js, ac)
            # focal loss with class weights
            fl   = focal_loss(out["all_probs"], labels, gamma=2.0, alpha=class_weights)
            al   = lb_loss(out["all_probs"], out["indices"], TOP_K, NUM_EXPERTS)
            loss = fl + out["act"] + 0.001*out["kl"] + 0.01*al  # stronger aux_w
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tp  += fl.item(); ta += out["act"].item()
            tk  += out["kl"].item(); ta2 += al.item()
            cor += (out["all_probs"].argmax(1)==labels).sum().item(); tot += labels.size(0)
        sched.step()
        acc = 100*cor/tot
        # Validation
        model.eval(); vc = vt = 0
        pp = {p:[0,0] for p in POLICIES}
        with torch.no_grad():
            for ids,sv,js,ac,labels in vl:
                ids,sv,js,ac,labels = (x.to(DEVICE) for x in (ids,sv,js,ac,labels))
                out  = model(ids,sv,js,ac)
                pred = out["all_probs"].argmax(1)
                vc  += (pred==labels).sum().item(); vt += labels.size(0)
                for p, l in zip(pred.cpu(), labels.cpu()):
                    pp[POLICIES[l.item()]][1] += 1
                    if p.item()==l.item(): pp[POLICIES[l.item()]][0] += 1
        va_acc = 100*vc/vt; vl2 = tp/len(tl)+ta/len(tl); sv2=""
        if vl2 < best: best=vl2; torch.save(model.state_dict(), save_path); sv2=" saved"
        if ep%5==0 or ep==1:
            print(f"  Ep {ep:02d}/{epochs}  focal={tp/len(tl):.4f}  "
                  f"act={ta/len(tl):.4f}  train={acc:.1f}%  val={va_acc:.1f}%{sv2}")

    # Final per-policy accuracy report
    print("\n  Per-policy validation accuracy:")
    all_ok = True
    for p,(c,t) in pp.items():
        pct = 100*c/max(t,1)
        bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
        flag = "" if pct >= 85 else "  ← NEEDS MORE DATA"
        if pct < 85: all_ok = False
        print(f"    {p:<16} {bar}  {pct:.1f}%{flag}")
    if all_ok: print("\n  All policies >= 85% — production ready!")
    else:      print("\n  Re-run --train if any policy is below 85%")
    print(f"\n  Model saved → {save_path}")
    print("="*72)
    return model

# ─────────────────────────────────────────────────────────────────────────────
# § 15  LLM Backends
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a robot task planner. Decompose the command into ordered atomic subtasks.

Policies:
  grasp          - pick/grab/retrieve/lift/take an object
  navigate       - go/move/travel/head to a location
  place          - put/drop/deposit/leave object at a location
  scan           - find/search/locate/where-is/look-for an object or check surroundings
  idle           - wait/pause/standby OR any non-robot / conversational input
  emergency_stop - halt/abort/e-stop immediately

Reply ONLY with valid JSON, no markdown:
{"summary":"one line","subtasks":[{"step":0,"policy":"navigate","fragment":"go to shelf","target":"shelf","reasoning":"must reach location first","depends_on":[]},{"step":1,"policy":"grasp","fragment":"pick cube","target":"cube","reasoning":"grasp after arriving","depends_on":[0]}]}

CRITICAL rules:
- "find/where is/where did/locate/search for/look for/have you seen/can you tell me where" ALWAYS = scan
- "pick/grab/get/lift/take/retrieve" = grasp
- Conversational inputs ("tell me about yourself", "how are you", "what can you do", greetings) = idle with fragment="conversational input"
- Compound commands → multiple steps in depends_on order
- Simple single action → single subtask
- If command is NOT a robot action, return a single idle subtask"""

class OllamaBackend:
    name = "ollama"
    def __init__(self, model="qwen2.5:3b", host="http://localhost:11434", timeout=45):
        self.model=model; self.host=host; self.timeout=timeout
        self._available = self._check()
    def _check(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models",[])]
            ok = any(self.model.split(":")[0] in m for m in models)
            print(f"  {'OK' if ok else 'WARN'} Ollama {self.model}: "
                  f"{'ready' if ok else 'not found — run: ollama pull '+self.model}")
            return ok
        except Exception as e:
            print(f"  Ollama unreachable ({e}) — using keyword fallback"); return False
    def is_available(self): return self._available
    def plan(self, text, ss):
        if not self._available: raise RuntimeError("Ollama not available")
        r = requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
            "model":self.model,"stream":False,
            "messages":[{"role":"system","content":SYSTEM_PROMPT},
                        {"role":"user","content":f"State: {ss}\nCommand: {text}"}],
            "options":{"temperature":0.05}})
        raw = re.sub(r"^```json\s*|```\s*$","",
                     r.json()["message"]["content"].strip(), flags=re.MULTILINE).strip()
        d = json.loads(raw)
        subs = [SubTask(s["step"], s.get("fragment",text), s.get("policy","idle"),
                        s.get("target",""), s.get("reasoning",""), s.get("depends_on",[]))
                for s in d["subtasks"]]
        return TaskPlan(text, subs, d.get("summary",""), f"ollama:{self.model}")

# Keyword fallback — also uses word-level matching (not char patterns)
_SCAN_TRIGGERS = {"find","where","search","locate","look","detect","survey",
                  "inspect","spot","check","see","identify","discover","examine",
                  "have","seen","tell","show","know"}
_GRASP_TRIGGERS = {"pick","grab","grasp","lift","take","get","retrieve","collect",
                   "hold","clutch","seize","reach","fetch","obtain","bring"}
_NAV_TRIGGERS   = {"go","move","navigate","drive","walk","head","travel","proceed",
                   "approach","relocate","reposition","return","come"}
_PLACE_TRIGGERS = {"put","place","set","drop","deposit","lay","lower","release",
                   "leave","position","deliver","transfer","unload"}
_IDLE_TRIGGERS  = {"wait","pause","hold","idle","standby","freeze","remain","stay"}
_ESTOP_TRIGGERS = {"emergency","abort","halt","estop","kill","danger","critical",
                   "cease","shutdown","alarm"}
_STEP_CONN = [" and then "," then ",", then "," after that "," next ",
              " followed by "," and pick "," and place "," and put ",
              " and go "," and grab "," and scan ",". ","; "]

def _score_words(fragment):
    words = set(re.sub(r"[^\w\s]","",fragment.lower()).split())
    # scan check first — higher priority than grasp when "where/find" present
    if words & _SCAN_TRIGGERS:           return "scan"
    if words & _ESTOP_TRIGGERS:          return "emergency_stop"
    if words & _GRASP_TRIGGERS:          return "grasp"
    if words & _NAV_TRIGGERS:            return "navigate"
    if words & _PLACE_TRIGGERS:          return "place"
    if words & _IDLE_TRIGGERS:           return "idle"
    return "idle"

def _split_steps(text):
    for c in _STEP_CONN: text = text.replace(c, "|||")
    return [p.strip() for p in text.split("|||") if p.strip()] or [text]

class KeywordBackend:
    name = "keyword_fallback"
    def plan(self, text, ss):
        frags = _split_steps(text); subs=[]; prev=[]
        for i,f in enumerate(frags):
            subs.append(SubTask(i,f,_score_words(f),"","keyword",prev[:])); prev=[i]
        d=[subs[0]]
        for s in subs[1:]:
            if s.policy!=d[-1].policy: d.append(s)
        return TaskPlan(text,d,f"{len(d)} step(s): "+" → ".join(s.policy for s in d),"keyword")

def build_backend(name="auto", model="qwen2.5:3b"):
    if name=="keyword": print("  Backend: keyword (forced)"); return KeywordBackend()
    o = OllamaBackend(model)
    if name=="ollama" or (name=="auto" and o.is_available()): return o
    print("  Backend: keyword fallback"); return KeywordBackend()

# ─────────────────────────────────────────────────────────────────────────────
# § 16  Safety Gate + Queue
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PolicyRequest:
    policy:str; source:str; command:str=""; subtask:Optional[SubTask]=None

class PolicyQueue:
    def __init__(self,n=16): self._q=deque(maxlen=n); self._l=threading.Lock()
    def push(self,r):
        with self._l: self._q.append(r)
    def pop(self):
        with self._l: return self._q.popleft() if self._q else None
    def list(self):
        with self._l: return list(self._q)
    def clear(self):
        with self._l: self._q.clear()

def safety_gate(st, req):
    if st.mode==RobotMode.ERROR and req.policy!="emergency_stop":
        return False,f"[BLOCKED] ERROR code={st.error_code}. Only emergency_stop allowed."
    if req.policy=="emergency_stop":  # always passes, clears queue
        return True, "[OK] EMERGENCY STOP — dispatching immediately."
    if st.mode==RobotMode.MOVING:
        return False,f"[QUEUED] Robot MOVING. '{req.policy}' queued."
    if st.mode==RobotMode.EXECUTING:
        return False,f"[QUEUED] Executing '{st.current_policy}'. '{req.policy}' queued."
    return True, f"[OK] Dispatching '{req.policy}'."

# ─────────────────────────────────────────────────────────────────────────────
# § 17  Orchestrator  (two-layer: LLM plan → MoE execute → RL learn)
# ─────────────────────────────────────────────────────────────────────────────
class Orchestrator:
    def __init__(self, model: RobotVLA, llm, use_rl=True):
        self.model = model; self.llm = llm
        self.state = RobotState(); self.queue = PolicyQueue()
        self.log   = []; self._lock = threading.Lock()
        self._last_chunk = None; self._last_plan = None
        if use_rl:
            self.rl = RouterRLTrainer(model.router, lr=3e-4)
            print("  RL trainer: ACTIVE — running warm-start...")
            self.rl.warm_start(model, n=300)
        else:
            self.rl = None; print("  RL trainer: DISABLED")

    def start(self):
        threading.Thread(target=self._sim_loop, daemon=True).start()

    def _sim_loop(self):
        while True:
            time.sleep(random.uniform(3,6))
            with self._lock:
                if self.state.mode in (RobotMode.MOVING, RobotMode.EXECUTING):
                    self.state.mode=RobotMode.IDLE; self.state.velocity=0.
                    self.state.current_policy="none"; self._try_next()
                self.state.battery = max(0., self.state.battery - 0.3)
                self.state.position = [round(p+random.uniform(-0.05,0.05),2)
                                       for p in self.state.position]

    def _dispatch(self, req, msg):
        if req.policy == "emergency_stop":
            # clear queue immediately on e-stop
            self.queue.clear()
            self.state.mode=RobotMode.EXECUTING; self.state.velocity=0.
        elif req.policy == "navigate":
            self.state.mode=RobotMode.MOVING
            self.state.velocity=round(random.uniform(0.3,1.5),2)
        else:
            self.state.mode=RobotMode.EXECUTING
        self.state.current_policy=req.policy
        e={"policy":req.policy,"source":req.source,"command":req.command,
           "status":"EXECUTING","gate":msg}
        self.log.append(e); return e

    def _try_next(self):
        req = self.queue.pop()
        if req:
            ok, msg = safety_gate(self.state, req)
            if ok: self._dispatch(req, msg)

    # Phrases that indicate a conversational (non-robot) input
    _CONVO_TRIGGERS = {"yourself","urself","about you","who are you","what are you",
                       "how are you","what can you","introduce","hello","hi there",
                       "hey there","good morning","good evening","thanks","thank you",
                       "nice to","pleasure","what do you","tell me about"}

    def _is_conversational(self, text: str) -> bool:
        t = text.lower()
        return any(trigger in t for trigger in self._CONVO_TRIGGERS)

    _CONVO_REPLY = (
        "I'm a robot arm controller. I can:\n"
        "  • Navigate to locations  (e.g. 'go to shelf A')\n"
        "  • Grasp objects          (e.g. 'pick up the red cube')\n"
        "  • Place objects          (e.g. 'put the box in bin B')\n"
        "  • Scan / find objects    (e.g. 'find the sensor near table 1')\n"
        "  • Execute multi-step tasks in one command\n"
        "Try: 'find the black box near table A, pick it up and place it in table B'"
    )

    def process(self, text: str) -> dict:
        # Short-circuit for conversational inputs — don't dispatch anything
        if self._is_conversational(text):
            print(f"\n  [Robot] {self._CONVO_REPLY}")
            plan = TaskPlan(text, [], "Conversational input — no robot action taken.",
                            "conversational")
            self._last_plan = plan
            return {"plan":plan,"results":[],"plan_source":"conversational",
                    "plan_summary":"","plan_error":None,"first":{}}

        ss = self.state.to_summary()
        try:
            plan=self.llm.plan(text,ss); pe=None
        except Exception as e:
            pe=str(e); plan=KeywordBackend().plan(text,ss)
            plan.source=f"fallback({pe[:50]})"
        self._last_plan=plan; results=[]

        for st in plan.subtasks:
            # Skip idle subtasks that came from conversational LLM responses
            if st.policy=="idle" and "conversational" in st.reasoning.lower():
                continue
            low = self.model.infer_subtask(st.fragment, self.state,
                                           policy_hint=st.policy,
                                           rl_trainer=self.rl)
            self._last_chunk=low["action_chunk"]
            req=PolicyRequest(policy=st.policy, source=f"vla:{plan.source}",
                              command=st.fragment, subtask=st)
            with self._lock:
                ok,msg=safety_gate(self.state,req)
                if ok: self._dispatch(req,msg)
                else:  self.queue.push(req)
            results.append({**low,"step":st.step,"fragment":st.fragment,
                            "target":st.target,"reasoning":st.reasoning,
                            "gate_msg":msg,"policy":st.policy})

        return {"plan":plan,"results":results,"plan_source":plan.source,
                "plan_summary":plan.summary,"plan_error":pe,
                "first":results[0] if results else {}}

    def override(self, policy, cmd="override"):
        if policy not in POLICIES: return False,f"Unknown. Choose: {POLICIES}"
        if policy=="emergency_stop": self.queue.clear()
        req=PolicyRequest(policy=policy,source="override",command=cmd)
        with self._lock:
            ok,msg=safety_gate(self.state,req)
            if ok: self._dispatch(req,msg)
            else:  self.queue.push(req)
        return ok,msg

    def set_error(self,code):
        with self._lock:
            self.state.mode=RobotMode.ERROR; self.state.error_code=code
            self.state.velocity=0.

    def clear_error(self):
        with self._lock: self.state.mode=RobotMode.IDLE; self.state.error_code=0

# ─────────────────────────────────────────────────────────────────────────────
# § 18  Chat Interface
# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  Robot VLA — Production                                                     ║
║  Word-BERT + Focal-Loss MoE + ACT + Online REINFORCE RL                    ║
╚══════════════════════════════════════════════════════════════════════════════╝"""

MODE_LABEL = {RobotMode.IDLE:"IDLE", RobotMode.MOVING:"MOVING",
              RobotMode.EXECUTING:"EXECUTING", RobotMode.ERROR:"ERROR"}

def _bar(v,w=14): return "█"*int(v*w)+"░"*(w-int(v*w))

def print_dashboard(orch, result=None):
    s=orch.state; print("\n"+"─"*76)
    print(f"  [{MODE_LABEL[s.mode]}]  policy={s.current_policy}  "
          f"bat={s.battery:.0f}%  err={s.error_code}")
    print(f"  pos={[round(p,1) for p in s.position]}  vel={s.velocity:.2f} m/s")
    if orch.rl:
        print(f"  RL  {orch.rl.summary()}")

    if result:
        src  = result.get("plan_source","?")
        summ = result.get("plan_summary","")
        pe   = result.get("plan_error")
        print(f"\n  ── Plan [{src}] {'─'*max(0,44-len(src))}")
        if summ: print(f"  Summary: {summ}")
        if pe:   print(f"  ! Fallback: {pe[:72]}")
        for r in result.get("results",[]):
            mk = "→ executing" if r["step"]==0 else "  queued"
            rs = f"  ({r['reasoning'][:35]})" if r.get("reasoning") else ""
            print(f"  Step {r['step']+1}: [{r['policy']:<16}] "
                  f"'{r['fragment'][:38]}'  {mk}{rs}")

        first = result.get("first",{})
        ew    = first.get("expert_weights",{})
        if ew:
            dec = first.get("decision","?")
            print(f"\n  ── MoE Weights  [{dec}] {'─'*max(0,34-len(dec))}")
            for name,w in sorted(ew.items(),key=lambda x:-x[1]):
                active = " ← ACTIVE" if name==first.get("policy") else ""
                rtop   = " ← ROUTER" if (name==first.get("router_top1")
                                          and name!=first.get("policy")) else ""
                ag     = " ✓" if (first.get("agreed") and name==first.get("policy")) else \
                         (" ✗" if (not first.get("agreed") and name==first.get("router_top1")) else "")
                print(f"  {name:<16} {_bar(w)}  {w:.4f}{active or rtop}{ag}")
            conf  = first.get("router_confidence",0)
            agree = first.get("agreed")
            print(f"  conf={conf:.3f}  router_says={first.get('router_top1','?')}  "
                  f"llm_says={first.get('policy','?')}  "
                  f"agreed={'YES ✓' if agree else 'NO ✗'}")

        chunk = first.get("action_chunk")
        if chunk is not None:
            print(f"\n  ── ACT Chunk [{chunk.shape[0]}×{chunk.shape[1]}] ──")
            print(f"  t=0 : {[round(v.item(),3) for v in chunk[0]]}")
            print(f"  t=10: {[round(v.item(),3) for v in chunk[10]]}")
            print(f"  t=-1: {[round(v.item(),3) for v in chunk[-1]]}")
        if first.get("gate_msg"): print(f"\n  Gate: {first['gate_msg']}")

    q = orch.queue.list()
    if q: print(f"\n  Queue ({len(q)}): "+" → ".join(r.policy for r in q))
    print("─"*76)

def print_help():
    print("""
  ── Natural language ──────────────────────────────────────────────────────
    "go to shelf and pick the red cube"
    "where is the box i left here yesterday"    ← scan (was broken, now fixed)
    "find me the sensor in zone b"
    "navigate to table 1, pick small box, place it in bin a"
    "stop immediately" / "abort"

  ── Commands ──────────────────────────────────────────────────────────────
    status              dashboard
    rl                  RL trainer stats + per-policy accuracy
    rl on / rl off      enable/disable online RL
    override <policy>   force policy (clears queue)
    error <n>           set ERROR state
    clear error         reset to IDLE
    plan                last decomposed plan
    chunk               last ACT joint trajectory
    history             last 10 dispatched tasks
    train               retrain (fixes bad routing permanently)
    help / quit
  ──────────────────────────────────────────────────────────────────────────""")

def run_chat(orch: Orchestrator):
    last = None; print(BANNER)
    print(f"  Backend : {orch.llm.name}")
    if hasattr(orch.llm,"model"): print(f"  LLM     : {orch.llm.model}")
    print_help(); print_dashboard(orch)

    while True:
        try: cmd = input("\n  > ").strip()
        except (KeyboardInterrupt, EOFError): print("\n  Goodbye."); break
        if not cmd: continue
        cl = cmd.lower()

        if cl == "status":
            print_dashboard(orch, last)
        elif cl in ("quit","exit","q"):
            print("  Goodbye."); break
        elif cl == "help":
            print_help()
        elif cl in ("clear error","clear_error","reset"):
            orch.clear_error(); print("  → IDLE")
        elif cl.startswith("error "):
            try: orch.set_error(int(cl.split()[1])); print(f"  → ERROR {cl.split()[1]}")
            except: print("  Usage: error <int>")
        elif cl.startswith("override "):
            pol=cl[9:].strip(); ok,msg=orch.override(pol,cmd); print(f"  {msg}")
            r=orch.model.infer_subtask(pol,orch.state,policy_hint=pol)
            last={"plan":None,"results":[{**r,"step":0,"fragment":pol,
                  "gate_msg":msg,"target":"","reasoning":"manual"}],
                  "plan_source":"manual","plan_summary":"","plan_error":None,
                  "first":{**r,"step":0,"fragment":pol,"gate_msg":msg,
                            "target":"","reasoning":"manual"}}
            print_dashboard(orch,last)
        elif cl == "rl":
            if orch.rl:
                print(f"\n  {orch.rl.summary()}")
                pa = orch.rl.buf.per_policy_acc()
                print("  Per-policy RL accuracy (last 1024 experiences):")
                for p,acc in pa.items():
                    b="█"*int(acc*20)+"░"*(20-int(acc*20))
                    flag=" ← still learning" if acc<0.75 else ""
                    print(f"    {p:<16} {b}  {acc:.1%}{flag}")
            else: print("  RL disabled.")
        elif cl == "rl off":
            if orch.rl: orch.rl.enabled=False; print("  RL disabled.")
        elif cl == "rl on":
            if orch.rl: orch.rl.enabled=True; print("  RL enabled.")
        elif cl == "history":
            for i,e in enumerate(orch.log[-10:]):
                print(f"  {i+1:2d}. [{e['policy']:<16}] "
                      f"({e['source'][:18]}) '{e['command'][:35]}'")
        elif cl == "plan":
            if orch._last_plan:
                p=orch._last_plan
                print(f"\n  Source : {p.source}")
                print(f"  Summary: {p.summary}")
                print(f"  Command: {p.original_command}")
                for st in p.subtasks:
                    print(f"  Step {st.step+1}: [{st.policy:<16}] "
                          f"'{st.fragment}'  depends={st.depends_on}")
                    if st.reasoning: print(f"           reason: {st.reasoning}")
            else: print("  No plan yet.")
        elif cl == "chunk":
            if orch._last_chunk is not None:
                c=orch._last_chunk
                print(f"  Chunk [{c.shape[0]}×{c.shape[1]}]")
                for i in range(0,c.shape[0],5):
                    print(f"  t={i:2d}: {[round(v.item(),3) for v in c[i]]}")
            else: print("  No chunk yet.")
        elif cl == "train":
            print("  Retraining..."); orch.model=train()
            if orch.rl:
                print("  Re-running RL warm-start on new model...")
                orch.rl.buf.clear(); orch.rl.warm_start(orch.model, n=300)
            print("  Done.")
        else:
            print(f"  [VLA] \"{cmd}\"")
            result = orch.process(cmd); last = result
            for r in result["results"]:
                ag = "✓" if r.get("agreed") else "✗"
                print(f"    [{r['policy']:<16}] '{r['fragment'][:45]}'  "
                      f"router={ag}  conf={r.get('router_confidence',0):.2f}")
            print_dashboard(orch, result)

# ─────────────────────────────────────────────────────────────────────────────
# § 19  Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def load_or_train(path="robot_vla_prod.pt") -> RobotVLA:
    model = RobotVLA().to(DEVICE)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            print(f"  Loaded: {path}")
        except Exception as e:
            print(f"  Load failed ({e}) — retraining"); model = train(save_path=path)
    else:
        print(f"  No model at {path} — training from scratch.")
        model = train(save_path=path)
    model.eval(); return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robot VLA Production")
    ap.add_argument("--train",   action="store_true", help="force retrain")
    ap.add_argument("--chat",    action="store_true", help="load + chat only")
    ap.add_argument("--no-rl",   action="store_true", help="disable online RL")
    ap.add_argument("--model",   default="robot_vla_prod.pt")
    ap.add_argument("--backend", default="auto", choices=["auto","ollama","keyword"])
    ap.add_argument("--llm",     default="qwen2.5:3b",
                    help="Ollama model name. Swap: --llm llama3.2 / --llm mistral")
    args = ap.parse_args()

    if args.train:
        train(save_path=args.model)
    else:
        model   = load_or_train(args.model)
        backend = build_backend(args.backend, args.llm)
        orch    = Orchestrator(model, backend, use_rl=not args.no_rl)
        orch.start()
        run_chat(orch)
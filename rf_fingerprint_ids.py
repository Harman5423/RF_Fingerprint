"""
Physical-layer fingerprinting for wireless intrusion detection
Reference implementation of the pipeline in:
  "Beyond MAC Addresses: Physical-Layer Fingerprinting to Close the Identity Gap
   in Wireless Intrusion Detection"

Pipeline (paper Fig. 1):
  IQ capture -> preamble extraction -> CFO correction + normalisation
  -> 1-D CNN embedding f_theta (triplet loss, cosine distance)
  -> prototype store (enrolled devices)
  -> decision engine: ACCEPT / MISMATCH / UNKNOWN / ABSTAIN

Since no real SDR capture is attached, a synthetic transmitter simulator
(IQ imbalance, DC offset, PA nonlinearity, CFO, multipath, AWGN) is included so
the whole thing runs end-to-end. To use real data (ORACLE / WiSig / SMoRFFI),
replace `DeviceSimulator` with a loader that returns complex arrays of shape
(N, 256) per device. Everything else stays the same.

Requires: numpy, torch
Run:      python rf_fingerprint_ids.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

PREAMBLE_LEN = 256      # first 256 samples (as for WiSig)
STF_PERIOD = 16         # 802.11 short-training-field period @ 20 MS/s
FS = 20e6               # 20 MHz Wi-Fi channel


# ----------------------------------------------------------------------------
# 1. Synthetic transmitter simulator (stand-in for the SDR front end)
# ----------------------------------------------------------------------------
def base_preamble():
    """Periodic (16-sample) unit-power preamble, like the 802.11 STF."""
    r = np.random.default_rng(123)
    sym = np.exp(1j * 2 * np.pi * r.random(STF_PERIOD))
    return np.tile(sym, PREAMBLE_LEN // STF_PERIOD)


PREAMBLE = base_preamble()


class DeviceSimulator:
    """One physical transmitter = one fixed set of hardware impairments."""

    def __init__(self, dev_rng):
        self.gain_imb = dev_rng.normal(0, 0.04)        # IQ amplitude imbalance
        self.phase_imb = dev_rng.normal(0, 0.04)       # IQ phase imbalance (rad)
        self.dc = dev_rng.normal(0, 0.02, 2)           # DC offset (I, Q)
        self.pa_a3 = dev_rng.normal(0.10, 0.04)        # cubic PA nonlinearity
        self.cfo_hz = dev_rng.normal(0, 4e3)           # clock error -> CFO

    def transmit(self, n_bursts, snr_db=25.0, r=rng):
        out = np.empty((n_bursts, PREAMBLE_LEN), dtype=np.complex64)
        n = np.arange(PREAMBLE_LEN)
        for i in range(n_bursts):
            x = PREAMBLE.copy()
            # PA nonlinearity (AM/AM compression)
            x = x - self.pa_a3 * (np.abs(x) ** 2) * x
            # IQ imbalance
            I, Q = x.real, x.imag
            I2 = (1 + self.gain_imb) * I
            Q2 = (1 - self.gain_imb) * (Q * np.cos(self.phase_imb) + I * np.sin(self.phase_imb))
            x = I2 + 1j * Q2 + (self.dc[0] + 1j * self.dc[1])
            # CFO (+ small per-burst jitter) and random carrier phase
            cfo = self.cfo_hz + r.normal(0, 150)
            x = x * np.exp(1j * (2 * np.pi * cfo * n / FS + r.uniform(0, 2 * np.pi)))
            # mild random multipath
            h = np.array([1, 0.1 * (r.normal() + 1j * r.normal())])
            x = np.convolve(x, h)[:PREAMBLE_LEN]
            # AWGN
            p = np.mean(np.abs(x) ** 2)
            sigma = np.sqrt(p / (10 ** (snr_db / 10)) / 2)
            x = x + sigma * (r.normal(size=PREAMBLE_LEN) + 1j * r.normal(size=PREAMBLE_LEN))
            out[i] = x
        return out


# ----------------------------------------------------------------------------
# 2. Preprocessing: CFO correction + amplitude normalisation
# ----------------------------------------------------------------------------
def preprocess(iq):
    """iq: (N, 256) complex -> (N, 2, 256) float32 (real/imag channels)."""
    iq = np.atleast_2d(iq)
    out = np.empty((iq.shape[0], 2, iq.shape[1]), dtype=np.float32)
    n = np.arange(iq.shape[1])
    for i, x in enumerate(iq):
        # Schmidl-Cox style CFO estimate from the periodic preamble
        corr = np.sum(x[STF_PERIOD:] * np.conj(x[:-STF_PERIOD]))
        cfo = np.angle(corr) * FS / (2 * np.pi * STF_PERIOD)
        x = x * np.exp(-1j * 2 * np.pi * cfo * n / FS)
        x = x * np.exp(-1j * np.angle(x[0]))            # remove carrier phase
        x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-9)  # amplitude norm
        out[i, 0], out[i, 1] = x.real, x.imag
    return out


def estimate_snr_db(iq):
    """Crude SNR estimate: periodic preamble -> compare period-to-period."""
    iq = np.atleast_2d(iq)
    snrs = []
    for x in iq:
        a, b = x[:-STF_PERIOD], x[STF_PERIOD:]
        b = b * np.exp(-1j * np.angle(np.sum(b * np.conj(a))))
        sig = np.mean(np.abs(a) ** 2)
        noise = np.mean(np.abs(b - a) ** 2) / 2 + 1e-12
        snrs.append(10 * np.log10(max(sig / noise, 1e-6)))
    return np.array(snrs)


# ----------------------------------------------------------------------------
# 3. Embedding network f_theta + triplet loss (Eq. 1)
# ----------------------------------------------------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2),
                                 nn.BatchNorm1d(o), nn.ReLU())
        self.features = nn.Sequential(blk(2, 32, 7), blk(32, 64, 5),
                                      nn.MaxPool1d(2), blk(64, 64, 5),
                                      nn.MaxPool1d(2), blk(64, 128, 3))
        self.head = nn.Linear(128, emb_dim)

    def forward(self, x):
        h = self.features(x).mean(dim=-1)          # global average pooling
        return F.normalize(self.head(h), dim=-1)   # unit-norm embedding


def cosine_dist(u, v):
    return 1 - (u * v).sum(-1)                     # d(u,v) = 1 - u.v/(|u||v|)


def triplet_loss(a, p, n, margin=0.3):
    return F.relu(cosine_dist(a, p) - cosine_dist(a, n) + margin).mean()


def train_embedding(data_by_dev, epochs=25, steps=60, batch=48, lr=1e-3, device="cpu"):
    """data_by_dev: list of (N_i, 2, 256) float32 arrays, one per training device."""
    net = EmbeddingNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    D = len(data_by_dev)
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for _ in range(steps):
            ai, ni = [], []
            a_idx, p_idx, n_idx = [], [], []
            for _ in range(batch):
                d = rng.integers(D)
                d_neg = rng.choice([k for k in range(D) if k != d])
                a_idx.append(data_by_dev[d][rng.integers(len(data_by_dev[d]))])
                p_idx.append(data_by_dev[d][rng.integers(len(data_by_dev[d]))])
                n_idx.append(data_by_dev[d_neg][rng.integers(len(data_by_dev[d_neg]))])
            a = torch.from_numpy(np.stack(a_idx)).to(device)
            p = torch.from_numpy(np.stack(p_idx)).to(device)
            n = torch.from_numpy(np.stack(n_idx)).to(device)
            loss = triplet_loss(net(a), net(p), net(n))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep + 1:02d}/{epochs}  triplet loss {tot / steps:.4f}")
    return net.eval()


@torch.no_grad()
def embed(net, x, device="cpu"):
    return net(torch.from_numpy(x).to(device)).cpu().numpy()


# ----------------------------------------------------------------------------
# 4. Prototype store + decision engine (Eq. 2, Section 3.4)
# ----------------------------------------------------------------------------
class PrototypeStore:
    """Enrolled devices: prototype c_k = mean embedding of N anchor bursts,
    with a per-device threshold tau_k calibrated on held-out enrolment bursts.
    Raw IQ is NOT stored (privacy, Section 3.6)."""

    def __init__(self):
        self.protos, self.tau = {}, {}

    def enroll(self, identity, emb_anchor, emb_calib, fa_quantile=0.99, slack=1.15):
        c = emb_anchor.mean(0)
        c /= np.linalg.norm(c)
        self.protos[identity] = c
        d = 1 - emb_calib @ c
        self.tau[identity] = float(np.quantile(d, fa_quantile) * slack)

    def distances(self, e):
        ids = list(self.protos)
        P = np.stack([self.protos[i] for i in ids])
        return ids, 1 - P @ e


ACCEPT, MISMATCH, UNKNOWN, ABSTAIN = "ACCEPT", "MISMATCH", "UNKNOWN", "ABSTAIN"


class DecisionEngine:
    def __init__(self, store, min_snr_db=10.0, unknown_scale=2.0):
        self.store = store
        self.min_snr_db = min_snr_db
        self.unknown_scale = unknown_scale   # tau_unknown = scale * max tau_k

    @property
    def tau_unknown(self):
        return self.unknown_scale * max(self.store.tau.values())

    def decide(self, emb, claimed_id, snr_db=None, score=None):
        """emb: embedding of the burst (or session-mean embedding);
        claimed_id: identity from the frame parser (BSSID/MAC/SSID).
        score: optionally a pre-aggregated s(x,k)."""
        if snr_db is not None and snr_db < self.min_snr_db:
            return ABSTAIN, {"reason": "low SNR (possible jamming)"}
        ids, d = self.store.distances(emb)
        dmin = float(d.min())
        info = {"nearest": ids[int(d.argmin())], "d_min": dmin}
        if claimed_id in self.store.protos:
            s = float(score if score is not None else d[ids.index(claimed_id)])
            info.update(score=s, tau=self.store.tau[claimed_id])
            if s < self.store.tau[claimed_id]:
                return ACCEPT, info
            info["hardware_known"] = dmin < self.tau_unknown
            return MISMATCH, info          # evil twin / MAC spoof signature
        # claimed identity not enrolled
        if dmin >= self.tau_unknown:
            return UNKNOWN, info           # open-set rejection
        info["note"] = "unenrolled identity on known hardware"
        return UNKNOWN, info


class SessionMonitor:
    """Aggregate evidence over a sliding window of bursts per claimed identity
    (single-frame decisions are too noisy, Section 3.3)."""

    def __init__(self, engine, window=8):
        self.engine, self.window, self.buf = engine, window, {}

    def update(self, claimed_id, emb, snr_db):
        b = self.buf.setdefault(claimed_id, [])
        b.append((emb, snr_db))
        del b[:-self.window]
        embs = np.stack([e for e, _ in b])
        mean = embs.mean(0); mean /= np.linalg.norm(mean)
        snr = float(np.mean([s for _, s in b]))
        score = None
        if claimed_id in self.engine.store.protos:
            score = float(np.mean(1 - embs @ self.engine.store.protos[claimed_id]))
        return self.engine.decide(mean, claimed_id, snr, score), len(b)


# ----------------------------------------------------------------------------
# 5. Evaluation helpers (EER, AUC, TAR@FAR)
# ----------------------------------------------------------------------------
def roc_metrics(genuine, impostor):
    """Scores are distances (lower = more genuine)."""
    th = np.sort(np.concatenate([genuine, impostor]))
    tar = np.array([(genuine <= t).mean() for t in th])
    far = np.array([(impostor <= t).mean() for t in th])
    frr = 1 - tar
    i = np.argmin(np.abs(far - frr))
    eer = (far[i] + frr[i]) / 2
    auc = getattr(np, "trapezoid", getattr(np, "trapz", None))(tar, far)
    tar_at_1 = tar[far <= 0.01].max() if (far <= 0.01).any() else 0.0
    return eer, auc, tar_at_1


# ----------------------------------------------------------------------------
# 6. End-to-end demo
# ----------------------------------------------------------------------------
def main(n_train_dev=30, n_enrol_dev=6, n_unknown_dev=6, train_bursts=200):
    print("== 1. Generate synthetic transmitters")
    mk = lambda: DeviceSimulator(rng)
    train_devs = [mk() for _ in range(n_train_dev)]       # used only for training f_theta
    enrol_devs = [mk() for _ in range(n_enrol_dev)]       # legitimate APs
    unknown_devs = [mk() for _ in range(n_unknown_dev)]   # never seen anywhere

    print("== 2. Train embedding network (triplet loss, cosine distance)")
    train_data = [preprocess(d.transmit(train_bursts, snr_db=rng.uniform(15, 30)))
                  for d in train_devs]
    net = train_embedding(train_data)

    print("== 3. Enrol legitimate APs (prototype + per-device threshold)")
    store = PrototypeStore()
    macs = {i: f"AA:BB:CC:00:00:{i:02X}" for i in range(n_enrol_dev)}
    for i, dev in enumerate(enrol_devs):
        anchors = embed(net, preprocess(dev.transmit(30, snr_db=25)))
        calib = embed(net, preprocess(dev.transmit(100, snr_db=rng.uniform(18, 28))))
        store.enroll(macs[i], anchors, calib)
    engine = DecisionEngine(store)
    print(f"   tau_unknown = {engine.tau_unknown:.3f}")

    print("== 4. Scenario evaluation")
    g_scores, imp_scores = [], []
    for i, dev in enumerate(enrol_devs):
        c = store.protos[macs[i]]
        g_scores += list(1 - embed(net, preprocess(dev.transmit(100, 22))) @ c)
        # evil twin / MAC clone: different hardware, claims macs[i]
        for j, other in enumerate(enrol_devs + unknown_devs):
            if other is dev:
                continue
            imp_scores += list(1 - embed(net, preprocess(other.transmit(10, 22))) @ c)
    eer, auc, tar1 = roc_metrics(np.array(g_scores), np.array(imp_scores))
    print(f"   Clone/spoof detection (single burst): EER={eer:.3f}  AUC={auc:.3f}  TAR@1%FAR={tar1:.3f}")

    # Decision-engine behaviour with session aggregation (8 bursts per decision)
    def scenario(label, dev, claimed_mac, snr):
        mon = SessionMonitor(engine, window=8)
        iq = dev.transmit(8, snr_db=snr)
        embs = embed(net, preprocess(iq))
        snrs = estimate_snr_db(iq)
        for e, s in zip(embs, snrs):
            (verdict, info), _ = mon.update(claimed_mac, e, s)
        info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}
        print(f"   {label:<38s} -> {verdict:<8s} {info}")

    print("   --- session-level decisions ---")
    scenario("1 genuine AP 0", enrol_devs[0], macs[0], 22)
    scenario("2 evil twin (clone of AP 0)", unknown_devs[0], macs[0], 22)
    scenario("3 enrolled AP 1 spoofing AP 0", enrol_devs[1], macs[0], 22)
    scenario("4 rogue, brand-new identity", unknown_devs[1], "DE:AD:BE:EF:00:01", 22)
    scenario("5 genuine AP 0 under jamming (SNR 3)", enrol_devs[0], macs[0], 3)


if __name__ == "__main__":
    main()

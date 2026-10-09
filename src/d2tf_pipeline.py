# ================================================================
# DUAL-DUELS TRANSLATION FRAMEWORK (D2-TF) v4
# NFL BIG DATA BOWL 2027
#
# Status:
#   - Tracking research (LPR, duels, recovery, onset) is descriptive.
#   - The combined GBM is diagnostic only and is NOT deployed.
#   - Scouting scores come from descriptive baselines until a
#     validated, independent outcome exists (outcome.csv).
#   - Prospect path: pre-draft features only, evaluated against a
#     position-mean baseline on a fixed holdout.
# ================================================================

import re
import gc
import json
import zipfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from scipy.stats import spearmanr

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DATASET_ROOT = (
    "/kaggle/input/competitions/nfl-big-data-bowl-2027/nfl-big-data-bowl-2027"
)

CHUNK_SIZE = 250_000
DT = 0.1

LATENT_DIM = 16
ATTN_D_MODEL = 32
ATTN_HEADS = 4
ATTN_DROPOUT = 0.1
WINDOW_LEN = 8
WINDOW_STRIDE = 4
LPR_EPOCHS = 30
LPR_BATCH_SIZE = 2048
LPR_LR = 1e-3
LPR_MAX_SAMPLES = 30_000
LPR_FEATURES = ["speed", "acceleration", "jerk", "snap", "curvature"]
PRES_MAX_WINDOWS = 20_000
PRES_BOOT_ITERS = 500

LEVERAGE_THRESHOLD = 1.2
TARGET_CUSHION = 2.5
RECOVERY_WINDOW = 15
DEV_NORMALIZATION = 0.5

ENDZONE_OFFSET = 110.0   # ASSUMPTION: yards_to_goal = ENDZONE_OFFSET - absoluteYardlineNumber
REDZONE_YARDS = 20
ROUTE_TURN_DEG = 45
ROUTE_LOOKBACK = 5
VERTICAL_MIN_YARDS = 12
PLAY_JOIN_MIN_COVERAGE = 0.5

MIN_EVENTS_TRANSFER = 3
ONSET_HORIZON = 5
ONSET_MAX_ROWS = 300_000

FUT_SPLIT_WEEK = 9
FUT_MIN_LATE_EVENTS = 3
FUT_PRIOR_STRENGTH = 10.0
FUT_CV_SPLITS = 5
FUT_CV_REPEATS = 10
FUT_BOOT_ITERS = 1000
FUT_TOP_FRAC = 0.2

SCOUT_SPLIT_WEEK = FUT_SPLIT_WEEK
SCOUT_MIN_LATE_EVENTS = 3
SCOUT_MIN_TRAIN = 60
SCOUT_GATE_AUC = 0.55
SCOUT_PRIOR_K = 10.0
SCOUT_HIGH_QUANTILE = 2 / 3
SCOUT_CV_SPLITS = 5

PROSPECT_HOLDOUT_FRACTION = 0.25
PROSPECT_MIN_HOLDOUT_N = 30

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCOUT_DIR = OUTPUT_DIR / "scout"
SCOUT_DIR.mkdir(parents=True, exist_ok=True)
PROSPECT_DIR = SCOUT_DIR / "prospect"
PROSPECT_DIR.mkdir(parents=True, exist_ok=True)

RUN_SELF_TEST = True


# ================================================================
# UTILITY
# ================================================================

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_column_name(c):
    c = str(c).strip()
    c = re.sub(r"[^A-Za-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c)
    return c.strip("_").lower()


def squash_key(c):
    return normalize_column_name(c).replace("_", "")


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def find_file(root, filename):
    direct = Path(root) / filename
    if direct.exists():
        return direct
    matches = list(Path(root).rglob(filename))
    return matches[0] if matches else None


# ================================================================
# DATASET DISCOVERY
# ================================================================

def find_dataset_root(preferred=None):
    candidates = []
    if preferred is not None:
        candidates.append(Path(preferred))
    candidates += [Path(DEFAULT_DATASET_ROOT), Path("/kaggle/input"), Path.cwd()]

    seen = set()
    for root in candidates:
        if not root.exists():
            continue
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        if (root / "players.csv").exists():
            print(f"[Dataset] Found root: {root}")
            return root
        try:
            matches = sorted(root.rglob("players.csv"),
                             key=lambda p: ("nfl-big-data-bowl-2027" not in str(p).lower(), len(str(p))))
            if matches:
                print(f"[Dataset] Found root: {matches[0].parent}")
                return matches[0].parent
        except Exception:
            pass
    raise FileNotFoundError("Could not locate players.csv. Set DEFAULT_DATASET_ROOT.")


# ================================================================
# TRACKING FILE DISCOVERY
# ================================================================

def inspect_columns(path):
    try:
        if path.suffix.lower() == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)
        return list(pd.read_parquet(path).columns)
    except Exception:
        return []


def looks_like_tracking(columns):
    norm = {normalize_column_name(c) for c in columns}
    return (
        bool(norm & {"nfl_id", "nflid", "nfl_player_id", "player_id", "player"})
        and bool(norm & {"game_id", "gameid", "game_key", "gamekey"})
        and bool(norm & {"play_id", "playid", "play_key", "playkey"})
        and bool(norm & {"x", "x_position"})
        and bool(norm & {"y", "y_position"})
        and bool(norm & {"s", "speed", "a", "acceleration", "dir", "direction",
                         "dis", "distance", "o", "orientation"})
    )


def discover_tracking_files(root):
    root = Path(root)
    print_section("DISCOVERING NFL GAME TRACKING FILES")
    known = [f"game_tracking_{y}.{ext}" for y in (2023, 2024, 2025) for ext in ("csv", "parquet")]
    found = sorted({p for name in known for p in root.rglob(name) if p.is_file()})
    if found:
        for p in found:
            print(f"  + {p}")
        return found

    all_files = sorted({p for pat in ("**/*.csv", "**/*.parquet") for p in root.glob(pat)})
    valid = []
    for p in all_files:
        name = p.name.lower()
        if "combine" in name:
            continue
        if ("game_tracking" in name or "game-tracking" in name or "tracking_week" in name
                or re.search(r"(^|_)tracking(_|\.|$)", name)):
            if looks_like_tracking(inspect_columns(p)):
                valid.append(p)
                print(f"  + accepted: {p.name}")
    if not valid:
        for p in all_files:
            if "combine" not in p.name.lower() and looks_like_tracking(inspect_columns(p)):
                valid.append(p)
                print(f"  + schema match: {p}")
    if not valid:
        raise FileNotFoundError("No valid NFL game-tracking files were discovered.")
    return sorted(valid)


# ================================================================
# CANONICALIZATION
# ================================================================

COLUMN_ALIASES = {
    "nfl_id": ["nfl_id", "nflid", "nfl_player_id", "player_id", "player"],
    "game_id": ["game_id", "gameid", "game_key", "gamekey"],
    "play_id": ["play_id", "playid", "play_key", "playkey"],
    "frame_id": ["frame_id", "frameid", "frame", "step", "time_step"],
    "x": ["x", "x_position"],
    "y": ["y", "y_position"],
    "speed": ["s", "speed", "velocity"],
    "acceleration": ["a", "acceleration"],
    "distance": ["dis", "distance"],
    "direction": ["dir", "direction", "heading"],
    "orientation": ["o", "orientation"],
    "team": ["team", "club", "team_abbr", "club_abbr"],
    "event": ["event"],
    "time": ["time", "timestamp", "datetime", "time_stamp"],
}


def _rename_to_canonical(df):
    lookup = {normalize_column_name(c): c for c in df.columns}
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            a = normalize_column_name(alias)
            if a in lookup:
                rename_map[lookup[a]] = canonical
                break
    return df.rename(columns=rename_map)


def canonicalize_tracking(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = _rename_to_canonical(df.copy())
    if [c for c in ["nfl_id", "game_id", "play_id", "x", "y"] if c not in df.columns]:
        return pd.DataFrame()

    for c in ["nfl_id", "game_id", "play_id", "frame_id", "x", "y",
              "speed", "acceleration", "distance", "direction", "orientation"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["nfl_id", "game_id", "play_id", "x", "y"])
    for c in ["nfl_id", "game_id", "play_id"]:
        df[c] = df[c].astype(np.int64)

    if "frame_id" not in df.columns:
        if "time" in df.columns:
            parsed = pd.to_datetime(df["time"], errors="coerce")
            if parsed.notna().any():
                df["_t"] = parsed
                df = df.sort_values(["game_id", "play_id", "nfl_id", "_t"]).drop(columns=["_t"])
        else:
            df = df.sort_values(["game_id", "play_id", "nfl_id"])
        df["frame_id"] = df.groupby(["game_id", "play_id", "nfl_id"]).cumcount().astype(np.int32) + 1

    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df = df.dropna(subset=["frame_id"])
    df["frame_id"] = df["frame_id"].astype(np.int32)

    for c in ["speed", "direction", "acceleration", "distance", "orientation"]:
        if c not in df.columns:
            df[c] = np.nan
    return df.reset_index(drop=True)


# ================================================================
# COHORT
# ================================================================

def isolate_cohort_data(root):
    print_section("ISOLATING DUAL-DUEL COHORT")
    players = pd.read_csv(find_file(root, "players.csv"), low_memory=False)
    lookup = {normalize_column_name(c): c for c in players.columns}

    id_column = next((lookup[c] for c in ["nfl_id", "nflid", "nfl_player_id", "player_id", "player"]
                    if c in lookup), None)
    if id_column is None:
        id_column = next((c for c in players.columns if "id" in c.lower()), None)
    position_column = next((lookup[c] for c in ["position", "pos", "officialposition", "player_position"]
                            if c in lookup), None)
    if position_column is None:
        position_column = next((c for c in players.columns if "pos" in c.lower()), None)
    if id_column is None or position_column is None:
        raise ValueError(f"players.csv needs an id and a position column. Columns: {list(players.columns)}")

    players = players.rename(columns={id_column: "nfl_id", position_column: "position"})
    players["nfl_id"] = pd.to_numeric(players["nfl_id"], errors="coerce")
    players["position"] = players["position"].astype(str).str.upper().str.strip()
    players = players.dropna(subset=["nfl_id"])
    players["nfl_id"] = players["nfl_id"].astype(np.int64)

    def ids(pos_set):
        return set(players.loc[players["position"].isin(pos_set), "nfl_id"])

    cohort = {
        "players": players,
        "trench_targets": ids({"DE", "EDGE", "OLB", "DL", "DT"}),
        "trench_adversaries": ids({"T", "OT", "OG", "G"}),
        "perimeter_targets": ids({"WR", "SE", "FL", "TE"}),
        "perimeter_adversaries": ids({"CB", "DB", "FS", "SS", "S"}),
        "qb": ids({"QB"}),
    }
    for k in ["trench_targets", "trench_adversaries", "perimeter_targets", "perimeter_adversaries", "qb"]:
        print(f"[Cohort] {k}: {len(cohort[k])}")
    return cohort


# ================================================================
# KINEMATICS
# ================================================================

def compute_kinematics(df):
    df = canonicalize_tracking(df)
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()
    g = ["game_id", "play_id", "nfl_id"]

    dx = df.groupby(g)["x"].diff().fillna(0)
    dy = df.groupby(g)["y"].diff().fillna(0)

    speed = pd.to_numeric(df["speed"], errors="coerce")
    direction = pd.to_numeric(df["direction"], errors="coerce")
    theta = np.deg2rad(direction)
    valid = speed.notna() & direction.notna()

    df["vx"] = np.where(valid, speed * np.sin(theta), dx / DT)
    df["vy"] = np.where(valid, speed * np.cos(theta), dy / DT)
    df["speed"] = np.where(speed.notna(), speed, np.sqrt(df["vx"] ** 2 + df["vy"] ** 2))

    ax = df.groupby(g)["vx"].diff().fillna(0) / DT
    ay = df.groupby(g)["vy"].diff().fillna(0) / DT
    existing_acc = pd.to_numeric(df["acceleration"], errors="coerce")
    df["acceleration"] = np.where(existing_acc.notna(), existing_acc, np.sqrt(ax ** 2 + ay ** 2))

    df["_ax"], df["_ay"] = ax, ay
    df["_jx"] = df.groupby(g)["_ax"].diff().fillna(0) / DT
    df["_jy"] = df.groupby(g)["_ay"].diff().fillna(0) / DT
    df["jerk"] = np.sqrt(df["_jx"] ** 2 + df["_jy"] ** 2)
    df["_sx"] = df.groupby(g)["_jx"].diff().fillna(0) / DT
    df["_sy"] = df.groupby(g)["_jy"].diff().fillna(0) / DT
    df["snap"] = np.sqrt(df["_sx"] ** 2 + df["_sy"] ** 2)

    numerator = np.abs(df["vx"] * ay - df["vy"] * ax)
    denominator = (df["vx"] ** 2 + df["vy"] ** 2) ** 1.5
    curv = np.where(denominator > 1e-8, numerator / denominator, 0.0)
    df["curvature"] = (pd.Series(curv, index=df.index)
                       .replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 5))

    reconstructed_direction = np.rad2deg(np.arctan2(df["vx"], df["vy"])) % 360
    df["direction"] = np.where(direction.notna(), direction, reconstructed_direction)

    o = pd.to_numeric(df["orientation"], errors="coerce")
    d = pd.to_numeric(df["direction"], errors="coerce")
    df["orientation_delta"] = np.abs(((o - d + 180.0) % 360.0) - 180.0)

    return df.drop(columns=["_ax", "_ay", "_jx", "_jy", "_sx", "_sy"]).reset_index(drop=True)


# ================================================================
# SCALING AND WINDOWS
# ================================================================

class FeatureScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_


def scale_windows(X, scaler):
    if len(X) == 0:
        return X
    F = X.shape[2]
    return scaler.transform(X.reshape(-1, F)).reshape(X.shape).astype(np.float32)


def build_windows(df, group_cols, feats, W=WINDOW_LEN, stride=WINDOW_STRIDE):
    if df is None or df.empty:
        return np.empty((0, W, len(feats)), np.float32), []
    df = df.sort_values(group_cols + ["frame_id"])
    arrays, meta = [], []
    for key, grp in df.groupby(group_cols, sort=False):
        X = np.nan_to_num(grp[feats].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if len(X) < W:
            continue
        frames = grp["frame_id"].to_numpy()
        for s in range(0, len(X) - W + 1, stride):
            arrays.append(X[s:s + W])
            meta.append((key, int(frames[s])))
    if not arrays:
        return np.empty((0, W, len(feats)), np.float32), []
    return np.stack(arrays).astype(np.float32), meta


# ================================================================
# ATTENTION LPR
# ================================================================

class AttentionLPR(nn.Module):
    def __init__(self, n_feat, window, d_model=ATTN_D_MODEL, latent=LATENT_DIM, dropout=ATTN_DROPOUT):
        super().__init__()
        self.window = window
        self.n_feat = n_feat
        self.embed = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=ATTN_HEADS,
                                           dim_feedforward=2 * d_model, dropout=dropout, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=1)
        self.to_latent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent), nn.LayerNorm(latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.SiLU(), nn.Linear(64, window * n_feat))

    def encode(self, x):
        h = self.attn(self.embed(x) + self.pos)
        return self.to_latent(h.mean(dim=1))

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z).view(-1, self.window, self.n_feat), z


def train_attention_lpr(X, epochs=LPR_EPOCHS, batch_size=LPR_BATCH_SIZE, lr=LPR_LR):
    print_section("TRAINING ATTENTION-BASED LATENT POLICY REGENERATOR")
    X = X[np.isfinite(X).all(axis=(1, 2))]
    if len(X) == 0:
        raise ValueError("No finite Combine windows available for LPR training.")
    if len(X) > LPR_MAX_SAMPLES:
        rng = np.random.default_rng(SEED)
        X = X[rng.choice(len(X), size=LPR_MAX_SAMPLES, replace=False)]

    W, F = X.shape[1], X.shape[2]
    model = AttentionLPR(F, W).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            rec, _ = model(batch)
            loss = loss_fn(rec, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
            n += len(batch)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1:02d}/{epochs} | reconstruction loss = {total / max(n, 1):.6f}")
    model.eval()
    return model


def encode_batched(model, X, batch=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
            out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, LATENT_DIM))


def preservation_from_latents(Zs, Zt):
    if len(Zs) == 0 or len(Zt) == 0:
        return 0.0
    cs, ct = Zs.mean(axis=0), Zt.mean(axis=0)
    denom = np.linalg.norm(cs) * np.linalg.norm(ct)
    if denom < 1e-12:
        return 0.0
    cosine = float(np.dot(cs, ct) / denom)
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def latent_preservation_with_ci(model, Xs, Xt, n_boot=PRES_BOOT_ITERS):
    rng = np.random.default_rng(SEED)
    if len(Xs) > PRES_MAX_WINDOWS:
        Xs = Xs[rng.choice(len(Xs), PRES_MAX_WINDOWS, replace=False)]
    if len(Xt) > PRES_MAX_WINDOWS:
        Xt = Xt[rng.choice(len(Xt), PRES_MAX_WINDOWS, replace=False)]

    model.eval()
    Zs, Zt = encode_batched(model, Xs), encode_batched(model, Xt)
    point = preservation_from_latents(Zs, Zt)

    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = preservation_from_latents(
            Zs[rng.integers(0, len(Zs), len(Zs))], Zt[rng.integers(0, len(Zt), len(Zt))])

    lo, hi = np.percentile(draws, [2.5, 97.5])
    if point < lo or point > hi:
        print(f"[LPR] point {point:.6f} outside bootstrap CI [{lo:.6f}, {hi:.6f}]; interval widened to include it.")
    return float(point), float(min(lo, point)), float(max(hi, point))


# ================================================================
# COMBINE TRACKING
# ================================================================

DRILL_COLUMN_CANDIDATES = ["drill", "drill_type", "drill_name", "event", "session", "workout"]


def canonicalize_combine(raw, cohort_ids):
    df = _rename_to_canonical(raw.copy())
    if not {"nfl_id", "x", "y"}.issubset(df.columns):
        print(f"[Combine] Missing nfl_id/x/y. Columns seen: {list(raw.columns)}")
        return pd.DataFrame()

    for c in ["nfl_id", "x", "y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["nfl_id", "x", "y"])
    df["nfl_id"] = df["nfl_id"].astype(np.int64)

    file_ids = set(df["nfl_id"].unique())
    overlap = file_ids & set(cohort_ids)
    print(f"[Combine] athletes in file = {len(file_ids):,} | matching cohort = {len(overlap):,}")
    if not overlap:
        return pd.DataFrame()

    df = df[df["nfl_id"].isin(cohort_ids)].copy()
    if "game_id" not in df.columns:
        df["game_id"] = 0
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").fillna(0).astype(np.int64)

    if "play_id" not in df.columns:
        drill_col = next((c for c in DRILL_COLUMN_CANDIDATES if c in df.columns), None)
        df["play_id"] = (df.groupby(["nfl_id", drill_col]).ngroup() + 1) if drill_col else 1
    df["play_id"] = pd.to_numeric(df["play_id"], errors="coerce").fillna(0).astype(np.int64)

    if "time" in df.columns:
        df["_t"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values(["game_id", "play_id", "nfl_id", "_t"]).drop(columns=["_t"])
    else:
        df = df.sort_values(["game_id", "play_id", "nfl_id"])

    df["frame_id"] = df.groupby(["game_id", "play_id", "nfl_id"]).cumcount().astype(np.int32) + 1
    for c in ["speed", "acceleration", "distance", "direction", "orientation"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def prepare_combine_kinematics(root, cohort_ids):
    print_section("PREPARING COMBINE KINEMATICS")
    path = find_file(root, "combine_tracking.csv")
    if path is None:
        raise FileNotFoundError("combine_tracking.csv not found.")
    raw = pd.read_csv(path, low_memory=False)
    print(f"[Combine] raw shape = {raw.shape} | columns = {list(raw.columns)}")
    df = canonicalize_combine(raw, cohort_ids)
    if df.empty:
        raise RuntimeError("No Combine tracking frames matched the cohort. See diagnostics above.")
    result = compute_kinematics(df)
    print(f"[Combine] isolated frames = {len(result):,}")
    return result


def load_combine_results(root):
    print_section("LOADING COMBINE DRILL RESULTS")
    cands = [p for p in Path(root).rglob("*.csv")
             if "combine" in p.name.lower() and "tracking" not in p.name.lower()]
    if not cands:
        print("[Combine results] none found; combine fields will be empty.")
        return None
    path = sorted(cands, key=lambda p: ("result" not in p.name.lower(), len(p.name)))[0]
    raw = pd.read_csv(path, low_memory=False)
    key = {squash_key(c): c for c in raw.columns}
    id_col = next((key[k] for k in ["nflid", "playerid", "nflplayerid", "player"] if k in key), None)
    if id_col is None:
        print(f"[Combine results] no ID column in {path.name}")
        return None

    out = pd.DataFrame({"nfl_id": pd.to_numeric(raw[id_col], errors="coerce")})
    for name, pats in {"forty": ["40", "forty"], "cone": ["cone"], "shuttle": ["shuttle"]}.items():
        col = next((orig for k, orig in key.items() if any(p in k for p in pats)), None)
        out[name] = pd.to_numeric(raw[col], errors="coerce") if col else np.nan
        print(f"[Combine results] {name} column = {col}")
    out = out.dropna(subset=["nfl_id"])
    out["nfl_id"] = out["nfl_id"].astype(np.int64)
    return out.drop_duplicates("nfl_id").reset_index(drop=True)


# ================================================================
# PLAY CONTEXT & DUELS
# ================================================================

def load_plays(root):
    cols = ["game_id", "play_id", "down", "yards_to_go", "los_x", "yards_to_goal"]
    path = find_file(root, "plays.csv")
    if path is None:
        print("[Plays] plays.csv not found; context features will be NaN.")
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(path, low_memory=False)
    key = {squash_key(c): c for c in raw.columns}
    gid, pid = key.get("gameid"), key.get("playid")
    if gid is None or pid is None:
        print(f"[Plays] game/play id not found. Columns: {list(raw.columns)}")
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame({"game_id": pd.to_numeric(raw[gid], errors="coerce"),
                       "play_id": pd.to_numeric(raw[pid], errors="coerce")})
    dn, ytg, los = key.get("down"), key.get("yardstogo"), key.get("absoluteyardlinenumber")
    df["down"] = pd.to_numeric(raw[dn], errors="coerce") if dn else np.nan
    df["yards_to_go"] = pd.to_numeric(raw[ytg], errors="coerce") if ytg else np.nan
    df["los_x"] = pd.to_numeric(raw[los], errors="coerce") if los else np.nan
    df["yards_to_goal"] = np.where(df["los_x"].notna(), ENDZONE_OFFSET - df["los_x"], np.nan)
    df = df.dropna(subset=["game_id", "play_id"])
    df["game_id"] = df["game_id"].astype(np.int64)
    df["play_id"] = df["play_id"].astype(np.int64)
    print(f"[Plays] rows = {len(df):,} | los present = {df['los_x'].notna().mean():.2f}")
    return df[cols].drop_duplicates(["game_id", "play_id"])


def add_play_context(duels, plays_df, qb_snap):
    out = duels.merge(plays_df, on=["game_id", "play_id"], how="left")
    matched = out["los_x"].notna().mean() if len(out) else 0.0
    print(f"[Context] duel events with a matched play = {matched:.2%}")
    if matched < PLAY_JOIN_MIN_COVERAGE:
        print("[Context] WARNING: low play-join coverage. Check id dtypes in tracking vs plays.csv.")

    out["los_dist"] = (out["target_x"] - out["los_x"]).abs()
    out["redzone"] = np.where(out["yards_to_goal"].notna(),
                            (out["yards_to_goal"] <= REDZONE_YARDS).astype(float), np.nan)
    if qb_snap is not None and not qb_snap.empty:
        out = out.merge(qb_snap, on=["game_id", "play_id"], how="left")
        out["pocket_depth"] = (out["qb_x0"] - out["los_x"]).abs()
    else:
        out["pocket_depth"] = np.nan

    trench_mean = (out[out["duel_type"] == "trench"]
                   .groupby(["game_id", "play_id"])["closing_speed"].mean()
                   .rename("collapse_index").reset_index())
    return out.merge(trench_mean, on=["game_id", "play_id"], how="left")


class DualDuelArbitrator:
    def __init__(self, leverage_threshold=LEVERAGE_THRESHOLD, target_cushion=TARGET_CUSHION,
                 window_frames=RECOVERY_WINDOW):
        self.leverage_threshold = leverage_threshold
        self.target_cushion = target_cushion
        self.window_frames = window_frames

    def analyze_trajectory(self, distances):
        distances = np.asarray(distances, dtype=float)
        distances = distances[np.isfinite(distances)]
        empty = {"compromised": False, "recovered": False, "recovery_rate": 0.0, "dev": 0.0}
        if len(distances) == 0:
            return empty
        compromised_idx = np.where(distances < self.leverage_threshold)[0]
        if len(compromised_idx) == 0:
            return empty

        recovered = False
        for idx in compromised_idx:
            end = min(idx + self.window_frames + 1, len(distances))
            future = distances[idx + 1:end]
            if len(future) and np.any(future >= self.leverage_threshold):
                recovered = True
                break

        deficit = np.maximum(0.0, self.target_cushion - distances)
        delta = np.diff(deficit)
        elimination = -delta[delta < 0]
        dev = float(np.mean(elimination)) if len(elimination) else 0.0
        return {"compromised": True, "recovered": recovered,
                "recovery_rate": float(recovered), "dev": dev}


DUEL_FEATS = ["x", "y", "speed", "acceleration", "jerk", "snap", "curvature",
              "direction", "orientation_delta", "vx", "vy"]


def build_duel_events(df, trench_targets, trench_adversaries, perimeter_targets, perimeter_adversaries):
    base_keys = ["game_id", "play_id", "frame_id"]
    output = []
    for duel_type, t_ids, a_ids in [("trench", trench_targets, trench_adversaries),
                                     ("perimeter", perimeter_targets, perimeter_adversaries)]:
        if not t_ids or not a_ids:
            continue
        targets = df[df["nfl_id"].isin(t_ids)]
        adversaries = df[df["nfl_id"].isin(a_ids)]
        if targets.empty or adversaries.empty:
            continue

        t = targets[base_keys + ["nfl_id"] + DUEL_FEATS].rename(
            columns={"nfl_id": "target_nfl_id", **{f: f"target_{f}" for f in DUEL_FEATS}})
        a = adversaries[base_keys + ["nfl_id"] + DUEL_FEATS].rename(
            columns={"nfl_id": "adv_nfl_id", **{f: f"adv_{f}" for f in DUEL_FEATS}})

        merged = t.merge(a, on=base_keys, how="inner")
        merged = merged[merged["target_nfl_id"] != merged["adv_nfl_id"]]
        if merged.empty:
            continue

        rel_x = merged["adv_x"] - merged["target_x"]
        rel_y = merged["adv_y"] - merged["target_y"]
        dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
        merged["duel_distance"] = dist

        rel_vx = merged["adv_vx"] - merged["target_vx"]
        rel_vy = merged["adv_vy"] - merged["target_vy"]
        safe = np.where(dist > 1e-6, dist, np.nan)
        merged["closing_speed"] = -(rel_x * rel_vx + rel_y * rel_vy) / safe

        t_mag = np.sqrt(merged["target_vx"] ** 2 + merged["target_vy"] ** 2)
        a_mag = np.sqrt(merged["adv_vx"] ** 2 + merged["adv_vy"] ** 2)
        denom = t_mag * a_mag
        merged["velocity_alignment"] = np.where(
            denom > 1e-6,
            (merged["target_vx"] * merged["adv_vx"] + merged["target_vy"] * merged["adv_vy"]) / denom,
            np.nan)

        idx = merged.groupby(["game_id", "play_id", "frame_id", "target_nfl_id"])["duel_distance"].idxmin()
        merged = merged.loc[idx].copy()
        merged["duel_type"] = duel_type
        output.append(merged)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def process_game_file(path, cohort_ids, qb_ids, trench_targets, trench_adversaries,
                      perimeter_targets, perimeter_adversaries):
    print(f"\n[Game] Processing: {path.name}")
    duel_chunks, qb_chunks = [], []
    reader = (pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False)
              if path.suffix.lower() == ".csv" else [pd.read_parquet(path)])

    raw_rows = 0
    for chunk_number, raw in enumerate(reader, start=1):
        raw_rows += len(raw)
        df = canonicalize_tracking(raw)
        if df.empty:
            continue

        qb_rows = df[df["nfl_id"].isin(qb_ids)]
        if not qb_rows.empty:
            first = qb_rows.sort_values("frame_id").drop_duplicates(["game_id", "play_id"], keep="first")
            qb_chunks.append(first[["game_id", "play_id", "frame_id", "x"]].rename(columns={"x": "qb_x0"}))

        df = df[df["nfl_id"].isin(cohort_ids)].copy()
        if df.empty:
            continue
        df = compute_kinematics(df)
        if df.empty:
            continue

        duels = build_duel_events(df, trench_targets, trench_adversaries,
                                  perimeter_targets, perimeter_adversaries)
        if not duels.empty:
            duel_chunks.append(duels)
        if chunk_number % 10 == 0:
            print(f"  chunk {chunk_number:,} | duel rows accumulated = {sum(len(x) for x in duel_chunks):,}")
        del raw, df
        gc.collect()

    print(f"[Game] raw rows read = {raw_rows:,}")
    duels_out = pd.concat(duel_chunks, ignore_index=True) if duel_chunks else pd.DataFrame()
    if not duels_out.empty:
        duels_out["source_file"] = path.name
        print(f"[Game] duel rows from {path.name} = {len(duels_out):,}")
    qb_out = pd.concat(qb_chunks, ignore_index=True) if qb_chunks else pd.DataFrame()
    return duels_out, qb_out


def prepare_game_kinematics(root, cohort):
    print_section("PREPARING REAL NFL GAME DUELS")
    tracking_files = discover_tracking_files(root)
    plays_df = load_plays(root)
    all_ids = (cohort["trench_targets"] | cohort["trench_adversaries"]
               | cohort["perimeter_targets"] | cohort["perimeter_adversaries"])

    duel_frames, qb_frames = [], []
    for path in tracking_files:
        duels, qb = process_game_file(path, all_ids, cohort["qb"],
                                      cohort["trench_targets"], cohort["trench_adversaries"],
                                      cohort["perimeter_targets"], cohort["perimeter_adversaries"])
        if not duels.empty:
            duel_frames.append(duels)
        if not qb.empty:
            qb_frames.append(qb)

    if not duel_frames:
        raise RuntimeError("Game tracking discovered, but no duel frames found.")

    duels = pd.concat(duel_frames, ignore_index=True)
    before = len(duels)
    duels = duels.drop_duplicates(
        subset=["game_id", "play_id", "frame_id", "target_nfl_id", "adv_nfl_id", "duel_type"])
    print(f"[Game] duel rows before dedup = {before:,} | after = {len(duels):,}")

    qb_snap = pd.DataFrame()
    if qb_frames:
        qb_snap = (pd.concat(qb_frames, ignore_index=True).sort_values("frame_id")
                   .drop_duplicates(["game_id", "play_id"], keep="first")[["game_id", "play_id", "qb_x0"]])

    duels = add_play_context(duels, plays_df, qb_snap)
    print(f"[Game] total real duel frames = {len(duels):,}")
    print(duels["duel_type"].value_counts().to_string())
    return duels


def game_lpr_frame(duels):
    cols = {f"target_{f}": f for f in LPR_FEATURES}
    keys = ["duel_type", "game_id", "play_id", "target_nfl_id", "frame_id"]
    return duels[keys + list(cols.keys())].rename(columns=cols)


# ================================================================
# EVENT METRICS & SUMMARIES
# ================================================================

def classify_route(g):
    dirs = g["target_direction"].to_numpy(dtype=float)
    xs = g["target_x"].to_numpy(dtype=float)
    ys = g["target_y"].to_numpy(dtype=float)
    turn = 0.0
    if len(dirs) > ROUTE_LOOKBACK:
        diffs = ((dirs[ROUTE_LOOKBACK:] - dirs[:-ROUTE_LOOKBACK] + 180.0) % 360.0) - 180.0
        if np.isfinite(diffs).any():
            turn = float(np.nanmax(np.abs(diffs)))
    depth = abs(xs[-1] - xs[0]) if len(xs) else 0.0
    lateral = abs(ys[-1] - ys[0]) if len(ys) else 0.0
    if turn >= ROUTE_TURN_DEG:
        return "breaking"
    if depth >= VERTICAL_MIN_YARDS and lateral <= 0.5 * depth:
        return "vertical"
    return "other"


def classify_compromise(row):
    rvx = row["adv_vx"] - row["target_vx"]
    rvy = row["adv_vy"] - row["target_vy"]
    if np.isnan(rvx) or np.isnan(rvy):
        return "unknown"
    return "bull_rush_like" if abs(rvx) >= abs(rvy) else "stunt_twist_like"


def bootstrap_rate_ci(flags, iters=500, seed=SEED):
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(flags), size=(iters, len(flags)))
    means = flags[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_events(duels, arbitrator):
    print_section("EVENT-LEVEL RESPONSE METRICS")
    keys_cols = ["duel_type", "game_id", "play_id", "target_nfl_id"]
    records = []
    for keys, g in duels.groupby(keys_cols, sort=False):
        g = g.sort_values("frame_id")
        d = g["duel_distance"].to_numpy(dtype=float)
        m = arbitrator.analyze_trajectory(d)
        first = g.iloc[0]
        rec = dict(zip(keys_cols, keys))
        rec.update({
            "frames": len(g),
            "mean_distance": float(np.nanmean(d)),
            "min_distance": float(np.nanmin(d)),
            "compromised": int(m["compromised"]),
            "recovered": int(m["recovered"]),
            "dev": float(m["dev"]),
            "down": first.get("down", np.nan),
            "yards_to_go": first.get("yards_to_go", np.nan),
            "yards_to_goal": first.get("yards_to_goal", np.nan),
            "redzone": first.get("redzone", np.nan),
            "los_dist_mean": float(g["los_dist"].mean()) if "los_dist" in g else np.nan,
            "pocket_depth": first.get("pocket_depth", np.nan),
            "collapse_index": first.get("collapse_index", np.nan),
            "route_type": classify_route(g) if keys[0] == "perimeter" else "trench",
            "compromise_type": "none",
            "onset_closing_speed": np.nan,
            "onset_los_dist": np.nan,
        })
        if m["compromised"]:
            onset = int(np.argmax(d < arbitrator.leverage_threshold))
            row = g.iloc[onset]
            rec["compromise_type"] = classify_compromise(row)
            rec["onset_closing_speed"] = float(row["closing_speed"])
            rec["onset_los_dist"] = float(row["los_dist"]) if "los_dist" in row else np.nan
        records.append(rec)

    event_df = pd.DataFrame(records)

    athlete_records = []
    for (dt, tid), g in event_df.groupby(["duel_type", "target_nfl_id"]):
        comp_mask = g["compromised"] == 1
        comp = int(comp_mask.sum())
        flags = g.loc[comp_mask, "recovered"].to_numpy(dtype=float)
        lo, hi = bootstrap_rate_ci(flags)
        athlete_records.append({
            "duel_type": dt, "target_nfl_id": int(tid), "duels": len(g),
            "compromised_events": comp, "recovered_events": int(flags.sum()),
            "recovery_rate": (flags.sum() / comp) if comp else np.nan,
            "rr_ci_low": lo, "rr_ci_high": hi,
            "DEV": float(g["dev"].mean()),
            "mean_distance": float(g["mean_distance"].mean()),
            "min_distance": float(g["min_distance"].mean()),
        })
    athlete_df = pd.DataFrame(athlete_records)

    total_comp = int(event_df["compromised"].sum())
    total_rec = int(event_df.loc[event_df["compromised"] == 1, "recovered"].sum())
    print(f"[Events] {len(event_df):,} duel events | {len(athlete_df):,} athlete-duel rows")
    print(f"[Events] pooled event recovery = {total_rec}/{total_comp} = "
          f"{(total_rec / total_comp) if total_comp else float('nan'):.4f}")
    return event_df, athlete_df


def train_transfer_model(event_df, combine_tbl):
    print_section("COMBINE-TO-GAME TRANSFER (descriptive, same season)")
    if combine_tbl is None:
        return None, None
    feats = ["forty", "cone", "shuttle"]
    ev = event_df[event_df["compromised"] == 1]
    pooled = (ev.groupby("target_nfl_id")
              .agg(recovery_rate=("recovered", "mean"), n_events=("recovered", "size"))
              .reset_index().rename(columns={"target_nfl_id": "nfl_id"}))
    pooled["nfl_id"] = pooled["nfl_id"].astype(np.int64)

    tbl = pooled.merge(combine_tbl, on="nfl_id", how="inner")
    tbl = tbl[tbl["n_events"] >= MIN_EVENTS_TRANSFER].dropna(subset=feats, how="all").copy()
    if len(tbl) < 20:
        print(f"[Transfer] Only {len(tbl)} athletes; skipping (need >= 20).")
        return None, None

    for f in feats:
        tbl[f] = tbl[f].fillna(tbl[f].median())
    Xr = -tbl[feats]
    X = ((Xr - Xr.mean()) / Xr.std().replace(0, 1)).to_numpy()
    y = tbl["recovery_rate"].to_numpy()

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    model = Ridge(alpha=1.0)
    cv_pred = cross_val_predict(model, X, y, cv=kf)
    r2 = float(r2_score(y, cv_pred))
    rho = float(spearmanr(cv_pred, y).correlation)
    model.fit(X, y)
    coefs = dict(zip(feats, [float(c) for c in model.coef_]))
    print(f"[Transfer] athletes = {len(tbl)} | CV R^2 = {r2:.4f} | CV Spearman = {rho:.4f}")
    tbl["transfer_score_cv"] = cv_pred
    stats = {"n_athletes": int(len(tbl)), "cv_r2": r2, "cv_spearman": rho, "coefficients": coefs}
    return tbl[["nfl_id", "recovery_rate", "n_events", "transfer_score_cv"]], stats


ONSET_FEATURES = [
    "target_speed", "adv_speed", "target_acceleration", "adv_acceleration",
    "target_jerk", "target_snap", "target_orientation_delta",
    "closing_speed", "velocity_alignment", "los_dist", "pocket_depth",
]


def build_onset_dataset(duels):
    H = ONSET_HORIZON
    parts = []
    keep = [c for c in ONSET_FEATURES if c in duels.columns] + ["game_id"]
    for _, g in duels.groupby(["duel_type", "game_id", "play_id", "target_nfl_id"], sort=False):
        if len(g) <= H:
            continue
        g = g.sort_values("frame_id")
        d = g["duel_distance"].to_numpy(dtype=float)
        n = len(d)
        fut = np.full(n, np.inf)
        for h in range(1, H + 1):
            fut[:-h] = np.minimum(fut[:-h], d[h:])
        mask = (d >= LEVERAGE_THRESHOLD) & np.isfinite(fut)
        mask[n - H:] = False
        if not mask.any():
            continue
        sub = g.loc[mask, keep].copy()
        sub["y"] = (fut[mask] < LEVERAGE_THRESHOLD).astype(int)
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    data = pd.concat(parts, ignore_index=True)
    if len(data) > ONSET_MAX_ROWS:
        data = data.sample(ONSET_MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return data


def train_onset_model(duels):
    print_section("COMPROMISE-ONSET MODEL (distance excluded)")
    data = build_onset_dataset(duels)
    if data.empty or data["y"].nunique() < 2:
        print("[Onset] Not enough labeled frames; skipping.")
        return None
    feats = [c for c in ONSET_FEATURES if c in data.columns]
    X, y, groups = data[feats], data["y"].to_numpy(), data["game_id"].to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED)
    model.fit(X.iloc[tr], y[tr])
    auc = float(roc_auc_score(y[te], model.predict_proba(X.iloc[te])[:, 1]))
    print(f"[Onset] rows = {len(data):,} | positive rate = {y.mean():.3f} | holdout AUC = {auc:.4f}")
    return {"auc": auc}


def run_translation_analysis(event_df, athlete_df, preservation, pres_lo, pres_hi, combine_df, game_duels):
    print_section("D2-TF TRANSLATION SUMMARY")
    comp = int(event_df["compromised"].sum())
    rec = int(event_df.loc[event_df["compromised"] == 1, "recovered"].sum())
    recovery_rate = (rec / comp) if comp > 0 else 0.0
    dev = float(event_df["dev"].mean())
    cti = float(np.clip(100.0 * (0.40 * preservation + 0.35 * recovery_rate + 0.25 * np.clip(dev / DEV_NORMALIZATION, 0.0, 1.0)), 0.0, 100.0))

    results = {
        "latent_policy_preservation": preservation,
        "preservation_ci95": [pres_lo, pres_hi],
        "reactive_recovery_rate_pooled": float(recovery_rate),
        "reactive_recovery_events": {"recovered": rec, "compromised": comp},
        "deficit_elimination_velocity": dev,
        "CTI": cti,
        "combine_frames": int(len(combine_df)),
        "game_duel_frames": int(len(game_duels)),
        "athletes": int(athlete_df["target_nfl_id"].nunique()),
    }
    with open(OUTPUT_DIR / "d2tf_summary.json", "w") as f:
        json.dump(_clean(results), f, indent=4)
    return results


# ================================================================
# MAIN PIPELINE EXECUTION
# ================================================================

def run_d2tf(dataset_root=None):
    print("\n" + "=" * 80)
    print("DUAL-DUELS TRANSLATION FRAMEWORK (D2-TF) v4")
    print("=" * 80)
    set_seed(SEED)
    root = find_dataset_root(dataset_root)
    print(f"\nDataset root: {root} | Device: {DEVICE}")

    cohort = isolate_cohort_data(root)
    combine_ids = cohort["trench_targets"] | cohort["perimeter_targets"]

    combine_df = prepare_combine_kinematics(root=root, cohort_ids=combine_ids)
    combine_windows, _ = build_windows(combine_df, ["game_id", "play_id", "nfl_id"], LPR_FEATURES)

    game_duels = prepare_game_kinematics(root=root, cohort=cohort)
    arbitrator = DualDuelArbitrator(LEVERAGE_THRESHOLD, TARGET_CUSHION, RECOVERY_WINDOW)
    event_df, athlete_df = summarize_events(game_duels, arbitrator)

    game_lpr = game_lpr_frame(game_duels)
    game_windows, _ = build_windows(game_lpr, ["duel_type", "game_id", "play_id", "target_nfl_id"], LPR_FEATURES)

    scaler = FeatureScaler().fit(combine_windows.reshape(-1, combine_windows.shape[2]))
    Xs = scale_windows(combine_windows, scaler)
    Xt = scale_windows(game_windows, scaler)
    model = train_attention_lpr(Xs)
    preservation, pres_lo, pres_hi = latent_preservation_with_ci(model, Xs, Xt)

    run_translation_analysis(event_df, athlete_df, preservation, pres_lo, pres_hi, combine_df, game_duels)
    print("\n" + "=" * 80)
    print("D2-TF PIPELINE COMPLETE")
    print(f"Results directory: {OUTPUT_DIR.resolve()}")
    return True


if __name__ == "__main__":
    run_d2tf(dataset_root=DEFAULT_DATASET_ROOT)

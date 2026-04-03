import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from itertools import combinations

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("keno_ultra")


# =========================================================
# CONSTANTS
# =========================================================
APP_TITLE = "🎯 Keno Ultra Master — 20/70"
PDF_CANDIDATES = ["keno.PDF", "keno.pdf", "kino.PDF", "kino.pdf"]

NUM_MIN = 1
NUM_MAX = 70
DRAW_PICK_COUNT = 20
DRAWNO_MIN = 100000

DEFAULT_SEED = 20260402
DEFAULT_CANDIDATES = 3000

FREQ_SHORT = 30
FREQ_MEDIUM = 100
FREQ_LONG = 300

INT_RE = re.compile(r"\d+")


# =========================================================
# CSS
# =========================================================
APP_CSS = """
<style>
:root{
  --bg0:#f4fbff;
  --bg1:#ffffff;
  --card:#ffffff;
  --txt:#000000;
  --mut:#1f2937;
  --blue:#2563eb;
  --blue2:#3b82f6;
  --green:#059669;
  --red:#dc2626;
  --gold:#d4af37;
  --border: rgba(37,99,235,0.18);
  --shadow: 0 10px 28px rgba(0,0,0,.08);
}

.stApp{
  background-color: var(--bg0) !important;
  background-image:
    radial-gradient(1000px 700px at 10% 10%, rgba(59,130,246,0.08), transparent 58%),
    radial-gradient(900px 600px at 90% 15%, rgba(5,150,105,0.06), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
  color: var(--txt) !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] *{
  color: var(--txt) !important;
}

.v-card{
  background: linear-gradient(180deg, #ffffff, #fbfdff);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: 18px;
  padding: 16px;
  margin-bottom: 16px;
}

.v-pill{
  display:inline-block;
  padding:6px 10px;
  margin:3px 4px 0 0;
  border-radius:999px;
  border:1px solid rgba(37,99,235,0.25);
  background:rgba(37,99,235,0.10);
  font-weight:900;
  color:#000000 !important;
}

.v-pill-cold{
  display:inline-block;
  padding:6px 10px;
  margin:3px 4px 0 0;
  border-radius:999px;
  border:1px solid rgba(5,150,105,0.25);
  background:rgba(5,150,105,0.10);
  font-weight:900;
  color:#000000 !important;
}

.v-pill-predict{
  display:inline-block;
  padding:6px 10px;
  margin:3px 4px 0 0;
  border-radius:999px;
  border:1px solid rgba(212,175,55,0.35);
  background:rgba(212,175,55,0.14);
  font-weight:900;
  color:#000000 !important;
}

.v-muted{
  opacity:.86;
  font-size:.92rem;
  color:#1f2937 !important;
}

.rank-card{
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  border: 1px solid rgba(37,99,235,0.18);
  border-radius: 16px;
  padding: 14px;
  margin: 10px 0;
  box-shadow: 0 8px 18px rgba(0,0,0,.05);
}

.rank-title{
  font-size: 1.05rem;
  font-weight: 900;
  margin-bottom: 6px;
}

.rank-main{
  font-size: 1.1rem;
  font-weight: 1000;
  letter-spacing: .6px;
  margin-bottom: 6px;
}

.rank-meta{
  font-size: .95rem;
  line-height: 1.5;
}

.help-box{
  background: rgba(37,99,235,0.06);
  border: 1px solid rgba(37,99,235,0.16);
  border-radius: 14px;
  padding: 12px;
  margin: 8px 0;
}

.tip-box{
  background: rgba(5,150,105,0.08);
  border: 1px solid rgba(5,150,105,0.16);
  border-radius: 14px;
  padding: 12px;
  margin: 8px 0;
}

.warn-box{
  background: rgba(220,38,38,0.06);
  border: 1px solid rgba(220,38,38,0.16);
  border-radius: 14px;
  padding: 12px;
  margin: 8px 0;
}

div.stButton > button[kind="primary"]{
  background: linear-gradient(90deg, var(--blue) 0%, var(--blue2) 100%) !important;
  color: #000000 !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.80rem 1.10rem !important;
  font-weight: 1000 !important;
  letter-spacing: .4px !important;
  box-shadow: 0 10px 22px rgba(37,99,235,0.20) !important;
}

.btn-gold > button{
  background: linear-gradient(90deg, #d4af37 0%, #ffdf00 100%) !important;
  color: #000 !important;
  box-shadow: 0 10px 22px rgba(212,175,55,0.35) !important;
}

[data-testid="stDataFrame"]{
  border-radius: 16px !important;
  overflow: hidden !important;
  border: 1px solid rgba(37,99,235,0.18) !important;
}
</style>
"""


# =========================================================
# DATA MODELS
# =========================================================
@dataclass
class KenoRecord:
    draw_no: Optional[int]
    nums: List[int]
    date_str: str = "—"


@dataclass
class TicketScore:
    ticket: List[int]
    final_score: float
    hot_score: float
    cold_score: float
    momentum_score: float
    overdue_score: float
    pair_score: float
    recent_penalty: float
    source: str


# =========================================================
# GENERAL HELPERS
# =========================================================
def resolve_pdf_path() -> Path:
    cwd = Path(os.getcwd())
    for name in PDF_CANDIDATES:
        p = cwd / name
        if p.exists():
            return p
    return cwd / PDF_CANDIDATES[0]


def sanitize_txt_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = "wynik.txt"
    name = name.replace("\\", "_").replace("/", "_").replace("..", "_")
    if not name.lower().endswith(".txt"):
        name += ".txt"
    return name


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def normalize_score_map(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    mn = min(vals)
    mx = max(vals)
    if mx == mn:
        return {k: 0.0 for k in score_map}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def similarity_to_recent(ticket: List[int], recent_draws: List[List[int]]) -> int:
    tset = set(ticket)
    if not recent_draws:
        return 0
    return max(len(tset.intersection(set(d))) for d in recent_draws)


def weighted_unique_sample(
    rng: np.random.Generator,
    population: List[int],
    weights: List[float],
    k: int
) -> List[int]:
    arr_pop = np.array(population, dtype=int)
    arr_w = np.array(weights, dtype=float)

    if len(arr_pop) < k:
        raise ValueError("Za mała pula do losowania.")

    if arr_w.sum() <= 0:
        arr_w = np.ones_like(arr_w, dtype=float)

    probs = arr_w / arr_w.sum()
    chosen = rng.choice(arr_pop, size=k, replace=False, p=probs)
    return sorted(chosen.tolist())


def lines_to_bytes(lines: List[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


# =========================================================
# PDF PARSING
# =========================================================
def _validate_pdf_bytes(pdf_bytes: bytes) -> None:
    if not pdf_bytes.startswith(b"%PDF"):
        head = pdf_bytes[:200].decode("utf-8", errors="replace")
        raise ValueError(
            "Plik nie wygląda jak prawdziwy PDF.\n"
            f"Początek pliku:\n{head}"
        )


def _read_pdf_pages_text(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text") or "")
    doc.close()
    return pages


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _is_noise_line(line: str) -> bool:
    low = line.lower()
    noise = [
        "keno 20/70",
        "kino 20/70",
        "www.",
        "©",
        "drukuj",
        "print",
        "reklama",
    ]
    return any(x in low for x in noise)


def _split_page_tokens(page_text: str) -> Tuple[List[int], List[int]]:
    lines = [_clean_line(x) for x in page_text.splitlines()]
    lines = [ln for ln in lines if ln and not _is_noise_line(ln)]

    result_tokens: List[int] = []
    draw_numbers: List[int] = []
    drawno_mode = False

    for ln in lines:
        ints = [int(x) for x in INT_RE.findall(ln)]
        if not ints:
            continue

        has_draw_no = any(x >= DRAWNO_MIN for x in ints)
        has_small_numbers = any(NUM_MIN <= x <= NUM_MAX for x in ints)

        if has_draw_no and not has_small_numbers:
            drawno_mode = True

        if not drawno_mode:
            for x in ints:
                if NUM_MIN <= x <= NUM_MAX:
                    result_tokens.append(x)
        else:
            for x in ints:
                if DRAWNO_MIN <= x < 100000000:
                    draw_numbers.append(x)

    return result_tokens, draw_numbers


def _chunk_tokens_to_draws(tokens: List[int]) -> List[List[int]]:
    if len(tokens) < DRAW_PICK_COUNT:
        return []

    best_draws = []
    best_score = -1

    for offset in range(DRAW_PICK_COUNT):
        sliced = tokens[offset:]
        usable = (len(sliced) // DRAW_PICK_COUNT) * DRAW_PICK_COUNT
        sliced = sliced[:usable]

        draws = []
        score = 0

        for i in range(0, len(sliced), DRAW_PICK_COUNT):
            draw = sorted(sliced[i:i + DRAW_PICK_COUNT])
            draws.append(draw)

            unique_ok = len(set(draw)) == DRAW_PICK_COUNT
            range_ok = all(NUM_MIN <= n <= NUM_MAX for n in draw)

            if unique_ok and range_ok:
                score += 3

        if score > best_score:
            best_score = score
            best_draws = draws

    validated = []
    for d in best_draws:
        if len(d) == DRAW_PICK_COUNT and len(set(d)) == DRAW_PICK_COUNT and all(NUM_MIN <= x <= NUM_MAX for x in d):
            validated.append(d)

    return validated


def _pair_draws_with_numbers(draws: List[List[int]], drawnos: List[int]) -> List[KenoRecord]:
    n = min(len(draws), len(drawnos))
    records: List[KenoRecord] = []

    for i in range(n):
        records.append(KenoRecord(
            draw_no=drawnos[i],
            nums=draws[i],
            date_str="—"
        ))

    for j in range(n, len(draws)):
        records.append(KenoRecord(
            draw_no=None,
            nums=draws[j],
            date_str="—"
        ))

    with_numbers = [r for r in records if r.draw_no is not None]
    if len(with_numbers) > 10:
        records.sort(key=lambda r: (r.draw_no is None, r.draw_no or -1), reverse=True)

    return records


@st.cache_data(show_spinner=False)
def load_keno_records_cached(pdf_bytes: bytes) -> List[KenoRecord]:
    _validate_pdf_bytes(pdf_bytes)
    pages = _read_pdf_pages_text(pdf_bytes)

    all_tokens: List[int] = []
    all_drawnos: List[int] = []

    for page_text in pages:
        tokens, drawnos = _split_page_tokens(page_text)
        all_tokens.extend(tokens)
        all_drawnos.extend(drawnos)

    draws = _chunk_tokens_to_draws(all_tokens)
    if not draws:
        raise RuntimeError("Nie udało się zbudować losowań Keno z PDF.")

    records = _pair_draws_with_numbers(draws, all_drawnos)

    if len(records) < 50:
        raise RuntimeError(f"Wyciągnięto zbyt mało rekordów: {len(records)}")

    return records


# =========================================================
# ANALYSIS
# =========================================================
@st.cache_data(show_spinner=False)
def compute_presence_df_cached(draws: List[List[int]]) -> pd.DataFrame:
    total = len(draws)
    counter = Counter()

    for draw in draws:
        for n in set(draw):
            counter[n] += 1

    rows = []
    for n in range(NUM_MIN, NUM_MAX + 1):
        hits = counter.get(n, 0)
        pct = (hits / total * 100.0) if total else 0.0
        rows.append({
            "Liczba": n,
            "Wystąpienia": hits,
            "Procent": pct
        })

    df = pd.DataFrame(rows).sort_values(
        ["Procent", "Wystąpienia", "Liczba"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def compute_pair_counter_cached(draws: List[List[int]]) -> Counter:
    pair_counter = Counter()
    for draw in draws:
        s = sorted(draw)
        for pair in combinations(s, 2):
            pair_counter[pair] += 1
    return pair_counter


@st.cache_data(show_spinner=False)
def compute_last_seen_map_cached(draws: List[List[int]]) -> Dict[int, int]:
    result = {}
    for n in range(NUM_MIN, NUM_MAX + 1):
        result[n] = 999999
        for idx, draw in enumerate(draws):
            if n in draw:
                result[n] = idx
                break
    return result


@st.cache_data(show_spinner=False)
def compute_gap_df_cached(draws: List[List[int]]) -> pd.DataFrame:
    chronological = list(reversed(draws))
    total = len(chronological)

    rows = []
    for num in range(NUM_MIN, NUM_MAX + 1):
        positions = []
        for idx, draw in enumerate(chronological):
            if num in draw:
                positions.append(idx)

        occurrences = len(positions)
        if occurrences >= 2:
            gaps = [b - a for a, b in zip(positions[:-1], positions[1:])]
            avg_gap = safe_mean(gaps, 0.0)
            current_gap = float((total - 1) - positions[-1])
            gap_ratio = (current_gap / avg_gap) if avg_gap > 0 else 0.0
        elif occurrences == 1:
            avg_gap = 0.0
            current_gap = float((total - 1) - positions[-1])
            gap_ratio = 0.0
        else:
            avg_gap = 0.0
            current_gap = float(total)
            gap_ratio = 0.0

        rows.append({
            "Liczba": num,
            "Wystąpienia": occurrences,
            "Średni_gap": round(avg_gap, 4),
            "Aktualny_gap": round(current_gap, 4),
            "Gap_ratio": round(gap_ratio, 4)
        })

    return pd.DataFrame(rows).sort_values(
        ["Gap_ratio", "Wystąpienia", "Liczba"],
        ascending=[False, False, True]
    ).reset_index(drop=True)


def build_window_freq_maps(draws: List[List[int]]) -> Dict[str, Dict[int, float]]:
    def make_map(sub_draws: List[List[int]]) -> Dict[int, float]:
        total = len(sub_draws)
        counter = Counter()
        for d in sub_draws:
            for n in set(d):
                counter[n] += 1
        return {
            n: (counter.get(n, 0) / total if total else 0.0)
            for n in range(NUM_MIN, NUM_MAX + 1)
        }

    return {
        "w30": make_map(draws[:min(FREQ_SHORT, len(draws))]),
        "w100": make_map(draws[:min(FREQ_MEDIUM, len(draws))]),
        "w300": make_map(draws[:min(FREQ_LONG, len(draws))]),
        "wall": make_map(draws),
    }


def build_feature_table(draws: List[List[int]]) -> pd.DataFrame:
    freq_maps = build_window_freq_maps(draws)
    gap_df = compute_gap_df_cached(draws)
    gap_map = dict(zip(gap_df["Liczba"], gap_df["Gap_ratio"]))
    last_seen_map = compute_last_seen_map_cached(draws)

    rows = []
    for n in range(NUM_MIN, NUM_MAX + 1):
        f30 = freq_maps["w30"].get(n, 0.0)
        f100 = freq_maps["w100"].get(n, 0.0)
        f300 = freq_maps["w300"].get(n, 0.0)
        fall = freq_maps["wall"].get(n, 0.0)
        last_seen = last_seen_map.get(n, 999999)
        gap_ratio = min(gap_map.get(n, 0.0), 3.0) / 3.0

        momentum = (
            f30 * 0.46 +
            f100 * 0.28 +
            f300 * 0.16 +
            fall * 0.10
        )

        if last_seen == 0:
            recency_penalty = 1.00
        elif last_seen == 1:
            recency_penalty = 0.65
        elif last_seen == 2:
            recency_penalty = 0.35
        else:
            recency_penalty = 0.0

        hot_strength = (
            momentum * 0.78 +
            fall * 0.22
        ) - recency_penalty * 0.10

        cold_strength = (
            (1.0 - fall) * 0.52 +
            (1.0 - f100) * 0.20 +
            (1.0 - f30) * 0.08 +
            gap_ratio * 0.20
        ) - recency_penalty * 0.04

        rows.append({
            "Liczba": n,
            "Freq30": f30,
            "Freq100": f100,
            "Freq300": f300,
            "FreqAll": fall,
            "GapRatio": gap_ratio,
            "LastSeenIdx": last_seen,
            "Momentum": momentum,
            "HotStrength": hot_strength,
            "ColdStrength": cold_strength,
        })

    df = pd.DataFrame(rows)

    for col in ["Freq30", "Freq100", "Freq300", "FreqAll", "GapRatio", "Momentum", "HotStrength", "ColdStrength"]:
        norm = normalize_score_map(dict(zip(df["Liczba"], df[col])))
        df[col + "_Norm"] = df["Liczba"].map(norm)

    return df


def build_hot_cold_groups(feature_df: pd.DataFrame, hot_size: int, cold_size: int) -> Tuple[List[int], List[int]]:
    hot_df = feature_df.sort_values(["HotStrength", "Liczba"], ascending=[False, True])
    cold_df = feature_df.sort_values(["ColdStrength", "Liczba"], ascending=[False, True])

    hot = hot_df.head(hot_size)["Liczba"].tolist()
    cold = cold_df.head(cold_size)["Liczba"].tolist()
    return hot, cold


# =========================================================
# PREDICTIVE MODE
# =========================================================
def _series_for_position(draws_newest_first: List[List[int]], pos_idx: int, window: int) -> List[int]:
    subset = draws_newest_first[:min(window, len(draws_newest_first))]
    chronological = list(reversed(subset))
    return [row[pos_idx] for row in chronological if len(row) == DRAW_PICK_COUNT]


def _linear_projection(series: List[int], recent_points: int = 8) -> float:
    if not series:
        return float(NUM_MIN)
    if len(series) == 1:
        return float(series[-1])

    use = series[-min(recent_points, len(series)):]
    x = np.arange(len(use), dtype=float)
    y = np.array(use, dtype=float)

    if len(use) < 2:
        return float(use[-1])

    a, b = np.polyfit(x, y, 1)
    return float(a * len(use) + b)


def _recent_delta_projection(series: List[int], max_deltas: int = 6) -> float:
    if not series:
        return float(NUM_MIN)
    if len(series) == 1:
        return float(series[-1])

    deltas = np.diff(series)
    if len(deltas) == 0:
        return float(series[-1])

    recent = deltas[-min(max_deltas, len(deltas)):]
    weights = np.linspace(1.0, 2.0, len(recent))
    step = float(np.average(recent, weights=weights))
    return float(series[-1] + step)


def _curvature_projection(series: List[int], max_points: int = 8) -> float:
    if not series:
        return float(NUM_MIN)
    if len(series) < 3:
        return _recent_delta_projection(series)

    use = series[-min(max_points, len(series)):]
    x = np.arange(len(use), dtype=float)
    y = np.array(use, dtype=float)

    if len(use) < 3:
        return _recent_delta_projection(series)

    coeff = np.polyfit(x, y, 2)
    poly = np.poly1d(coeff)
    return float(poly(len(use)))


def _pattern_projection(series: List[int], step_len: int = 4) -> Optional[float]:
    if len(series) < step_len + 6:
        return None

    target = np.diff(series[-(step_len + 1):]).tolist()
    candidates = []

    for end_idx in range(step_len, len(series) - 1):
        test = np.diff(series[end_idx - step_len:end_idx + 1]).tolist()
        if len(test) != len(target):
            continue

        distance = sum(abs(a - b) for a, b in zip(target, test))
        if distance <= 5:
            candidates.append((distance, series[end_idx + 1]))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    vals = [v for _, v in candidates[:10]]
    return float(sum(vals) / len(vals))


def _smooth_prediction(preds: List[Optional[float]], fallback: float) -> float:
    clean = [p for p in preds if p is not None and not np.isnan(p)]
    if not clean:
        return fallback
    return float(sum(clean) / len(clean))


def _enforce_sorted_unique_20(raw_values: List[int]) -> List[int]:
    vals = sorted([clamp(v, NUM_MIN, NUM_MAX) for v in raw_values])

    fixed = []
    used = set()

    for v in vals:
        if v not in used:
            fixed.append(v)
            used.add(v)
        else:
            for c in range(NUM_MIN, NUM_MAX + 1):
                if c not in used:
                    fixed.append(c)
                    used.add(c)
                    break

    fixed = sorted(fixed)

    if len(fixed) < DRAW_PICK_COUNT:
        for c in range(NUM_MIN, NUM_MAX + 1):
            if c not in used:
                fixed.append(c)
                used.add(c)
            if len(fixed) == DRAW_PICK_COUNT:
                break

    return sorted(fixed[:DRAW_PICK_COUNT])


def build_predictive_draw(draws_newest_first: List[List[int]], window: int) -> Dict:
    use_window = min(window, len(draws_newest_first))
    details = []
    raw_forecast = []

    for pos in range(DRAW_PICK_COUNT):
        series = _series_for_position(draws_newest_first, pos, use_window)

        if not series:
            pred = pos + 1
            details.append({
                "Pozycja": pos + 1,
                "Ostatnia": pred,
                "Prognoza": pred,
                "Metoda": "fallback"
            })
            raw_forecast.append(pred)
            continue

        last_val = float(series[-1])

        p1 = _linear_projection(series, recent_points=8)
        p2 = _recent_delta_projection(series, max_deltas=6)
        p3 = _curvature_projection(series, max_points=8)
        p4 = _pattern_projection(series, step_len=4)

        blended = _smooth_prediction([p1, p2, p3, p4], fallback=last_val)
        pred = int(round(blended))
        pred = clamp(pred, NUM_MIN, NUM_MAX)

        methods = ["linia", "delta", "krzywizna"]
        if p4 is not None:
            methods.append("wzorzec")

        details.append({
            "Pozycja": pos + 1,
            "Ostatnia": int(series[-1]),
            "Prognoza": pred,
            "Metoda": ", ".join(methods)
        })
        raw_forecast.append(pred)

    final_set = _enforce_sorted_unique_20(raw_forecast)

    for i in range(DRAW_PICK_COUNT):
        details[i]["Po_korekcie"] = final_set[i]

    return {
        "set20": final_set,
        "details": details,
        "window_used": use_window
    }


# =========================================================
# TICKET SCORING
# =========================================================
def build_pair_strength_map(pair_counter: Counter) -> Dict[Tuple[int, int], float]:
    if not pair_counter:
        return {}
    max_pair = max(pair_counter.values())
    return {k: v / max_pair for k, v in pair_counter.items()}


def score_ticket(
    ticket: List[int],
    feature_df: pd.DataFrame,
    pair_strength_map: Dict[Tuple[int, int], float],
    recent_draws: List[List[int]],
    mode: str,
    source: str
) -> TicketScore:
    s = sorted(ticket)
    fmap = feature_df.set_index("Liczba").to_dict("index")

    hot_score = safe_mean([fmap[n]["HotStrength_Norm"] for n in s], 0.0)
    cold_score = safe_mean([fmap[n]["ColdStrength_Norm"] for n in s], 0.0)
    momentum_score = safe_mean([fmap[n]["Momentum_Norm"] for n in s], 0.0)
    overdue_score = safe_mean([fmap[n]["GapRatio_Norm"] for n in s], 0.0)

    pair_score = safe_mean(
        [pair_strength_map.get(tuple(pair), 0.0) for pair in combinations(s, 2)],
        0.0
    )

    recent_overlap = similarity_to_recent(s, recent_draws)
    recent_penalty = 0.0
    if recent_overlap >= max(2, len(ticket) // 2):
        recent_penalty = recent_overlap * 0.18
    else:
        recent_penalty = recent_overlap * 0.05

    if mode == "hot":
        final_score = (
            hot_score * 2.25 +
            momentum_score * 1.45 +
            overdue_score * 0.55 +
            pair_score * 0.75
            - recent_penalty
        )
    elif mode == "cold":
        final_score = (
            cold_score * 2.10 +
            overdue_score * 1.10 +
            momentum_score * 0.35 +
            pair_score * 0.40
            - recent_penalty
        )
    elif mode == "half":
        final_score = (
            hot_score * 1.20 +
            cold_score * 1.20 +
            momentum_score * 0.80 +
            overdue_score * 0.60 +
            pair_score * 0.55
            - recent_penalty
        )
    else:  # balanced or custom
        final_score = (
            hot_score * 1.35 +
            cold_score * 0.75 +
            momentum_score * 1.05 +
            overdue_score * 0.65 +
            pair_score * 0.65
            - recent_penalty
        )

    return TicketScore(
        ticket=s,
        final_score=round(final_score, 6),
        hot_score=round(hot_score, 6),
        cold_score=round(cold_score, 6),
        momentum_score=round(momentum_score, 6),
        overdue_score=round(overdue_score, 6),
        pair_score=round(pair_score, 6),
        recent_penalty=round(recent_penalty, 6),
        source=source
    )


# =========================================================
# GENERATION
# =========================================================
def build_generation_weights(feature_df: pd.DataFrame, mode: str) -> Dict[int, float]:
    fmap = feature_df.set_index("Liczba").to_dict("index")
    weights = {}

    for n in range(NUM_MIN, NUM_MAX + 1):
        row = fmap[n]

        if mode == "hot":
            w = (
                row["HotStrength_Norm"] * 0.62 +
                row["Momentum_Norm"] * 0.22 +
                row["Freq100_Norm"] * 0.10 +
                row["FreqAll_Norm"] * 0.06
            )
        elif mode == "cold":
            w = (
                row["ColdStrength_Norm"] * 0.60 +
                row["GapRatio_Norm"] * 0.20 +
                (1.0 - row["FreqAll_Norm"]) * 0.20
            )
        elif mode == "half":
            w = (
                row["HotStrength_Norm"] * 0.40 +
                row["ColdStrength_Norm"] * 0.40 +
                row["Momentum_Norm"] * 0.10 +
                row["GapRatio_Norm"] * 0.10
            )
        else:
            w = (
                row["HotStrength_Norm"] * 0.45 +
                row["ColdStrength_Norm"] * 0.20 +
                row["Momentum_Norm"] * 0.20 +
                row["GapRatio_Norm"] * 0.15
            )

        weights[n] = max(w, 0.0001)

    return weights


def generate_totally_random_ticket(rng: np.random.Generator, pick_count: int) -> Tuple[List[int], str]:
    population = list(range(NUM_MIN, NUM_MAX + 1))
    ticket = sorted(rng.choice(population, size=pick_count, replace=False).tolist())
    return ticket, "total_random"


def generate_candidate_from_groups(
    rng: np.random.Generator,
    mode: str,
    pick_count: int,
    hot_group: List[int],
    cold_group: List[int],
    generation_weights: Dict[int, float],
    custom_hot_count: int = 0,
    custom_cold_count: int = 0
) -> Tuple[List[int], str]:
    full_pool = list(range(NUM_MIN, NUM_MAX + 1))

    if mode == "hot":
        pool = hot_group if len(hot_group) >= pick_count else full_pool
        weights = [generation_weights[n] for n in pool]
        ticket = weighted_unique_sample(rng, pool, weights, pick_count)
        return ticket, "hot_only"

    if mode == "cold":
        pool = cold_group if len(cold_group) >= pick_count else full_pool
        weights = [generation_weights[n] for n in pool]
        ticket = weighted_unique_sample(rng, pool, weights, pick_count)
        return ticket, "cold_only"

    if mode == "half":
        hot_take = pick_count // 2
        cold_take = pick_count - hot_take

        hot_pool = hot_group[:] if hot_group else full_pool
        cold_pool = [x for x in cold_group if x not in hot_pool]
        if len(cold_pool) < cold_take:
            cold_pool = [x for x in full_pool if x not in hot_pool]

        hot_weights = [generation_weights[n] for n in hot_pool]
        h = weighted_unique_sample(rng, hot_pool, hot_weights, min(hot_take, len(hot_pool)))

        rem_pool = [x for x in cold_pool if x not in h]
        if len(rem_pool) < cold_take:
            rem_pool = [x for x in full_pool if x not in h]

        rem_weights = [generation_weights[n] for n in rem_pool]
        c = weighted_unique_sample(rng, rem_pool, rem_weights, cold_take)

        return sorted(h + c), "half_hot_half_cold"

    if mode == "custom":
        hot_take = clamp(custom_hot_count, 0, pick_count)
        cold_take = clamp(custom_cold_count, 0, pick_count - hot_take)
        rest_take = pick_count - hot_take - cold_take

        hot_pool = hot_group[:] if hot_group else full_pool
        hot_weights = [generation_weights[n] for n in hot_pool]
        h = weighted_unique_sample(rng, hot_pool, hot_weights, hot_take) if hot_take > 0 else []

        cold_pool = [x for x in cold_group if x not in h]
        if len(cold_pool) < cold_take:
            cold_pool = [x for x in full_pool if x not in h]
        cold_weights = [generation_weights[n] for n in cold_pool]
        c = weighted_unique_sample(rng, cold_pool, cold_weights, cold_take) if cold_take > 0 else []

        used = set(h + c)
        rest_pool = [x for x in full_pool if x not in used]
        rest_weights = [generation_weights[n] for n in rest_pool]
        r = weighted_unique_sample(rng, rest_pool, rest_weights, rest_take) if rest_take > 0 else []

        return sorted(h + c + r), "custom_mix"

    # balanced
    hot_take = max(1, pick_count // 2)
    cold_take = max(1, pick_count // 3)
    if hot_take + cold_take > pick_count:
        cold_take = max(1, pick_count - hot_take)

    hot_pool = hot_group[:] if hot_group else full_pool
    hot_weights = [generation_weights[n] for n in hot_pool]
    h = weighted_unique_sample(rng, hot_pool, hot_weights, min(hot_take, len(hot_pool)))

    cold_pool = [x for x in cold_group if x not in h]
    if len(cold_pool) < cold_take:
        cold_pool = [x for x in full_pool if x not in h]
    cold_weights = [generation_weights[n] for n in cold_pool]
    c = weighted_unique_sample(rng, cold_pool, cold_weights, cold_take)

    remaining = pick_count - len(h) - len(c)
    used = set(h + c)
    rem_pool = [x for x in full_pool if x not in used]
    rem_weights = [generation_weights[n] for n in rem_pool]
    r = weighted_unique_sample(rng, rem_pool, rem_weights, remaining) if remaining > 0 else []

    return sorted(h + c + r), "balanced"


def mutate_ticket(
    rng: np.random.Generator,
    ticket: List[int],
    generation_weights: Dict[int, float],
    replace_count: int
) -> List[int]:
    base = sorted(ticket)
    replace_count = min(replace_count, len(base))
    to_remove = set(rng.choice(base, size=replace_count, replace=False).tolist())
    kept = [x for x in base if x not in to_remove]

    candidates = [n for n in range(NUM_MIN, NUM_MAX + 1) if n not in kept]
    weights = [generation_weights[n] for n in candidates]
    added = weighted_unique_sample(rng, candidates, weights, len(base) - len(kept))

    return sorted(kept + added)


def build_ranked_tickets(
    draws: List[List[int]],
    feature_df: pd.DataFrame,
    build_source: str,
    build_mode: str,
    pick_count: int,
    candidate_count: int,
    final_count: int,
    hot_group: List[int],
    cold_group: List[int],
    seed: int,
    custom_hot_count: int = 0,
    custom_cold_count: int = 0
) -> List[TicketScore]:
    rng = np.random.default_rng(seed)
    generation_weights = build_generation_weights(feature_df, mode=build_mode if build_mode in ["hot", "cold", "half", "balanced"] else "balanced")
    pair_counter = compute_pair_counter_cached(draws)
    pair_strength_map = build_pair_strength_map(pair_counter)
    recent_draws = draws[:10]

    seen = set()
    scored: List[TicketScore] = []

    attempts = 0
    max_attempts = candidate_count * 5

    while len(scored) < candidate_count and attempts < max_attempts:
        attempts += 1

        if build_source == "random":
            ticket, source = generate_totally_random_ticket(rng, pick_count)
            mode_for_score = "balanced"
        else:
            if rng.random() < 0.78:
                ticket, source = generate_candidate_from_groups(
                    rng=rng,
                    mode=build_mode,
                    pick_count=pick_count,
                    hot_group=hot_group,
                    cold_group=cold_group,
                    generation_weights=generation_weights,
                    custom_hot_count=custom_hot_count,
                    custom_cold_count=custom_cold_count
                )
            else:
                seed_ticket, _ = generate_candidate_from_groups(
                    rng=rng,
                    mode=build_mode,
                    pick_count=pick_count,
                    hot_group=hot_group,
                    cold_group=cold_group,
                    generation_weights=generation_weights,
                    custom_hot_count=custom_hot_count,
                    custom_cold_count=custom_cold_count
                )
                ticket = mutate_ticket(rng, seed_ticket, generation_weights, replace_count=int(rng.choice([1, 2])))
                source = "mutation"

            mode_for_score = build_mode if build_mode in ["hot", "cold", "half"] else "balanced"

        key = tuple(ticket)
        if key in seen:
            continue
        seen.add(key)

        scored.append(score_ticket(
            ticket=ticket,
            feature_df=feature_df,
            pair_strength_map=pair_strength_map,
            recent_draws=recent_draws,
            mode=mode_for_score,
            source=source
        ))

    scored.sort(key=lambda x: x.final_score, reverse=True)
    return scored[:final_count]


# =========================================================
# EXPORTS
# =========================================================
def make_txt_for_results(records: List[KenoRecord]) -> bytes:
    lines = []
    for r in records:
        draw_no = str(r.draw_no) if r.draw_no is not None else "—"
        nums = " ".join(f"{x:02d}" for x in r.nums)
        lines.append(f"Losowanie: {draw_no} | Wynik: {nums}")
    return lines_to_bytes(lines)


def make_txt_for_tickets(tickets: List[TicketScore]) -> bytes:
    lines = []
    for i, t in enumerate(tickets, start=1):
        nums = " ".join(f"{x:02d}" for x in t.ticket)
        lines.append(
            f"#{i:03d} | {nums} | Final={t.final_score:.6f} | "
            f"Hot={t.hot_score:.6f} | Cold={t.cold_score:.6f} | "
            f"Momentum={t.momentum_score:.6f} | Overdue={t.overdue_score:.6f} | "
            f"Pair={t.pair_score:.6f} | RecentPenalty={t.recent_penalty:.6f} | "
            f"Source={t.source}"
        )
    return lines_to_bytes(lines)


def make_txt_for_single_set(title: str, nums: List[int]) -> bytes:
    return lines_to_bytes([title, " ".join(f"{x:02d}" for x in nums)])


def make_txt_for_predictive(pred_data: Dict) -> bytes:
    lines = [
        "TRYB PRZEWIDUJĄCY — pełne 20 liczb",
        " ".join(f"{x:02d}" for x in pred_data["set20"]),
        "",
        f"Zakres użyty: {pred_data['window_used']}",
        "",
        "Szczegóły pozycji:"
    ]
    for d in pred_data["details"]:
        lines.append(
            f"Pozycja {d['Pozycja']}: ostatnia={d['Ostatnia']}, prognoza={d['Prognoza']}, "
            f"po korekcie={d['Po_korekcie']}, metoda={d['Metoda']}"
        )
    return lines_to_bytes(lines)


# =========================================================
# UI HELPERS
# =========================================================
def render_ticket_card(idx: int, t: TicketScore) -> None:
    nums = " ".join(f"{x:02d}" for x in t.ticket)
    st.markdown(
        f"""
<div class="rank-card">
  <div class="rank-title">Zestaw #{idx:03d}</div>
  <div class="rank-main">{nums}</div>
  <div class="rank-meta">
    Final Score: <b>{t.final_score:.4f}</b><br>
    Hot: <b>{t.hot_score:.4f}</b> | Cold: <b>{t.cold_score:.4f}</b> | Momentum: <b>{t.momentum_score:.4f}</b><br>
    Overdue: <b>{t.overdue_score:.4f}</b> | Pair: <b>{t.pair_score:.4f}</b> | Kara recent: <b>{t.recent_penalty:.4f}</b><br>
    Źródło: <b>{t.source}</b>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )


def session_init() -> None:
    defaults = {
        "show_results": False,
        "ranked_tickets": None,
        "predictive_result": None,
        "hot_group": None,
        "cold_group": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_help_sections() -> None:
    with st.expander("📘 Opisy wszystkich funkcji i jak najlepiej je ustawiać", expanded=False):
        st.markdown("""
### 1. Źródło budowy zestawu
**Losowy**  
Aplikacja nie używa PDF i buduje zestawy całkowicie losowo.  
**Najlepsze użycie:** kiedy chcesz czysty los bez analizy historii.

**Na bazie pliku PDF**  
Aplikacja czyta historię z `keno.PDF` / `kino.PDF` i buduje zestawy na podstawie tej bazy.  
**Najlepsze użycie:** to jest główny tryb do pracy analitycznej.

---

### 2. Sposób budowy zestawu z PDF
**Gorące**  
Buduje zestawy głównie z liczb, które najczęściej i najmocniej wypadały w historii.  
**Najlepiej ustawić:** gdy chcesz iść w liczby aktualnie aktywne.

**Zimne**  
Buduje zestawy z liczb rzadziej występujących, dodatkowo patrzy na przerwy.  
**Najlepiej ustawić:** gdy chcesz grać odwrotnie niż mainstream, bardziej pod rzadkie liczby.

**Pół na pół**  
Połowa zestawu z gorących, połowa z zimnych.  
**Najlepiej ustawić:** gdy chcesz kompromis.

**Zbalansowany**  
Miesza gorące, zimne, momentum i rozkład ogólny.  
**Najlepiej ustawić:** jako domyślny tryb dla większości testów.

**Własne proporcje**  
Sam wybierasz ile liczb ma pochodzić z gorących i ile z zimnych.  
**Najlepiej ustawić:** gdy chcesz eksperymentować.

---

### 3. Tryb przewidujący
Ten tryb patrzy na każdą z 20 pozycji osobno w całej historii i próbuje przewidzieć,
jak może ułożyć się kolejna wartość tej pozycji.  
Czyli:
- pozycja 1 analizowana osobno,
- pozycja 2 analizowana osobno,
- ...
- pozycja 20 analizowana osobno.

Na końcu aplikacja składa z tego pełen przewidywany układ 20 liczb.

**Najlepiej ustawić:** na większym zakresie historii, np. 300–1000 losowań, jeśli chcesz używać trybu wizualno-pozycyjnego.

---

### 4. Wielkość zestawu
Możesz wybrać np. 2–10 liczb.  
**Najlepiej ustawić:** zgodnie z tym, ile liczb chcesz grać w Keno.  
Jeżeli chcesz najmocniejszy wybór jakościowy, często dobrze sprawdza się zakres 5–8 liczb.

---

### 5. Pula gorących i pula zimnych
To liczba topowych liczb, z których aplikacja buduje kandydatów.

**Przykład:**
- pula gorących = 20 → bierze 20 najmocniejszych hotów,
- pula zimnych = 20 → bierze 20 najmocniejszych coldów.

**Najlepiej ustawić:**
- bardziej agresywnie: 12–18,
- bardziej szeroko i spokojnie: 20–30.

---

### 6. Ilu kandydatów ocenić
Im więcej kandydatów, tym lepiej silnik może wybrać końcowe zestawy, ale tym wolniej działa.

**Najlepiej ustawić:**
- szybka praca: 1000–2000,
- solidna analiza: 3000–5000,
- mocny test: 5000+.

---

### 7. Seed
To ziarno losowości.  
Ten sam seed = te same wyniki przy tych samych ustawieniach.

**Najlepiej ustawić:** stały seed, jeśli chcesz porównywać ustawienia.  
Zmieniać seed, jeśli chcesz nowe warianty.

---

### 8. Ważna uczciwa uwaga
Aplikacja robi zaawansowaną analizę historii i buduje zestawy logiczniej niż zwykłe losowanie,
ale nie daje gwarancji wygranej, bo Keno nadal jest grą losową.
        """)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    st.set_page_config(
        page_title="Keno Ultra Master",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.title(APP_TITLE)
    st.write(
        "Przyjazna dla użytkownika aplikacja do budowy zestawów Keno 20/70: "
        "**losowo**, **na bazie historii z PDF** albo w **trybie przewidującym**."
    )

    session_init()
    render_help_sections()

    pdf_path = resolve_pdf_path()

    st.markdown('<div class="v-card">', unsafe_allow_html=True)
    st.subheader("📄 Dane wejściowe")
    st.write(f"Szukanie pliku: `{pdf_path}`")
    st.write("Obsługiwane nazwy: `keno.PDF`, `keno.pdf`, `kino.PDF`, `kino.pdf`")
    st.write("Silnik odczytu PDF: **PyMuPDF (fitz)**")
    st.markdown(
        '<div class="v-muted">Plik powinien mieć układ jak w Twoim PDF: rzędy po 20 liczb i na dole numery losowań.</div>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    pdf_records = None
    if pdf_path.exists():
        try:
            pdf_bytes = pdf_path.read_bytes()
            pdf_records = load_keno_records_cached(pdf_bytes)
        except Exception as e:
            logger.exception("Błąd odczytu PDF")
            st.error("❌ Nie udało się odczytać PDF.")
            st.code(str(e))
    else:
        st.warning("⚠️ Nie znaleziono pliku PDF. Tryb losowy nadal będzie działał.")

    max_records = len(pdf_records) if pdf_records is not None else 0

    st.markdown('<div class="v-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Ustawienia główne")

    source_mode = st.selectbox(
        "Skąd budować zestawy?",
        ["PDF - analiza historii", "Losowy zestaw bez użycia PDF", "Tryb przewidujący z PDF"],
        index=0,
        help="Wybierz czy aplikacja ma używać historii z pliku, całkowicie zignorować plik i losować, czy wejść w tryb przewidujący po pozycjach."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        ticket_size = st.selectbox(
            "Ile liczb ma mieć końcowy zestaw?",
            [2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=4,
            help="Dla większości testów dobry zakres to 5–8."
        )
    with c2:
        final_count = st.slider(
            "Ile finalnych zestawów pokazać?",
            1, 50, 10, 1,
            help="Im więcej zestawów pokażesz, tym więcej wariantów dostaniesz."
        )
    with c3:
        seed = st.number_input(
            "Seed",
            min_value=1,
            max_value=999999999,
            value=DEFAULT_SEED,
            step=1,
            help="Ten sam seed i te same ustawienia dadzą te same wyniki."
        )

    if source_mode != "Losowy zestaw bez użycia PDF":
        default_window_options = [100, 250, 500, 750, 1000]
        if max_records > 0 and max_records not in default_window_options:
            default_window_options.append(max_records)
        default_window_options = sorted(set(default_window_options))

        analysis_window = st.selectbox(
            "Ile ostatnich losowań analizować?",
            options=default_window_options if default_window_options else [100],
            index=min(len(default_window_options) - 1, 2) if default_window_options else 0,
            help="Większy zakres daje stabilniejszy obraz. Dla trybu przewidującego dobrze brać większe zakresy."
        )
    else:
        analysis_window = 100

    if source_mode == "PDF - analiza historii":
        build_mode = st.selectbox(
            "Jak budować zestaw z danych PDF?",
            ["hot", "cold", "half", "balanced", "custom"],
            index=3,
            help="balanced to zwykle najbezpieczniejszy start. hot daje bardziej aktywne liczby, cold bardziej rzadkie."
        )
    else:
        build_mode = "balanced"

    if source_mode == "PDF - analiza historii":
        c4, c5 = st.columns(2)
        with c4:
            hot_pool_size = st.slider(
                "Pula gorących liczb",
                10, 35, 20, 1,
                help="Mniejsza pula = bardziej agresywna selekcja. Większa = bardziej szeroki wybór."
            )
        with c5:
            cold_pool_size = st.slider(
                "Pula zimnych liczb",
                10, 35, 20, 1,
                help="Mniejsza pula = bardziej agresywne zimne liczby. Większa = łagodniejsze podejście."
            )

        c6, c7 = st.columns(2)
        with c6:
            candidate_count = st.slider(
                "Ilu kandydatów ocenić?",
                200, 10000, DEFAULT_CANDIDATES, 100,
                help="3000–5000 to dobry zakres na mocniejszą analizę."
            )
        with c7:
            custom_hot_count = 0
            custom_cold_count = 0
            if build_mode == "custom":
                custom_hot_count = st.slider(
                    "Ile liczb ma pochodzić z gorących?",
                    0, ticket_size, max(1, ticket_size // 2), 1,
                    help="Ustaw ile z końcowego zestawu ma pochodzić z grupy hot."
                )
                custom_cold_count = st.slider(
                    "Ile liczb ma pochodzić z zimnych?",
                    0, ticket_size - custom_hot_count, max(0, ticket_size // 3), 1,
                    help="Ustaw ile z końcowego zestawu ma pochodzić z grupy cold."
                )
            else:
                st.markdown(
                    '<div class="tip-box"><b>Podpowiedź:</b> jeśli nie wiesz co wybrać, zacznij od <b>balanced</b>, pula hot = 20, pula cold = 20, kandydaci = 3000.</div>',
                    unsafe_allow_html=True
                )
    else:
        hot_pool_size = 20
        cold_pool_size = 20
        candidate_count = DEFAULT_CANDIDATES
        custom_hot_count = 0
        custom_cold_count = 0

    st.markdown("</div>", unsafe_allow_html=True)

    if source_mode == "PDF - analiza historii" and pdf_records is None:
        st.error("❌ Wybrałeś tryb PDF, ale plik nie został poprawnie wczytany.")
        st.stop()

    if source_mode == "Tryb przewidujący z PDF" and pdf_records is None:
        st.error("❌ Wybrałeś tryb przewidujący, ale plik nie został poprawnie wczytany.")
        st.stop()

    if pdf_records is not None:
        used_window = min(int(analysis_window), len(pdf_records))
        used_records = pdf_records[:used_window]
        draws = [r.nums for r in used_records]
        feature_df = build_feature_table(draws)
        hot_group, cold_group = build_hot_cold_groups(
            feature_df=feature_df,
            hot_size=hot_pool_size,
            cold_size=cold_pool_size
        )
    else:
        used_window = 0
        used_records = []
        draws = []
        feature_df = None
        hot_group = []
        cold_group = []

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="v-card">', unsafe_allow_html=True)
        st.subheader("📊 Tabela analityczna liczb")
        if feature_df is not None:
            display_df = feature_df.copy()
            for col in ["Freq30", "Freq100", "Freq300", "FreqAll", "GapRatio", "Momentum", "HotStrength", "ColdStrength"]:
                display_df[col] = display_df[col].map(lambda x: round(float(x), 6))
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("W trybie losowym tabela analityczna nie jest potrzebna, bo zestawy nie używają PDF.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="v-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ Diagnostyka")
        st.write(f"Tryb pracy: **{source_mode}**")
        st.write(f"Wielkość zestawu: **{ticket_size}**")
        st.write(f"Ilość finalnych zestawów: **{final_count}**")
        if pdf_records is not None:
            st.write(f"Rekordów w PDF: **{len(pdf_records)}**")
            st.write(f"Analizowanych losowań: **{used_window}**")
        st.markdown("</div>", unsafe_allow_html=True)

        if feature_df is not None:
            st.markdown('<div class="v-card">', unsafe_allow_html=True)
            st.subheader("🔥 Gorące liczby")
            st.markdown(" ".join([f'<span class="v-pill">{n:02d}</span>' for n in hot_group]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="v-card">', unsafe_allow_html=True)
            st.subheader("❄️ Zimne liczby")
            st.markdown(" ".join([f'<span class="v-pill-cold">{n:02d}</span>' for n in cold_group]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="v-card">', unsafe_allow_html=True)
    st.subheader("🎟️ Narzędzia")

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        generate_btn = st.button("🎯 GENERUJ ZESTAWY", type="primary", use_container_width=True)
    with c2:
        show_results_btn = st.button("📋 POKAŻ WYNIKI PDF", type="primary", use_container_width=True)
    with c3:
        show_groups_btn = st.button("🔥/❄️ POKAŻ GOTOWE HOT/COLD", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if show_results_btn:
        st.session_state["show_results"] = not st.session_state["show_results"]

    if show_groups_btn and feature_df is not None:
        st.session_state["hot_group"] = hot_group
        st.session_state["cold_group"] = cold_group

    if generate_btn:
        if source_mode == "Tryb przewidujący z PDF":
            with st.spinner("Buduję pełny przewidywany układ 20 liczb na podstawie pozycji w historii..."):
                st.session_state["predictive_result"] = build_predictive_draw(
                    draws_newest_first=[r.nums for r in pdf_records],
                    window=used_window
                )
                st.session_state["ranked_tickets"] = None
        else:
            with st.spinner("Buduję kandydatów i wybieram najlepsze zestawy..."):
                st.session_state["ranked_tickets"] = build_ranked_tickets(
                    draws=draws if draws else [[x for x in range(1, 21)]],
                    feature_df=feature_df if feature_df is not None else pd.DataFrame({
                        "Liczba": list(range(NUM_MIN, NUM_MAX + 1)),
                        "Freq30_Norm": [0.5] * 70,
                        "Freq100_Norm": [0.5] * 70,
                        "FreqAll_Norm": [0.5] * 70,
                        "Momentum_Norm": [0.5] * 70,
                        "GapRatio_Norm": [0.5] * 70,
                        "HotStrength_Norm": [0.5] * 70,
                        "ColdStrength_Norm": [0.5] * 70,
                    }),
                    build_source="random" if source_mode == "Losowy zestaw bez użycia PDF" else "pdf",
                    build_mode=build_mode,
                    pick_count=ticket_size,
                    candidate_count=candidate_count,
                    final_count=final_count,
                    hot_group=hot_group,
                    cold_group=cold_group,
                    seed=int(seed),
                    custom_hot_count=custom_hot_count,
                    custom_cold_count=custom_cold_count
                )
                st.session_state["predictive_result"] = None

    # =====================================================
    # OUTPUTS
    # =====================================================
    if st.session_state["show_results"] and pdf_records is not None:
        st.markdown("### 📋 Ostatnie wyniki Keno z PDF")
        show_n = st.selectbox("Ile wyników pokazać?", [10, 25, 50, 100], index=2)
        view_records = pdf_records[:min(show_n, len(pdf_records))]
        df_results = pd.DataFrame({
            "Numer losowania": [r.draw_no if r.draw_no is not None else "—" for r in view_records],
            "Wynik": [" ".join(f"{x:02d}" for x in r.nums) for r in view_records],
        })
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        result_name = sanitize_txt_filename(st.text_input("Nazwa pliku wyników .txt", value="keno_wyniki.txt"))
        st.download_button(
            "⬇️ Pobierz wyniki jako TXT",
            data=make_txt_for_results(view_records),
            file_name=result_name,
            mime="text/plain",
            use_container_width=True
        )

    if st.session_state["hot_group"] is not None and st.session_state["cold_group"] is not None:
        hot_nums = st.session_state["hot_group"][:ticket_size]
        cold_nums = st.session_state["cold_group"][:ticket_size]

        st.markdown("### 🔥 / ❄️ Szybkie gotowe zestawy")
        st.markdown(f"**HOT {ticket_size}:** {' '.join(f'{x:02d}' for x in hot_nums)}")
        st.markdown(f"**COLD {ticket_size}:** {' '.join(f'{x:02d}' for x in cold_nums)}")

        hot_name = sanitize_txt_filename(st.text_input("Nazwa pliku HOT .txt", value="keno_hot.txt"))
        cold_name = sanitize_txt_filename(st.text_input("Nazwa pliku COLD .txt", value="keno_cold.txt"))

        st.download_button(
            "⬇️ Pobierz HOT jako TXT",
            data=make_txt_for_single_set(f"Keno HOT {ticket_size}", hot_nums),
            file_name=hot_name,
            mime="text/plain",
            use_container_width=True
        )
        st.download_button(
            "⬇️ Pobierz COLD jako TXT",
            data=make_txt_for_single_set(f"Keno COLD {ticket_size}", cold_nums),
            file_name=cold_name,
            mime="text/plain",
            use_container_width=True
        )

    predictive = st.session_state.get("predictive_result")
    if predictive is not None:
        st.markdown("### 🌟 Tryb przewidujący — pełny układ 20 liczb")
        st.markdown(
            " ".join([f'<span class="v-pill-predict">{x:02d}</span>' for x in predictive["set20"]]),
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="tip-box"><b>Zakres użyty:</b> {predictive["window_used"]} losowań. '
            'To jest pełny przewidywany układ 20 liczb zbudowany po analizie każdej pozycji osobno.</div>',
            unsafe_allow_html=True
        )
        with st.expander("Pokaż szczegóły każdej pozycji 1–20"):
            st.dataframe(pd.DataFrame(predictive["details"]), use_container_width=True, hide_index=True)

        predict_name = sanitize_txt_filename(st.text_input("Nazwa pliku trybu przewidującego .txt", value="keno_predictive.txt"))
        st.download_button(
            "⬇️ Pobierz tryb przewidujący jako TXT",
            data=make_txt_for_predictive(predictive),
            file_name=predict_name,
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("### ✂️ Przytnij przewidywany układ do wielkości swojego zestawu")
        cut_predict = predictive["set20"][:ticket_size]
        st.markdown(f"**Predict {ticket_size}:** {' '.join(f'{x:02d}' for x in cut_predict)}")

    ranked = st.session_state.get("ranked_tickets")
    if ranked:
        st.markdown("### 🎯 Najlepsze zestawy")
        for idx, t in enumerate(ranked, start=1):
            render_ticket_card(idx, t)

        with st.expander("Pokaż pełną tabelę rankingową"):
            df_ranked = pd.DataFrame([
                {
                    "Zestaw": " ".join(f"{x:02d}" for x in t.ticket),
                    "Final Score": t.final_score,
                    "Hot": t.hot_score,
                    "Cold": t.cold_score,
                    "Momentum": t.momentum_score,
                    "Overdue": t.overdue_score,
                    "Pair": t.pair_score,
                    "RecentPenalty": t.recent_penalty,
                    "Source": t.source,
                }
                for t in ranked
            ])
            st.dataframe(df_ranked, use_container_width=True, hide_index=True)

        tickets_name = sanitize_txt_filename(st.text_input("Nazwa pliku zestawów .txt", value="keno_zestawy.txt"))
        st.download_button(
            "⬇️ Pobierz zestawy jako TXT",
            data=make_txt_for_tickets(ranked),
            file_name=tickets_name,
            mime="text/plain",
            use_container_width=True
        )

    if pdf_records is not None:
        with st.expander("✅ Kontrola odczytu pierwszych 10 rekordów"):
            for i, r in enumerate(pdf_records[:10], start=1):
                st.write(f"{i}. Losowanie: {r.draw_no} | Wynik: {' '.join(f'{x:02d}' for x in r.nums)}")

    with st.expander("📌 Szybkie rekomendacje ustawień"):
        st.markdown("""
**Najbezpieczniejszy start:**  
- Źródło: **PDF - analiza historii**  
- Budowa: **balanced**  
- Wielkość zestawu: **5–8**  
- Pula gorących: **20**  
- Pula zimnych: **20**  
- Kandydaci: **3000**  

**Gdy chcesz agresywnie iść w aktywne liczby:**  
- Budowa: **hot**  
- Pula gorących: **12–18**  

**Gdy chcesz bardziej eksperymentalnie:**  
- Budowa: **cold** albo **custom**  

**Gdy chcesz testować układ wizualny całego pliku:**  
- Tryb: **Tryb przewidujący z PDF**  
- Zakres historii: **500–1000**  
        """)

    with st.expander("⚠️ Ważna uczciwa informacja"):
        st.markdown(
            '<div class="warn-box">Ta aplikacja analizuje historię, buduje rankingi i przewidywania pozycyjne, '
            'ale nie daje gwarancji wygranej. To narzędzie do mocniejszego i bardziej świadomego budowania zestawów, '
            'a nie pewny sposób na pokonanie losowości.</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()

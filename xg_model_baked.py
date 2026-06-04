import numpy as np
import pandas as pd

# --- frozen MLS-trained xG model -------------------------------------------
PENALTY_XG = 0.76
GOAL_ARTIFACT_X = 50
POSTS = (45.2, 54.8)
FEATURE_COLS = ['dist', 'angle', 'head', 'open_play', 'assisted_throughball']

_COEF = np.array([-0.6424441787583652, 0.4791501045551712, -0.42190649386398515, 0.08967940645030949, 0.19115100916665412])
_INTERCEPT = -2.4724226381668792
_MEAN = np.array([16.811269915757205, 28.680392216327835, 0.16542859223654505, 0.6755152574466535, 0.02625446070934382])
_SCALE = np.array([7.495108262090593, 19.11797348513632, 0.3715669160557489, 0.46818200991007225, 0.15989110044716476])
# ---------------------------------------------------------------------------


def _shot_angle(x, y):
    a = np.array([100 - x, POSTS[0] - y])
    b = np.array([100 - x, POSTS[1] - y])
    cosang = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def _build_features(shots, events):
    shots = shots.copy()
    shots["dist"] = np.sqrt(
        (100 - shots["x"]) ** 2 + ((50 - shots["y"]) * (68 / 105)) ** 2
    )
    shots["angle"] = shots.apply(lambda r: _shot_angle(r["x"], r["y"]), axis=1)
    shots["head"] = (shots["shotBodyType"] == "Head").astype(int)
    shots["open_play"] = shots["shotOpenPlay"].fillna(False).astype(int)

    ev = events.sort_values(["matchId", "minute", "second", "id"])
    prev = ev.groupby("matchId")["keyPassThroughball"].shift(1).fillna(False).astype(int)
    ev = ev.assign(assisted_throughball=prev.values)
    flags = ev.loc[ev["isShot"] == True, ["id", "assisted_throughball"]]

    shots = shots.merge(flags, on="id", how="left")
    shots["assisted_throughball"] = shots["assisted_throughball"].fillna(0).astype(int)
    return shots


def _is_penalty(df):
    cols = [c for c in ["penaltyScored", "penaltyMissed"] if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    return df[cols].fillna(False).astype(bool).any(axis=1)


def _predict(shots):
    X = shots[FEATURE_COLS].to_numpy(dtype=float)
    z = ((X - _MEAN) / _SCALE) @ _COEF + _INTERCEPT
    return 1.0 / (1.0 + np.exp(-z))


def add_xg(events):
    """Applies the frozen MLS xG model. No training, no model file.
    Returns events with xG, assist_playerId, and xG_assisted columns."""
    events = events.copy()
    events["xG"] = np.nan

    is_shot = events["type"].isin(["Goal", "SavedShot", "MissedShots", "ShotOnPost"])
    pens = _is_penalty(events)
    events.loc[is_shot & pens, "xG"] = PENALTY_XG

    artifact = (events["type"] == "Goal") & (events["x"] < GOAL_ARTIFACT_X)
    mask = is_shot & ~pens & ~artifact
    shots = _build_features(events.loc[mask], events).dropna(subset=["dist", "angle"])

    if len(shots):
        shots = shots.assign(xG=_predict(shots))
        xg_by_id = shots.set_index("id")["xG"]
        events["xG"] = events["xG"].fillna(events["id"].map(xg_by_id))

    return _add_xg_assisted(events)


def _add_xg_assisted(events):
    ev = events.sort_values(["matchId", "minute", "second", "id"]).copy()
    ev["prev_type"] = ev.groupby("matchId")["type"].shift(1)
    ev["prev_playerId"] = ev.groupby("matchId")["playerId"].shift(1)

    is_shot = ev["type"].isin(["Goal", "SavedShot", "MissedShots", "ShotOnPost"])
    assisted = is_shot & (ev["prev_type"] == "Pass") & (ev["prev_playerId"] != ev["playerId"])

    ev["assist_playerId"] = np.where(assisted, ev["prev_playerId"], np.nan)
    ev["xG_assisted"] = np.where(assisted, ev["xG"], np.nan)
    return ev.drop(columns=["prev_type", "prev_playerId"]).sort_index()


def player_totals(events):
    pens = _is_penalty(events)
    shots = events[events["xG"].notna()]
    xg = shots.groupby("playerId")["xG"].sum().rename("xG")
    npxg = shots[~pens.reindex(shots.index, fill_value=False)] \
        .groupby("playerId")["xG"].sum().rename("npxG")
    xga = events.groupby("assist_playerId")["xG_assisted"].sum().rename("xG_assisted")
    xga.index.name = "playerId"
    return pd.concat([xg, npxg, xga], axis=1).fillna(0.0)

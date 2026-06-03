import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

PENALTY_XG = 0.76
GOAL_ARTIFACT_X = 50
POSTS = (45.2, 54.8)
FEATURE_COLS = ["dist", "angle", "head", "open_play", "assisted_throughball"]


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


def add_xg(events, return_model=False):
    """
    Accepts a full Opta/WhoScored events DataFrame and returns it with an 'xG'
    column. Non-shot rows get NaN, penalties get a flat value, and all other
    shots get a model prediction. The model is trained on the non-penalty shots
    within the input.
    """
    events = events.copy()
    events["xG"] = np.nan

    is_shot = events["type"].isin(["Goal", "SavedShot", "MissedShots", "ShotOnPost"])
    pens = _is_penalty(events)

    events.loc[is_shot & pens, "xG"] = PENALTY_XG

    artifact = (events["type"] == "Goal") & (events["x"] < GOAL_ARTIFACT_X)
    model_mask = is_shot & ~pens & ~artifact
    shots = _build_features(events.loc[model_mask], events)
    shots["scored"] = (shots["type"] == "Goal").astype(int)
    shots = shots.dropna(subset=["dist", "angle"])

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(shots[FEATURE_COLS], shots["scored"])

    events.loc[shots.index, "xG"] = model.predict_proba(shots[FEATURE_COLS])[:, 1]

    events = _add_xg_assisted(events)

    if return_model:
        return events, model
    return events


def _add_xg_assisted(events):
    """
    Credits each shot's xG to the player whose pass immediately preceded it.
    Adds 'assist_playerId' (the passer) and 'xG_assisted' (= the shot's xG) on
    shot rows that were preceded by a teammate's pass. Self-passes (dribble then
    shot) and non-pass precursors get NaN.
    """
    ev = events.sort_values(["matchId", "minute", "second", "id"]).copy()
    ev["prev_type"] = ev.groupby("matchId")["type"].shift(1)
    ev["prev_playerId"] = ev.groupby("matchId")["playerId"].shift(1)

    is_shot = ev["type"].isin(["Goal", "SavedShot", "MissedShots", "ShotOnPost"])
    assisted = is_shot & (ev["prev_type"] == "Pass") & (ev["prev_playerId"] != ev["playerId"])

    ev["assist_playerId"] = np.where(assisted, ev["prev_playerId"], np.nan)
    ev["xG_assisted"] = np.where(assisted, ev["xG"], np.nan)

    return ev.drop(columns=["prev_type", "prev_playerId"]).sort_index()


def player_totals(events):
    """
    Aggregates per-player xG (as shooter) and xG assisted (as passer) from an
    events frame already processed by add_xg. npxG strips penalty shots.
    """
    pens = _is_penalty(events)
    shots = events[events["xG"].notna()]

    xg = shots.groupby("playerId")["xG"].sum().rename("xG")
    npxg = shots[~pens.reindex(shots.index, fill_value=False)] \
        .groupby("playerId")["xG"].sum().rename("npxG")
    xga = events.groupby("assist_playerId")["xG_assisted"].sum().rename("xG_assisted")
    xga.index.name = "playerId"

    return pd.concat([xg, npxg, xga], axis=1).fillna(0.0)

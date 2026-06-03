import ast
import numpy as np
import pandas as pd

"""
I did not write this, Claude did, and it kind of sucks. There might be a better way to do it.
"""

PERIOD_ORDER = {"PreMatch": 0, "FirstHalf": 1, "SecondHalf": 2,
                "ExtraFirstHalf": 3, "ExtraSecondHalf": 4, "PenaltyShootout": 5, "PostGame": 6}

ON_BALL_NEXT = {"Pass", "Shot", "MissedShots", "SavedShot", "Goal", "TakeOn",
                "BallTouch", "Foul", "Dispossessed", "OffsidePass", "Clearance"}
ORIGIN_TYPES = {"Pass", "Tackle", "Interception", "BlockedPass", "Clearance",
                "BallRecovery", "Aerial", "BallTouch", "TakeOn", "KeeperPickup",
                "Save", "ShieldBallOpp"}
DEAD_TYPES = {"Foul", "Card", "OffsideGiven", "CornerAwarded", "Start", "End",
              "SubstitutionOn", "SubstitutionOff", "FormationSet", "FormationChange"}
SET_PIECE_QUALS = {"ThrowIn", "FreekickTaken", "CornerTaken", "GoalKick", "KeeperThrow", "Penalty"}
PITCH_LEN, PITCH_WID = 105.0, 68.0


def _has_setpiece(q):
    if isinstance(q, str):
        try:
            q = ast.literal_eval(q)
        except Exception:
            return False
    if isinstance(q, list):
        return any(d.get("type") in SET_PIECE_QUALS for d in q)
    return False


def extract_carries(df, min_dist=3.0, max_time=15.0):
    df = df.copy()
    df["_ord"] = df["period"].map(PERIOD_ORDER).fillna(99)
    df["_t"] = df["minute"] * 60 + df["second"].fillna(0)
    df = df.sort_values(["_ord", "_t", "eventId"], kind="stable").reset_index(drop=True)
    n = len(df)
    rows = []

    def gap_ok(ax, ay, bx, by, ta, tb):
        if any(pd.isna(v) for v in (ax, ay, bx, by)):
            return None
        dt = tb - ta
        if dt < 0 or dt > max_time:
            return None
        dist = np.hypot((bx - ax) / 100 * PITCH_LEN, (by - ay) / 100 * PITCH_WID)
        return dist if dist >= min_dist else None

    for i in range(n):
        cur = df.iloc[i]
        if cur["type"] not in ORIGIN_TYPES or cur["outcomeType"] != "Successful":
            continue
        pid, tid = cur["playerId"], cur["teamId"]
        if pd.isna(pid):
            continue
        sx, sy = cur["x"], cur["y"]
        ta = cur["minute"] * 60 + (cur["second"] or 0)
        for j in range(i + 1, n):
            nxt = df.iloc[j]
            if nxt["teamId"] != tid or nxt["type"] in DEAD_TYPES:
                break
            if nxt["playerId"] != pid or nxt["type"] not in ON_BALL_NEXT:
                continue
            tb = nxt["minute"] * 60 + (nxt["second"] or 0)
            if gap_ok(sx, sy, nxt["x"], nxt["y"], ta, tb) is not None:
                rows.append((pid, sx, sy, nxt["x"], nxt["y"], cur["eventId"]))
            break

    for i in range(n):
        cur = df.iloc[i]
        if cur["type"] != "Pass" or cur["outcomeType"] != "Successful":
            continue
        if _has_setpiece(cur["qualifiers"]):
            continue
        tid, rx, ry = cur["teamId"], cur["endX"], cur["endY"]
        ta = cur["minute"] * 60 + (cur["second"] or 0)
        for j in range(i + 1, n):
            nxt = df.iloc[j]
            if nxt["teamId"] != tid or nxt["type"] in DEAD_TYPES:
                break
            if nxt["type"] not in ON_BALL_NEXT:
                continue
            if pd.isna(nxt["playerId"]):
                break
            tb = nxt["minute"] * 60 + (nxt["second"] or 0)
            if gap_ok(rx, ry, nxt["x"], nxt["y"], ta, tb) is not None:
                rows.append((nxt["playerId"], rx, ry, nxt["x"], nxt["y"], nxt["eventId"]))
            break

    out = pd.DataFrame(rows, columns=["playerId", "startX", "startY", "endX", "endY", "_endEvt"])
    out = out.drop_duplicates(subset=["_endEvt", "playerId"]).drop(columns="_endEvt").reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = pd.read_excel("/mnt/user-data/uploads/Sample_Events.xlsx")
    carries = extract_carries(df)
    print(carries.head(15).to_string())
    print("total carries:", len(carries))

from collections import defaultdict
import json, math
import pandas as pd

pd.options.display.float_format = '{:.2f}'.format

filename = "3rdIteration.json"
raw_data = pd.read_json(filename)
raw_data = raw_data[raw_data.model_b != 'string']
raw_data = raw_data[raw_data.model_a != "string"]
raw_data = raw_data[raw_data.answer_a != "string"]
raw_data = raw_data[raw_data.answer_b != "string"]

battles = raw_data.reset_index(drop=True)
print(battles)


def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        ra = 832
        rb = 1168
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == model_a:
            sa = 1
        elif winner == model_b:
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        elif winner == "both-bad":
            sa = -1.5
        else:
            raise Exception(f"unexpected vote {winner} {model_a} {model_b}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating


def preety_print_elo_ratings(ratings):
    df = pd.DataFrame([
        [n, elo_ratings[n]] for n in elo_ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    return df


elo_ratings = compute_elo(battles)
print(elo_ratings)
df = preety_print_elo_ratings(elo_ratings)
print(df)

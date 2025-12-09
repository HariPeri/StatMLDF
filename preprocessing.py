import pandas as pd
import os
import glob

def load_all_data(data_path="data"):
    all_files = sorted(glob.glob(os.path.join(data_path, "atp_matches_*.csv")))
    df_list = []

    for file in all_files:
        year = int(os.path.basename(file).split("_")[2].split(".")[0])

        temp_df = pd.read_csv(file)
        temp_df["year"] = year  

        df_list.append(temp_df)

    full_df = pd.concat(df_list, ignore_index=True)

    return full_df

import pandas as pd

def convert_to_player_level(df):

    # --- Player-side mappings ---
    player_map = {
        "id": ["winner_id", "loser_id"],
        "age": ["winner_age", "loser_age"],
        "ht": ["winner_ht", "loser_ht"],
        "hand": ["winner_hand", "loser_hand"],
        "rank": ["winner_rank", "loser_rank"],
        "rank_points": ["winner_rank_points", "loser_rank_points"],
        "ace": ["w_ace", "l_ace"],
        "df": ["w_df", "l_df"],
        "sv_gms": ["w_SvGms", "l_SvGms"],
        "svpt": ["w_svpt", "l_svpt"],
        "first_in": ["w_1stIn", "l_1stIn"],
        "first_won": ["w_1stWon", "l_1stWon"],
        "second_won": ["w_2ndWon", "l_2ndWon"],
        "bp_saved": ["w_bpSaved", "l_bpSaved"],     
        "bp_faced": ["w_bpFaced", "l_bpFaced"]      
    }

    # --- Opponent-side mappings ---
    opponent_map = {
        "id": ["loser_id", "winner_id"],
        "age": ["loser_age", "winner_age"],
        "ht": ["loser_ht", "winner_ht"],
        "hand": ["loser_hand", "winner_hand"],
        "rank": ["loser_rank", "winner_rank"],
        "rank_points": ["loser_rank_points", "winner_rank_points"],
        "ace": ["l_ace", "w_ace"],
        "df": ["l_df", "w_df"],
        "sv_gms": ["l_SvGms", "w_SvGms"],
        "bp_saved": ["l_bpSaved", "w_bpSaved"],     
        "bp_faced": ["l_bpFaced", "w_bpFaced"]      
    }

    # --- Context preserved columns ---
    context_cols = [
        "tourney_id", "tourney_name", "surface", "draw_size",
        "tourney_level", "tourney_date", "round", "minutes", "year"
    ]

    # --- Build winner-as-player rows ---
    winner_df = pd.DataFrame({
        "player_id": df[player_map["id"][0]],
        "player_age": df[player_map["age"][0]],
        "player_ht": df[player_map["ht"][0]],
        "player_hand": df[player_map["hand"][0]],
        "player_rank": df[player_map["rank"][0]],
        "player_rank_points": df[player_map["rank_points"][0]],
        "player_ace": df[player_map["ace"][0]],
        "player_df": df[player_map["df"][0]],
        "player_SvGms": df[player_map["sv_gms"][0]],
        "player_svpt": df[player_map["svpt"][0]],
        "player_1stIn": df[player_map["first_in"][0]],
        "player_1stWon": df[player_map["first_won"][0]],
        "player_2ndWon": df[player_map["second_won"][0]],
        "player_bpSaved": df[player_map["bp_saved"][0]],      
        "player_bpFaced": df[player_map["bp_faced"][0]],      

        "opponent_id": df[opponent_map["id"][0]],
        "opponent_age": df[opponent_map["age"][0]],
        "opponent_ht": df[opponent_map["ht"][0]],
        "opponent_hand": df[opponent_map["hand"][0]],
        "opponent_rank": df[opponent_map["rank"][0]],
        "opponent_rank_points": df[opponent_map["rank_points"][0]],
        "opponent_ace": df[opponent_map["ace"][0]],
        "opponent_df": df[opponent_map["df"][0]],
        "opponent_SvGms": df[opponent_map["sv_gms"][0]],
        "opponent_bpSaved": df[opponent_map["bp_saved"][0]],  
        "opponent_bpFaced": df[opponent_map["bp_faced"][0]],  

        "outcome": "win",
    })

    for col in context_cols:
        winner_df[col] = df[col]

    # --- Build loser-as-player rows ---
    loser_df = pd.DataFrame({
        "player_id": df[player_map["id"][1]],
        "player_age": df[player_map["age"][1]],
        "player_ht": df[player_map["ht"][1]],
        "player_hand": df[player_map["hand"][1]],
        "player_rank": df[player_map["rank"][1]],
        "player_rank_points": df[player_map["rank_points"][1]],
        "player_ace": df[player_map["ace"][1]],
        "player_df": df[player_map["df"][1]],
        "player_SvGms": df[player_map["sv_gms"][1]],
        "player_svpt": df[player_map["svpt"][1]],
        "player_1stIn": df[player_map["first_in"][1]],
        "player_1stWon": df[player_map["first_won"][1]],
        "player_2ndWon": df[player_map["second_won"][1]],
        "player_bpSaved": df[player_map["bp_saved"][1]],      
        "player_bpFaced": df[player_map["bp_faced"][1]],      

        "opponent_id": df[opponent_map["id"][1]],
        "opponent_age": df[opponent_map["age"][1]],
        "opponent_ht": df[opponent_map["ht"][1]],
        "opponent_hand": df[opponent_map["hand"][1]],
        "opponent_rank": df[opponent_map["rank"][1]],
        "opponent_rank_points": df[opponent_map["rank_points"][1]],
        "opponent_ace": df[opponent_map["ace"][1]],
        "opponent_df": df[opponent_map["df"][1]],
        "opponent_SvGms": df[opponent_map["sv_gms"][1]],
        "opponent_bpSaved": df[opponent_map["bp_saved"][1]],  
        "opponent_bpFaced": df[opponent_map["bp_faced"][1]],  

        "outcome": "lose",
    })

    for col in context_cols:
        loser_df[col] = df[col]

    combined = pd.concat([winner_df, loser_df], ignore_index=True)

    combined["df_rate"] = combined["player_df"] / combined["player_SvGms"].replace(0, pd.NA)

    return combined

def add_engineered_features(df):
    df = df.copy()

    # Handle divisions safely
    eps = 1e-9

    # First serve percentage
    df["first_serve_pct"] = df["player_1stIn"] / (df["player_svpt"] + eps)

    # First-serve win percentage
    df["first_serve_win_pct"] = df["player_1stWon"] / (df["player_1stIn"] + eps)

    # Second-serve win percentage
    df["second_serve_win_pct"] = df["player_2ndWon"] / (
        (df["player_svpt"] - df["player_1stIn"]) + eps
    )

    # Ace rate
    df["ace_rate"] = df["player_ace"] / (df["player_SvGms"] + eps)

    # Break point pressure
    df["bp_pressure"] = df["player_bpFaced"] / (df["player_SvGms"] + eps)

    # Break point clutch (saved / faced)
    df["bp_clutch"] = df["player_bpSaved"] / (df["player_bpFaced"] + eps)

    # Opponent ace rate
    df["opp_ace_rate"] = df["opponent_ace"] / (df["opponent_SvGms"] + eps)

    # Opponent pressure
    df["opp_bp_pressure"] = df["opponent_bpFaced"] / (df["opponent_SvGms"] + eps)

    # Rank difference
    df["rank_diff"] = df["player_rank"] - df["opponent_rank"]

    # Height difference
    df["ht_diff"] = df["player_ht"] - df["opponent_ht"]

    return df


if __name__ == "__main__":
    match_df = load_all_data("data")
    match_df.to_csv("data/processed/matches.csv", index=False)
    print(match_df.shape)
    print(match_df.head())

    player_df = convert_to_player_level(match_df)
    player_df.to_csv("data/processed/player_level_raw.csv", index=False)
    print(player_df.shape)
    print(player_df.head())

    engineered_df = add_engineered_features(player_df)
    engineered_df.to_csv("data/processed/player_level_engineered.csv", index=False)
    print(engineered_df.shape)
    print(engineered_df.head())







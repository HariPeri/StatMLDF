import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/player_level_engineered.csv")

# DF Rate Distribution
sns.histplot(df["df_rate"], bins=40, kde=True)
plt.title("Distribution of Double Fault Rate")
plt.show()

yearly = df.groupby("year")["df_rate"].mean()

plt.figure(figsize=(8,5))
sns.lineplot(x=yearly.index, y=yearly.values)
plt.title("Average DF Rate by Year")
plt.ylabel("DF Rate")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x="round", y="df_rate", data=df)
plt.xticks(rotation=45)
plt.title("Double Fault Rate by Tournament Round")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="surface", y="df_rate")
plt.title("Double Fault Rate by Surface")
plt.xlabel("Surface")
plt.ylabel("Double Fault Rate")
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(x="ace_rate", y="df_rate", data=df, alpha=0.2)
plt.title("Ace Rate vs Double Fault Rate")
plt.show()

sns.jointplot(
    data=df, x="first_serve_pct", y="df_rate",
    kind="hex", gridsize=30
)
plt.show()

df["pressure_bucket"] = pd.qcut(df["bp_pressure"], 10)

df.groupby("pressure_bucket")["df_rate"].mean().plot(kind="bar", figsize=(10,5))
plt.title("DF Rate vs Serve Pressure (Binned)")
plt.ylabel("Average DF Rate")
plt.show()

sns.scatterplot(x="opp_bp_pressure", y="df_rate", data=df, alpha=0.3)
plt.title("Opponent Pressure vs Double Fault Rate")
plt.show()

df["rank_bucket"] = pd.qcut(df["rank_diff"], 10)

rank_trend = df.groupby("rank_bucket")["df_rate"].mean()

rank_trend.plot(kind="bar", figsize=(10,5))
plt.title("DF Rate by Rank Difference Decile")
plt.ylabel("Average DF Rate")
plt.show()


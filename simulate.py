# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from util.market import Market
from util.learners import CrossSectionalDML


def lift_fn(x: pd.DataFrame, treatment_period: int) -> pd.DataFrame:
    x["lift"] = x["treated"] * (x["time_period"] == treatment_period) * (
        0.1 * x["attribute_0"] + 0.2 * x["attribute_1"] + - 0.1 * x["product_class"]
    )
    x["sales"] = x["sales"] + x["lift"]
    return x

def pull_forward_fn(x: pd.DataFrame, treatment_period: int) -> pd.DataFrame:
    # compute decay factor
    decay = (x["time_period"] > treatment_period) * 0.1 ** (x["time_period"] - treatment_period)

    # label treated period
    event = (x["time_period"] == treatment_period) & (x["treated"] == 1)

    # compute pull-forward effect (demand lost from later periods)
    x["pull_forward"] = -0.1 * decay * x["treated"] * x["sales"] * (x["product_class"] + 2)

    # move the shifted demand to the event period
    x["__to_shift__"] = x.groupby("product_id")["pull_forward"].transform(lambda x: x.sum())
    x.loc[event, "pull_forward"] = x.loc[event, "__to_shift__"] * -1
    x["sales"] = x["sales"] + x["pull_forward"]
    x = x.drop(columns=["__to_shift__"])

    return x

def growth_fn(x: pd.DataFrame, treatment_period: int) -> pd.DataFrame:
    # compute decay factor
    decay = (x["time_period"] > treatment_period) * 0.5 ** (x["time_period"] - treatment_period)

    # compute growth effect (demand gained in later periods)
    x["growth"] = 20 * x["treated"] * decay * (
        0.1 * x["attribute_3"] + 0.2 * x["attribute_4"]
    ) 
    x["sales"] = x["sales"] + x["growth"]
    return x

market = Market(
    N=100000,
    T=20,
    fn_lift=lift_fn,
    fn_pull_forward=pull_forward_fn,
    fn_growth=growth_fn,
    seed=42,
    rho=0.9
)

# %%
sns.scatterplot(
    data=market.data
        .query("time_period >= 10")
        .query("product_class.isin([1, 4])")
        .replace({"product_class": {1: "Low pull-forward", 4: "High pull-forward"}})
        .groupby(["time_period", "product_class"])["ite"].mean().reset_index(),
    x="time_period",
    y="ite",
    hue="product_class",
    palette={"Low pull-forward": "red", "High pull-forward": "blue"},
)
plt.legend(title="Product class")
plt.xlabel("Time period")
plt.ylabel("CATE")

plt.show()

# %%
model = CrossSectionalDML()
model.fit(market, 10)

# %%
df = market.data.query("time_period == 10")
print(
    model.fit_model[10].ate_inference(
        X=df.query("product_class == 1")[market.features],
        T0=0,
        T1=1,
    ).mean_point,
    model.fit_model[10].ate_inference(
        X=df.query("product_class == 4")[market.features],
        T0=0,
        T1=1,
    ).mean_point
)

# %%
for i in tqdm(range(11, market.T)):
    model.fit(market, i)

# %%
results = []
for i in range(10, market.T):
    df = market.data.query(f"time_period == {i}")
    results.append([
        i,
        df[["sales_control", "sales_treated"]].mean().diff().iloc[-1],
        model.fit_model[i].ate_inference(
            X=df[market.features],
            T0=0,
            T1=1,
        ).mean_point,
        df.query("product_class == 1")[["sales_control", "sales_treated"]].mean().diff().iloc[-1],
        model.fit_model[i].ate_inference(
            X=df.query("product_class == 1")[market.features],
            T0=0,
            T1=1,
        ).mean_point,
        df.query("product_class == 4")[["sales_control", "sales_treated"]].mean().diff().iloc[-1],
        model.fit_model[i].ate_inference(
            X=df.query("product_class == 4")[market.features],
            T0=0,
            T1=1,
        ).mean_point
    ])
results = pd.DataFrame(results, columns=["time_period", "ground_truth", "dml", "ground_truth_1", "dml_1", "ground_truth_4", "dml_4"])

# %%
df = results.melt(value_vars = ['dml', 'dml_1', 'dml_4', 'ground_truth', 'ground_truth_1', 'ground_truth_4'], id_vars="time_period")

# %%
sns.scatterplot(
    data=df
        .query("variable.isin(['dml_1', 'dml_4'])")
        .replace({"variable": {"dml_1": "Low pull-forward", "dml_4": "High pull-forward"}}),
    x="time_period",
    y="value",
    hue="variable",
    palette={"Low pull-forward": "red", "High pull-forward": "blue"},
)
plt.legend(title="Product class")
plt.xlabel("Time period")
plt.ylabel("CATE Estimate")

plt.show()
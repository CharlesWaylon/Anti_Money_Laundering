"""GNN part 2"""

import pandas as pd

data = pd.read_csv("aml_syn_data")

# Parse time
data["Time_step"] = pd.to_datetime(data["Time_step"])

# Map accounts to node indices
all_accounts = pd.unique(
    pd.concat([data["Sender_Account"], data["Bene_Account"]], ignore_index=True)
)
acct2idx = {acct: i for i, acct in enumerate(all_accounts)}

data["src"] = data["Sender_Account"].map(acct2idx)
data["dst"] = data["Bene_Account"].map(acct2idx)
data = data.sort_values("Time_step").reset_index(drop=True)
n = len(data)

train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_df = data.iloc[:train_end]
val_df   = data.iloc[train_end:val_end]
test_df  = data.iloc[val_end:]

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Log-scale amount
data["log_amount"] = np.log1p(data["USD_amount"])

# Time features
data["hour"] = data["Time_step"].dt.hour
data["dayofweek"] = data["Time_step"].dt.dayofweek
data["month"] = data["Time_step"].dt.month

# Normalize continuous features (fit on train only)
cont_edge_cols = ["log_amount", "hour", "dayofweek", "month"]
scaler = StandardScaler()
scaler.fit(data.loc[train_df.index, cont_edge_cols])
data[cont_edge_cols] = scaler.transform(data[cont_edge_cols])

# Categorical edge features (transaction type)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(data.loc[train_df.index, ["Transaction_Type"]])
tx_type_ohe = ohe.transform(data[["Transaction_Type"]])

# Assemble edge_attr matrix
import numpy as np

edge_cont = data[cont_edge_cols].to_numpy()
edge_attr = np.hstack([edge_cont, tx_type_ohe])

# Store edge_attr indices per split
edge_attr_train = edge_attr[train_df.index]
edge_attr_val   = edge_attr[val_df.index]
edge_attr_test  = edge_attr[test_df.index]
label_map = {"GOOD": 0, "BAD": 1}
data["y"] = data["Label"].map(label_map)

y_train = data.loc[train_df.index, "y"].to_numpy()
y_val   = data.loc[val_df.index, "y"].to_numpy()
y_test  = data.loc[test_df.index, "y"].to_numpy()

# Build basic per-account stats using all data
sent_stats = train_df.groupby("src").agg(
    sent_count=("src", "size"),
    sent_total_amount=("USD_amount", "sum"),
    sent_avg_amount=("USD_amount", "mean"),
)

recv_stats = train_df.groupby("dst").agg(
    recv_count=("dst", "size"),
    recv_total_amount=("USD_amount", "sum"),
    recv_avg_amount=("USD_amount", "mean"),
)

num_nodes = len(all_accounts)
node_stats = (
    sent_stats.join(recv_stats, how="outer")
    .reindex(range(num_nodes))
    .fillna(0.0)
)

assert node_stats.shape[0] <= num_nodes

from sklearn.preprocessing import StandardScaler
import numpy as np

def most_frequent_or_nan(x):
    vc = x.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.index[0]

# Dominant sender country per account (by src)
dom_country = (
    data.groupby("src")["Sender_Country"]
        .agg(most_frequent_or_nan)
        .reindex(range(num_nodes))  # ensure index 0..num_nodes-1
)

# Optionally fill unknowns
dom_country = dom_country.fillna("UNKNOWN")

from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data

ohe_country = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
country_ohe = ohe_country.fit_transform(dom_country.to_frame())

# --- Scale numeric node stats and concatenate with country one-hot ---
node_scaler = StandardScaler()
node_num = node_scaler.fit_transform(node_stats.values.astype("float32"))

node_features = np.hstack([node_num, country_ohe])
x = torch.from_numpy(node_features).float()

edge_index = torch.as_tensor(
    np.vstack([data["src"].to_numpy(), data["dst"].to_numpy()]),
    dtype=torch.long
)

edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
y = torch.tensor(data["y"].to_numpy(), dtype=torch.float)

data2 = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr_t
)


# Edge masks for splits
train_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
val_mask   = torch.zeros(edge_index.size(1), dtype=torch.bool)
test_mask  = torch.zeros(edge_index.size(1), dtype=torch.bool)

train_mask[train_df.index] = True
val_mask[val_df.index] = True
test_mask[test_df.index] = True

data2.train_mask = train_mask
data2.val_mask   = val_mask
data2.test_mask  = test_mask
data2.y          = y  # edge-level labels
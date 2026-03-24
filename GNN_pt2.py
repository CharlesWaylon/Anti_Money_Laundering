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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class EdgeClassifierGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def embed_nodes(self, x, mp_edge_index):
        h = F.relu(self.conv1(x, mp_edge_index))
        h = F.relu(self.conv2(h, mp_edge_index))
        return h

    def score_edges(self, h, pred_edge_index, pred_edge_attr):
        src, dst = pred_edge_index
        edge_repr = torch.cat([h[src], h[dst], pred_edge_attr], dim=-1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(self, x, mp_edge_index, pred_edge_index, pred_edge_attr=None):
        h = self.embed_nodes(x, mp_edge_index)
        return self.score_edges(h, pred_edge_index, pred_edge_attr)
    



import copy

best_f1 = -1
best_epoch = -1
best_state = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EdgeClassifierGNN(
    node_in_dim=data2.x.size(1),
    edge_in_dim=data2.edge_attr.size(1),
    hidden_dim=64
).to(device)

data2 = data2.to(device)
y_train_t = data2.y[data2.train_mask]
num_pos = (y_train_t == 1).sum()
num_neg = (y_train_t == 0).sum()
pos_weight = num_neg / num_pos

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

train_losses = []
val_f1s = []

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()

    logits = model(data2.x, data2.edge_index, data2.edge_attr)
    loss = criterion(logits[data2.train_mask], data2.y[data2.train_mask])

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = logits[data2.val_mask]
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).float()
        val_labels = data2.y[data2.val_mask]

        # simple metrics
        from sklearn.metrics import f1_score
        f1 = f1_score(val_labels.cpu(), val_preds.cpu())

    val_f1s.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())


    print(f"Epoch {epoch} | train_loss={loss.item():.4f} | val_f1={f1:.4f}")

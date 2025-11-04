import pandas as pd
import numpy as np

## Step 0 Fetching World Bank Data

### Step 0.0: View each indicator at https://data.worldbank.org/indicator/{indicator_code}
### Step 0.1: Download CSV file for each indicator from https://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=csv
### Step 0.2: Load and merge on 'Country Name'
df_list = []
for code in ['IT.NET.USER.ZS','IT.NET.BBND.P2','SE.TER.ENRR','SP.POP.SCIE.RD.P6',
             'EG.USE.ELEC.KH.PC','EG.FEC.RNEW.ZS','NY.GDP.PCAP.CD',
             'TX.VAL.TECH.MF.ZS','GB.XPD.RSDV.GD.ZS']:
    temp = pd.read_csv(f'WorldBankData/API_{code}.csv')
    temp = temp[['Country Name','2021']].rename(columns={'2021': code})
    df_list.append(temp)

data = df_list[0]
for d in df_list[1:]:
    data = pd.merge(data, d, on='Country Name', how='inner')

## Step 1: Data Cleaning and Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_and_scale(df: pd.DataFrame, exclude=['Country Name'], verbose=True):
    """
    Clean and scale numeric columns of a DataFrame for clustering.
    - Coerces non-numeric values to NaN.
    - Drops any row (country) with NaN values.
    - Removes constant columns.
    - Scales numeric columns to [0, 1].
    Returns the cleaned and scaled DataFrame.
    """
    data = df.copy()

    # 1. Convert all non-excluded columns to numeric
    for col in data.columns:
        if col not in exclude:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 2. Drop rows with any NaN (strict cleaning)
    before = len(data)
    data = data.dropna().reset_index(drop=True)
    after = len(data)
    if verbose:
        print(f"Dropped {before - after} rows with missing or invalid values.")

    # 3. Remove constant columns (zero variance)
    num_cols = [c for c in data.columns if c not in exclude]
    constant_cols = [c for c in num_cols if data[c].nunique() <= 1]
    if constant_cols and verbose:
        print(f"Removing constant columns: {constant_cols}")
    num_cols = [c for c in num_cols if c not in constant_cols]

    # 4. GDP and Electricity Consumption are often right-skewed, so we apply log transformation
    log_cols = [c for c in ['NY.GDP.PCAP.CD','EG.USE.ELEC.KH.PC'] if c in num_cols]
    if log_cols:
        data[log_cols] = np.log1p(data[log_cols])

    # 5. Apply MinMax scaling to [0,1]
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(data[num_cols]),
        columns=num_cols
    )

    # 6. Reattach excluded columns
    for col in exclude:
        scaled[col] = data[col].values

    if verbose:
        print(f"Final dataset: {scaled.shape[0]} rows × {scaled.shape[1]} columns.")
        print(f"Scaled features: {num_cols}")

    return scaled

### Step 1.2: Scale numeric columns appropriately
data = clean_and_scale(data, exclude=['Country Name'])
print(data.isna().sum())
# for i, row in data.iterrows():
#     print(row)
#     print('-' * 10)

### Step 1.3: Separate country names from data for clustering
country_names = data['Country Name'].tolist()
country_data = data.drop(columns=['Country Name'])

## Step 2: Clustering
### Step 2.0 Helper: PCA for 2D plots
from sklearn.decomposition import PCA

def pca2d(X):
    # print(X)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)

X = country_data.values
XY = pca2d(X)

### Step 2.1: DBSCAN with simple eps search
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# showed that 0.3-0.5 is an optimal eps value
def plot_dbscan_possible_eps_values(X):
    min_samples = 5
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    (distances, _) = neighbors_fit.kneighbors(X)

    # Sort distances of each point to its kth neighbor
    distances = np.sort(distances[:, min_samples-1])

    plt.figure(figsize=(6,4))
    plt.plot(distances)
    plt.ylabel(f"Distance to {min_samples}th nearest neighbor")
    plt.xlabel("Points sorted by distance")
    plt.title("DBSCAN eps selection (k-distance graph)")
    plt.grid(True)
    plt.show()

# plot_dbscan_possible_eps_values(X)

eps = 0.3
min_samples = 3

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points (outliers).")

mask_core = labels != -1
countries_core = np.array(country_names)[mask_core]
cluster_ids = labels[mask_core]
X_core = X[mask_core]

print("Core countries and their cluster IDs:", list(zip(countries_core, cluster_ids)))

### Step 2.2: K-Means Clustering For Sub-Clusters and K-Medoids for Representatives
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def medoid_index(X_sub):
    """Return the index (within X_sub) of the point minimizing total distance to all others."""
    D = cdist(X_sub, X_sub, metric='euclidean')
    return int(np.argmin(D.sum(axis=1)))

kmeans_labels_global = np.full(len(X_core), fill_value=-1, dtype=int)
subcluster_global_id = 0
subclusters_info = []  # details for later plotting
medoids = []

for c in sorted(set(cluster_ids)):
    # absolute indices in the core set for DBSCAN cluster c
    abs_idx = np.where(cluster_ids == c)[0]
    X_sub = X_core[abs_idx]
    sub_countries = countries_core[abs_idx]

    # Pick best k using silhouette
    best = {'k': None, 'score': -1, 'labels': None}
    for k in range(2, min(6, len(X_sub))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbl = km.fit_predict(X_sub)
        if len(set(lbl)) < 2:
            continue
        try:
            s = silhouette_score(X_sub, lbl)
        except Exception:
            s = -1
        if s > best['score']:
            best = {'k': k, 'score': s, 'labels': lbl}

    # If too small or silhouette failed, assign single subgroup
    if best['labels'] is None:
        kmeans_labels_global[abs_idx] = subcluster_global_id
        mloc = medoid_index(X_sub) if len(X_sub) > 1 else 0
        medoids.append({'dbscan': c, 'kmeans_gid': subcluster_global_id,
                        'medoid_country': sub_countries[mloc]})
        subclusters_info.append({'parent': c, 'kmeans_gid': subcluster_global_id,
                                 'countries': sub_countries})
        subcluster_global_id += 1
        continue

    # Map local k-means labels to global ids using ABSOLUTE indices
    for local_k in range(best['k']):
        local_members = np.where(best['labels'] == local_k)[0]
        global_members = abs_idx[local_members]
        kmeans_labels_global[global_members] = subcluster_global_id
        X_grp = X_core[global_members]
        if len(X_grp) > 0:
            mloc = medoid_index(X_grp) if len(X_grp) > 1 else 0
            medoids.append({'dbscan': c, 'kmeans_gid': subcluster_global_id,
                            'medoid_country': countries_core[global_members[mloc]]})
        subclusters_info.append({'parent': c, 'kmeans_gid': subcluster_global_id,
                                 'countries': countries_core[global_members]})
        subcluster_global_id += 1

print("\nRepresentative countries (medoids):")
for m in medoids:
    print(f"DBSCAN {m['dbscan']}, K-Means {m['kmeans_gid']} -> {m['medoid_country']}")

print(np.unique(kmeans_labels_global, return_counts=True))

## Step 3: Visualization
import mplcursors

def add_shared_hover(ax, scatters, labels_list):
    artist2labels = {sc: np.asarray(lbls) for sc, lbls in zip(scatters, labels_list)}
    cursor = mplcursors.cursor(scatters, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        labels = artist2labels[sel.artist]
        sel.annotation.set_text(str(labels[sel.index]))

### Step 3.1: DBSCAN Visualization
pca = PCA(n_components=2, random_state=42)
XY = pca.fit_transform(X)

plt.figure(figsize=(7,6))
mask_noise = labels == -1
ax = plt.gca()
sc_clusters = plt.scatter(XY[~mask_noise,0], XY[~mask_noise,1],
            c=labels[~mask_noise], cmap='tab10', s=60, label='Clusters')
sc_noise = plt.scatter(XY[mask_noise,0], XY[mask_noise,1],
            c='k', s=30, alpha=0.5, label='Noise')
add_shared_hover(
    ax,
    [sc_clusters, sc_noise],
    [np.array(country_names)[~mask_noise], np.array(country_names)[mask_noise]]
)
plt.title("Level 1 – DBSCAN Clusters (PCA Projection)")
plt.legend()
plt.tight_layout()
plt.show()

### Step 3.2: K-Means Subcluster Visualization
XY_core = pca.fit_transform(X_core)

fig, ax = plt.subplots(figsize=(7,6))
sc_k = ax.scatter(XY_core[:,0], XY_core[:,1],
                  c=kmeans_labels_global, s=60, cmap='tab20')
add_shared_hover(ax, [sc_k], [countries_core])
ax.set_title("Level 2 – K-Means Subclusters (within DBSCAN Groups)")
plt.tight_layout()
plt.show()


### Step 3.3: K-Medoids Representative Visualization
if len(medoids) > 0:
    fig, ax = plt.subplots(figsize=(7,6))
    sc_sub = ax.scatter(XY_core[:,0], XY_core[:,1],
                        c=kmeans_labels_global, s=36, alpha=0.9, cmap='tab20', label='Subclusters')
    add_shared_hover(ax, [sc_sub], [countries_core])

    medoid_names = [m['medoid_country'] for m in medoids]
    medoid_indices = [np.where(countries_core == name)[0][0] for name in medoid_names]
    ax.scatter(XY_core[medoid_indices,0], XY_core[medoid_indices,1],
               s=200, facecolors='none', edgecolors='black', linewidths=2, label='Medoids')
    for idx, name in zip(medoid_indices, medoid_names):
        ax.text(XY_core[idx,0]+0.02, XY_core[idx,1]+0.02, name, fontsize=8)

    ax.set_title("Level 3 – K-Medoids Representatives (PCA Projection)")
    ax.legend()
    plt.tight_layout()
    plt.show()

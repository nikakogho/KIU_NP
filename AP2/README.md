# Multi-Level Clustering of Global Technological Development To Assess AGI Readiness

**(World Bank snapshot, 2021; three-layer unsupervised pipeline with custom L¹/L²/L∞ distances)**

## 1) Problem, idea, and scope

We want a realistic, reproducible way to **group countries by "AGI readiness"** - not as a single score, but as **natural clusters** that emerge from comparable indicators. We use **three hierarchical layers**:

1. **Level 1:** **DBSCAN** to discover coarse, density-based groups and mark outliers.
2. **Level 2:** **K-Means** *within each DBSCAN cluster* to find sub-profiles.
3. **Level 3:** **K-Medoids** (one representative per sub-cluster) to pick real **exemplar countries**.

This is an **unsupervised description**, not a forecast. Still, these groupings are useful for **policy triage**, **benchmarking**, and **trend monitoring** (when repeated across years).

**Data source:** only World Bank indicators (CSV files) for **2021**; stored in the **`WorldBankData`** folder.
**Figures:** screenshots of all plots live in the **`screenshots`** folder (DBSCAN clusters, K-Means subclusters, and K-Medoids representatives; PCA 2D projections).

---

## 2) Dataset and features (what each field means)

We merge indicators by **Country Name** and keep one value per indicator (year **2021**). The final feature set:

* **IT.NET.USER.ZS** — Individuals using the Internet **(% of population)**.
* **IT.NET.BBND.P2** — Fixed broadband subscriptions **(per 100 people)**.
* **SE.TER.ENRR** — Gross enrollment ratio, **tertiary education (%)**.
* **SP.POP.SCIE.RD.P6** — **Researchers in R&D (per million people)**.
* **EG.USE.ELEC.KH.PC** — Electric power consumption **(kWh per capita)**.
* **EG.FEC.RNEW.ZS** — Renewable energy consumption **(% of final energy)**.
* **NY.GDP.PCAP.CD** — **GDP per capita (current US$)**.
* **TX.VAL.TECH.MF.ZS** — **High-tech exports** (% of manufactured exports).
* **GB.XPD.RSDV.GD.ZS** — **R&D expenditure** (% of GDP).

These collectively span **digital access**, **infrastructure**, **education**, **human capital for R&D**, **energy usage and transition**, and **innovation outputs/inputs**—a defensible proxy mix for “global tech development.”

**Data location:** `WorldBankData/API_{indicator}.csv` as downloaded from the World Bank (indicator pages are linked in code comments).

---

## 3) Preprocessing and the scaling pitfall (and how we solved it)

Clustering is sensitive to feature scales. Previously, a large-range metric (e.g., GDP per capita) **dominated** others, producing misleading clusters. We fixed this in three steps:

1. **Strict cleaning:** cast to numeric; **drop any row with NaN** (no imputations); remove **constant** (zero-variance) columns.
2. **Skew handling:** apply **log1p** to **GDP per capita** and **electricity consumption** (both are right-skewed).
3. **Range alignment:** **Min-Max scaling to [0,1]** for all features.

This ensures each dimension contributes comparably, while the log transform gives **more resolution at lower values** (exactly what you wanted when you said “I care more about resolution at the low end”).

---

## 4) Distance model: manual L¹ / L² / L∞ without `np.linalg.norm`

To keep the math modular and transparent, we implemented **vector norms by hand** and expose a single switch:

* **L¹** (Manhattan): `sum(|xᵢ|)`
* **L²** (Euclidean): `sqrt(sum(xᵢ²))`
* **L∞** (Chebyshev): `max(|xᵢ|)`

A project-wide flag `SELECTED_NORM ∈ {'1','2','inf'}` controls the distance everywhere:

* **DBSCAN**: uses our custom distance via a callable metric.
* **K-Medoids**: uses our pairwise distance matrix computed with the same norm.
* **Silhouette**: computed either directly with the callable metric or via a precomputed distance matrix.

> Note: **K-Means** inherently optimizes Euclidean distortions (L² to centroids). We retain K-Means for its interpretability and speed at the sub-cluster layer; evaluation and medoids still use the selected norm.

---

## 5) Hierarchical design and hyperparameters

### Level 1 — DBSCAN (coarse structure + outliers)

* **Why DBSCAN first?** It’s shape-agnostic, detects **dense regions** vs **noise**, and doesn’t force every country into a cluster.
* **Choosing `eps`:** we use a **k-distance plot** (k = `min_samples`) and look for the elbow. In our runs, the useful band was about **0.30–0.50**; we proceeded with **`eps = 0.30`** and **`min_samples = 3`**.
* **Outcome (example run):** DBSCAN reported **3 clusters** and **29 noise points**. This is consistent with world heterogeneity: a few dense tech-profiles plus many unique/edge cases marked as noise.

**Visualization:** “Level 1 – DBSCAN Clusters (PCA Projection)” (see `screenshots/`). Points have **hover tooltips** with country names (via `mplcursors`); noise points plotted in grey/black.

### Level 2 — K-Means inside each DBSCAN cluster

* For each DBSCAN cluster, we try **k = 2..5** (capped by cluster size) and pick **k with the best silhouette score** (using our chosen norm for distances).
* This yields **sub-profiles** such as “digital-rich but moderate R&D,” “strong education + R&D,” or “energy-intensive traditional with lower broadband,” depending on the region of the feature space.

**Visualization:** “Level 2 – K-Means Subclusters (within DBSCAN Groups)” (PCA). Hover tooltips show country names.

### Level 3 — K-Medoids representatives

* Within each K-Means subcluster, we compute the **medoid** (the real country minimizing total distance to others under the selected norm).
* These medoids are **practical exemplars** for each sub-profile and good anchors for qualitative interpretation or case studies.

**Visualization:** “Level 3 – K-Medoids Representatives (PCA Projection)”, with **rings around medoids** and labels.

---

## 6) Experimental procedure (what we actually did)

1. **Fetch & merge** the nine indicators (2021) by Country Name from the World Bank.
2. **Clean** (strict NaN row drops, remove constants), **log1p** two skewed features, **Min-Max** to [0,1].
3. **Set** `SELECTED_NORM` (default **'2'**); all distances (DBSCAN metric, silhouette, medoids) follow this choice.
4. **Tune DBSCAN**: inspect k-distance plot; start in the **0.30–0.50** `eps` band; pick **`eps = 0.30`**, `min_samples = 3`.
5. **Filter to core points** (DBSCAN labels != −1).
6. **For each DBSCAN cluster**, fit **K-Means** with k in 2..5; select **best k by silhouette** (custom norm).
7. **For each K-Means subcluster**, compute **medoid** (custom norm).
8. **Visualize** all three levels in **PCA 2D** with **hover tooltips** and export **screenshots**.

---

## 7) Results and interpretation (how to read the plots)

* **DBSCAN (Level 1):**
  You’ll see **islands of dense countries**—often, one cluster of **high-connectivity + high-R&D economies**, another of **emerging adopters** with rising education/connectivity, and a third of **lower-infrastructure** profiles.
  **Noise points** are countries with **unusual mixes** (e.g., high electricity consumption but lower broadband, or very high renewables share but weak R&D, etc.). DBSCAN is intentionally conservative about forcing these into clusters.

* **K-Means (Level 2):**
  Inside each island, the subclusters reveal **finer signatures**—for instance, two countries may be in the same DBSCAN region but split by **education vs broadband intensity**, or **R&D spending vs high-tech export share**.

* **K-Medoids (Level 3):**
  The medoid per subcluster is your **go-to representative**. If you want to narrate the profile, start with the medoid’s indicator values and compare neighbors. This is also handy for **policy analogies**: “Country A resembles medoid M; reforms that worked for M may translate.”

Because all visuals are **PCA projections**, axes are not original features; they’re just a **faithful 2D shadow** of the high-dimensional geometry. Always interpret clusters by going back to the **feature averages** or medoids’ **actual indicator values**.

---

## 8) Reproducibility, files, and how to run

* **Data**: `WorldBankData/` (nine `API_{code}.csv` files; 2021 column used).
* **Screenshots**: `screenshots/` (DBSCAN level, K-Means subclusters, K-Medoids medoids).
* **Interactivity**: hover labels require `mplcursors` (installable via `pip install mplcursors`).
* **One-flag norm switch**: set `SELECTED_NORM` to `'1'`, `'2'`, or `'inf'`.
* **Determinism**: we fix `random_state=42` in PCA and K-Means for consistent runs.

---

## 9) Takeaways

* A **three-layer** unsupervised pipeline gives a **coarse-to-fine map** of global tech development using only **open World Bank data**.
* Our **scaling strategy** (log1p + Min-Max) solves the dominance problem and respects **low-end resolution**.
* The **norm switch** (L¹/L²/L∞) is **project-wide** and implemented **manually**, ensuring transparency.
* **Medoids** turn abstract clusters into **named exemplars**, making insights actionable.
* With annual updates, this becomes a **living dashboard** for **tech-readiness convergence/divergence** across countries.

> See the **`screenshots`** folder for the three figures (DBSCAN, K-Means subclusters, K-Medoids representatives). All inputs used are in **`WorldBankData`**.

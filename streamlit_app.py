from clustering_utils import cluster_texts, vectorize_texts
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")
OUT = "outputs"                 

# ╭──────────────────────  DATA LOAD (cached)  ─────────────────────────╮
@st.cache_data
def load_data():
    reviews_df = pd.read_csv(f"{OUT}/clean_reviews.csv")
    company_df = pd.read_csv(f"{OUT}/company_clusters_with_topics.csv")
    model_df   = pd.read_csv(f"{OUT}/model_results.csv")

    def _try_json(fn):
        try:
            with open(f"{OUT}/{fn}", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    cluster_terms  = _try_json("cluster_terms.json")
    lda_topics     = _try_json("lda_topics.json")
    cluster_labels = _try_json("cluster_labels.json")   # NEW – pretty names
    return reviews_df, company_df, model_df, cluster_terms, lda_topics, cluster_labels

reviews_df, company_df, model_df, cluster_terms, lda_topics, cluster_labels = load_data()

# ╭───────────────────  SIDEBAR NAVIGATION  ────────────────────────────╮
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "Data Overview",
        "Sentiment Analysis",
        "Clustering",
        "Topic Modeling",
        "Model Results",
        "Company Insight",
    ],
)

# ╭───────────────────────  PAGE 1 – OVERVIEW  ─────────────────────────╮
if page == "Data Overview":
    st.title("ITviec Reviews – Data Overview")
    st.markdown(f"**Total reviews:** {len(reviews_df):,}")
    st.markdown(f"**Unique companies:** {company_df['id'].nunique():,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Star Rating Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Rating", data=reviews_df, palette="viridis", ax=ax)
        ax.set_xlabel("Rating (1–5)")
        st.pyplot(fig)

    with col2:
        st.subheader("Recommend? (Yes / No)")
        fig, ax = plt.subplots()
        order = reviews_df["Recommend?"].value_counts().index
        sns.countplot(x="Recommend?", data=reviews_df, order=order, palette="Set2", ax=ax)
        st.pyplot(fig)

    st.subheader("Review Length (tokens)")
    fig, ax = plt.subplots()
    reviews_df["clean_review"].str.split().str.len().hist(bins=30, ax=ax)
    ax.set_xlabel("Tokens"); ax.set_ylabel("Count")
    st.pyplot(fig)

# ╭──────────────────────  PAGE 2 – SENTIMENT  ─────────────────────────╮
elif page == "Sentiment Analysis":
    st.title("Lexicon-Based Sentiment Analysis")
    counts = (
        reviews_df["sentiment"]
        .value_counts().reindex(["positive", "neutral", "negative"])
        .fillna(0).astype(int)
    )
    st.bar_chart(counts)
    st.write("**Proportions(%)**")
    st.write(((counts / counts.sum())*100).rename("proportion").to_frame())

    st.subheader("Random Examples")
    rng = np.random.default_rng(42)
    for sent in ["positive", "neutral", "negative"]:
        st.markdown(f"**{sent.title()}**")
        sample = reviews_df.loc[reviews_df["sentiment"] == sent, "review"]
        for txt in rng.choice(sample.values, size=min(3, len(sample)), replace=False):
            st.write(f"> {txt}")
        st.write("---")

# ╭──────────────────────  PAGE 3 – CLUSTERING  ────────────────────────╮
elif page == "Clustering":
    st.title("Company Clustering Analysis")

    if "cluster_best" not in company_df.columns:
        st.warning("Pre-computed clusters not found – running a quick clustering using TF-IDF.")
        if "doc" not in company_df.columns:
            st.error("No aggregated review text ('doc') found to run clustering.")
            st.stop()
        results = cluster_texts(company_df["doc"].fillna("").tolist())
        best = results.iloc[0]
        st.info(f"Temporary best model: {best.algorithm} with k={best.k} (sil={best.silhouette:.3f})")
        if best.algorithm == "KMeans":
            model = KMeans(n_clusters=int(best.k), n_init=10, random_state=42)
        elif best.algorithm == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=int(best.k), linkage="ward")
        else:
            model = GaussianMixture(n_components=int(best.k), random_state=42)
        X = vectorize_texts(company_df["doc"].fillna("").tolist())
        scaler = StandardScaler()
        company_df["cluster_best"] = model.fit_predict(scaler.fit_transform(X))

    algo = "cluster_best"
    st.info(f"Displaying results for the best identified algorithm, stored in the `{algo}` column.")

    # --- Display Cluster Sizes ---
    st.subheader("Company Count per Cluster")
    # Drop NA values before counting to avoid errors
    cluster_counts = company_df[algo].dropna().value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # --- PCA Scatter Plot ---
    st.subheader("PCA Visualization of Clusters")
    
    # Check if necessary PCA columns exist
    if {"ldaPC1", "ldaPC2"}.issubset(company_df.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot only companies that have been clustered
        plot_df = company_df.dropna(subset=[algo, 'ldaPC1', 'ldaPC2'])
        
        sns.scatterplot(data=plot_df, x="ldaPC1", y="ldaPC2",
                        hue=algo, palette="Set2", s=70, alpha=0.85, ax=ax)
        
        ax.set_title(f"PCA Visualization of Company Clusters")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)
    else:
        st.info("PCA columns (ldaPC1, ldaPC2) not found in the data.")

    # --- Signature Terms Display ---
    st.subheader("Signature Terms for Each Cluster")

    if cluster_terms and isinstance(cluster_terms, dict):
        # Iterate directly through the items in the loaded JSON file
        for key, words in cluster_terms.items():
            # Extract the cluster number from the key (e.g., "cluster_0_terms" -> "0")
            cluster_id_str = key.split('_')[1]
            st.markdown(f"**Cluster {cluster_id_str}:** {', '.join(words)}")
    else:
        st.info("No signature-term dictionary (cluster_terms.json) available or it is empty.")

# ╭──────────────────────  PAGE 4 – TOPIC MODEL ───────────────────────╮
elif page == "Topic Modeling":
    st.title("LDA Topic Modeling Analysis")    
    # Check for necessary data
    if not lda_topics:
        st.warning("lda_topics.json not found or is empty.")
        st.stop()
    if 'lda_topic' not in company_df.columns or 'cluster_best' not in company_df.columns:
        st.warning("The 'lda_topic' and/or 'cluster_best' columns were not found in the data file.")
        st.stop()

    # Display Top Words per LDA Topic
    st.subheader("Top Words per Discovered Topic")
    st.info("These are the fundamental topics the LDA model learned from the review text.")
    for topic, words in lda_topics.items():
        st.markdown(f"**{topic}:** {', '.join(words)}")

    # Show Topic Distribution within Final Clusters
    st.subheader("Topic Distribution within Final Clusters")
    st.info("This chart shows which topics are most prevalent in each of the final company clusters.")
    
    df_filtered = company_df.dropna(subset=['cluster_best', 'lda_topic'])

    if not df_filtered.empty:
        # Convert IDs to integers for cleaner plotting and labels
        df_filtered['cluster_best'] = df_filtered['cluster_best'].astype(int)
        df_filtered['lda_topic'] = df_filtered['lda_topic'].astype(int)
        
        # Create a cross-tabulation of clusters vs topics to see the distribution
        crosstab_df = pd.crosstab(
            df_filtered['cluster_best'],
            df_filtered['lda_topic']
        )
        
        if not crosstab_df.empty:
            # Rename index and columns for clarity in the chart
            crosstab_df.index.name = "Cluster"
            crosstab_df.columns.name = "Topic"
            
            # Plot the grouped bar chart using Streamlit's native function for interactivity
            st.bar_chart(crosstab_df)
        else:
            st.warning("Could not generate topic distribution chart. The data might be empty after filtering.")
    else:
        st.warning("No companies with both cluster and topic assignments found to create distribution chart.")


# ╭──────────────────────  PAGE 5 – MODEL RESULTS  ─────────────────────╮
elif page == "Model Results":
    st.title("Sentiment-Model Leaderboard")
    st.dataframe(model_df.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}"}))

# ╭──────────────────────  PAGE 6 – COMPANY INSIGHT  ───────────────────────╮
elif page == "Company Insight":
    st.title("🔍 Company-Level Insight")

    # This assumes 'company_df' and 'reviews_df' are loaded earlier in the script.

    # 1) Pick a company
    all_ids = company_df["id"].sort_values().unique()
    comp_id = st.number_input(
        "Enter company ID",
        min_value=int(all_ids.min()),
        max_value=int(all_ids.max()),
        # Use a default company ID that is known to have data for a better user experience
        value=6, 
        step=1,
    )
    if comp_id not in all_ids:
        st.error("Company ID not found.")
        st.stop()

    # 2) Filter the data
    comp_meta = company_df.loc[company_df["id"] == comp_id].iloc[0]
    comp_revs = reviews_df[reviews_df["id"] == comp_id]

    # Display company name header first
    st.header(f"{comp_meta.CompanyName}  (ID {comp_id})")

    # ### HIGHLIGHT: Added this check to handle companies with no reviews ###
    # Check if there are any reviews available for this company.
    if len(comp_revs) == 0:
        st.warning("Not enough data available for this company to generate insights.")
        # Stop executing the rest of the page.
        st.stop()

    # --- If there is data, the rest of the code below will run ---

    # 3) Basic stats
    st.markdown(f"- **Reviews analysed:** {len(comp_revs)}")
    # Safely display LDA topic by checking if it's not NaN
    if pd.notna(comp_meta.get("lda_topic")):
        st.markdown(f"- **LDA topic:** {int(comp_meta.lda_topic)}")
    if pd.notna(comp_meta.get("cluster_best")):
         st.markdown(f"- **Final Cluster:** {int(comp_meta.cluster_best)}")


    # 4) Sentiment distribution
    st.subheader("Sentiment Distribution")
    sent_counts = (
        comp_revs["sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"])
        .fillna(0)
        .astype(int)
    )
    st.bar_chart(sent_counts)

    # 5) Actionable Insights & Recommendations
    st.subheader("Actionable Insights & Recommendations")

    # Calculate distinctive keywords first to use them in recommendations
    top_kw = []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = reviews_df["clean_review"].fillna("").astype(str).tolist()
        comp_texts = comp_revs["clean_review"].fillna("").astype(str).tolist()
        if comp_texts:
            tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words=None)
            X_all = tfidf.fit_transform(corpus)
            X_comp = tfidf.transform(comp_texts)
            global_mean = X_all.mean(axis=0).A1
            comp_mean   = X_comp.mean(axis=0).A1
            delta       = comp_mean - global_mean
            top_idx = delta.argsort()[-10:][::-1]
            top_kw  = tfidf.get_feature_names_out()[top_idx]
    except (ValueError, IndexError):
        top_kw = [] # Ensure top_kw is a list even if TF-IDF fails

    # Logic to provide suggestions based on sentiment profile
    if sent_counts['positive'] > (sent_counts['negative'] + sent_counts['neutral']):
        st.success(
            "**You are doing great!** Employees consistently report positive experiences. "
            "To maintain this momentum, continue focusing on the aspects that are working well and keep listening to your team's feedback."
        )
    elif sent_counts['negative'] > sent_counts['positive']:
        st.error(
            "**Area for Improvement:** Reviews indicate significant challenges. "
            "Based on the feedback, focusing on the following themes could be beneficial:"
        )
        if top_kw:
            # Create a bulleted list from the keywords
            for keyword in top_kw:
                st.markdown(f"- {keyword.replace('_', ' ').capitalize()}")
        else:
            st.markdown("- General workplace culture and management.")
    else:
        st.info(
            "**Mixed Feedback:** The company has a neutral sentiment profile. While there are positive aspects, "
            "addressing the topics raised in neutral and negative reviews could significantly boost overall employee satisfaction."
        )

    # 6) Most frequent words
    st.subheader("Most Frequent Words")
    words = (
        pd.Series(" ".join(comp_revs["clean_review"].fillna("")).split())
        .value_counts()
        .head(15)
        .rename_axis("word")
        .reset_index(name="freq")
    )
    st.table(words)

    # 7) Distinctive keywords (TF-IDF)
    st.subheader("Distinctive keywords (TF-IDF)")
    # This section already handles the case with no reviews, which is good practice.
    corpus = reviews_df["clean_review"].fillna("").astype(str).tolist()
    comp_texts = comp_revs["clean_review"].fillna("").astype(str).tolist()

    if not comp_texts:
        st.info("No reviews for this company → cannot compute TF-IDF keywords.")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            stop_words=None
        )
        X_all = tfidf.fit_transform(corpus)
        try:
            X_comp = tfidf.transform(comp_texts)
            global_mean = X_all.mean(axis=0).A1
            comp_mean   = X_comp.mean(axis=0).A1
            delta       = comp_mean - global_mean
            top_idx = delta.argsort()[-10:][::-1]
            top_kw  = tfidf.get_feature_names_out()[top_idx]
            st.write("**Top 10 distinctive keywords:**")
            st.write(", ".join(top_kw))
        except ValueError:
            st.info("Not enough vocabulary overlap to compute distinctive keywords.")

# ╰────────────────────────────────────────────────────────────────────────────╯

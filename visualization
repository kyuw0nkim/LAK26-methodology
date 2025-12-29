import umap
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_3d_clusters(df, embeddings):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, metric='cosine', random_state=42)
    embed_3d = reducer.fit_transform(embeddings)
    
    plot_df = pd.DataFrame(embed_3d, columns=['x', 'y', 'z'])
    plot_df['topic'] = df['topic_final'].astype(str)
    plot_df['content'] = df['content_cleaned']
    
    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='topic', title="LAK 2026 Hierarchical BERTopic")
    fig.show()

def plot_lift_heatmap(df):
    labeled_df = df.dropna(subset=['label']).copy()
    labeled_df = labeled_df[~labeled_df['topic_final'].astype(str).str.contains("-1")]
    
    overall_ratio = labeled_df['label'].value_counts(normalize=True)
    ctab = pd.crosstab(labeled_df['topic_final'], labeled_df['label'])
    topic_ratio = ctab.div(ctab.sum(axis=1), axis=0)
    lift_table = topic_ratio.div(overall_ratio, axis=1).fillna(0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(lift_table, annot=True, cmap="coolwarm", vmin=0, vmax=3)
    plt.title('Topic-Label Lift Score')
    plt.show()

import numpy as np
from bertopic import BERTopic
import hdbscan

def run_hierarchical_bertopic(df, embeddings, min_topic_size=15):
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_topic_size, metric='euclidean', 
        cluster_selection_method='eom', prediction_data=True
    )

    # 1차 모델링
    topic_model = BERTopic(hdbscan_model=hdbscan_model, min_topic_size=min_topic_size, verbose=True)
    topics, probs = topic_model.fit_transform(df['content_cleaned'].tolist(), embeddings, y=df['semi_supervised_label'].values)
    
    df['topic_level1'] = topics
    df['initial_membership_strength'] = probs
    df['topic_final'] = topics # 기본값

    # 2차 계층적 분해 (가장 큰 토픽 대상)
    topic_info = topic_model.get_topic_info()
    actual_topics = topic_info[topic_info.Topic != -1]
    
    if not actual_topics.empty:
        largest_id = actual_topics.sort_values(by='Count', ascending=False).iloc[0]['Topic']
        mask = df['topic_level1'] == largest_id
        
        if mask.sum() > min_topic_size * 2:
            sub_model = BERTopic(hdbscan_model=hdbscan_model, min_topic_size=min_topic_size)
            sub_topics, _ = sub_model.fit_transform(
                df[mask]['content_cleaned'].tolist(), embeddings[mask.values]
            )
            df.loc[mask, 'topic_final'] = [f"{largest_id}_{st}" for st in sub_topics]
            
    return df, topic_model

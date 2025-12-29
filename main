import pandas as pd
from utils import get_device, clean_text
from fine_tuning import run_fine_tuning
from data_clustering import run_hierarchical_bertopic
from visualization import plot_3d_clusters, plot_lift_heatmap

# 설정
DATA_PATH = 'your_data.csv'
BASE_MODEL = 'jhgan/ko-sroberta-multitask'
MODEL_SAVE_PATH = './fine_tuned_model_lak'

def main():
    device = get_device()
    df = pd.read_csv(DATA_PATH)
    df['content_cleaned'] = df['content'].apply(clean_text)
    
    # 1. Fine-tuning
    model = run_fine_tuning(df, BASE_MODEL, MODEL_SAVE_PATH, device)
    
    # 2. Embedding 생성
    embeddings = model.encode(df['content_cleaned'].tolist(), show_progress_bar=True)
    
    # 3. Topic Modeling
    df, topic_model = run_hierarchical_bertopic(df, embeddings)
    
    # 4. Visualization
    plot_3d_clusters(df, embeddings)
    plot_lift_heatmap(df)
    
    # 5. Save
    df.to_csv('lak_final_results.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()

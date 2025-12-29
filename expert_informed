import os
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def run_fine_tuning(df, base_model_name, save_path, device):
    if os.path.exists(save_path):
        print(f"기존 모델 발견: {save_path}를 사용합니다.")
        return SentenceTransformer(save_path, device=device)

    print("Fine-tuning을 시작합니다 (20% Expert-hint).")
    df_labeled = df.dropna(subset=['label']).copy()
    
    # 20% 분리 (Stratified split)
    label_counts = df_labeled['label'].value_counts()
    valid_labels = label_counts[label_counts > 1].index
    df_labeled = df_labeled[df_labeled['label'].isin(valid_labels)]
    
    train_indices, _ = train_test_split(
        df_labeled.index, train_size=0.2, random_state=42, stratify=df_labeled['label']
    )
    train_df = df_labeled.loc[train_indices]

    # Triplet 생성 로직
    train_examples = []
    labels = train_df['label'].unique()
    for label in labels:
        pos = train_df[train_df['label'] == label]['content_cleaned'].tolist()
        neg_pool = train_df[train_df['label'] != label]['content_cleaned'].tolist()
        if len(pos) > 1 and neg_pool:
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    train_examples.append(InputExample(texts=[pos[i], pos[j], random.choice(neg_pool)]))

    model = SentenceTransformer(base_model_name, device=device)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model=model)
    
    model.fit(train_objectives=[(train_loader, train_loss)], epochs=1, warmup_steps=100)
    model.save(save_path)
    return model

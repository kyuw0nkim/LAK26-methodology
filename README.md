# LAK 26 Project Page (Work-in-Progress)
This is the project page for the paper "Productive Discussion Moves in Groups Addressing Controversial Issues"
This repository provides the code for the Dialogue-Centric Learning Analytics (DCLA) framework presented at LAK 2026.
The core of this work is a **Hybrid Approach** that bridges the gap between top-down expert coding and bottom-up data-driven discovery to analyze complex AI ethics discussions.

## 🛠️ Key Methodology: The Hybrid Approach

Our methodology integrates human expertise with machine learning to identify 14 distinct discussion moves across five categories.

### 1. Expert-Informed Fine-tuning (Top-Down)
To ensure the model captures the specific nuances of ethical dilemmas (such as *Emotional Expression* and *Ambiguity Acknowledgment*), we use expert domain knowledge:
* **Seed Labels**: A subset of the data (20%) is coded by experts.
* **Triplet Loss Optimization**: We fine-tune a **Korean Sentence-BERT (S-BERT)** model using these expert labels. This transforms the embedding space to group semantically similar discussion moves together based on expert intuition.

### 2. Data-Driven Discovery (Bottom-Up)
The fine-tuned model then acts as a sophisticated feature extractor for the entire dataset:
* **Hierarchical Semi-Supervised BERTopic**: We apply BERTopic to the optimized embeddings. By using the expert-informed labels as "hints" (Semi-supervised), the model discovers granular patterns that pure manual coding might miss.
* **Topic Decomposition**: For large, complex clusters (e.g., "Elaborating Ideas"), the framework performs a second-level hierarchical decomposition to reach a finer resolution of 14 specific moves.

### 3. Evaluation
Following the hybrid pipeline, we validate the discovered moves using Lift Scores. This statistical measure identifies which discussion moves are significantly over-represented in high-quality (high Integrative Complexity) discussion sessions, providing a quantitative link between dialogue patterns and learning productivity.

## 📂 Project Structure
```text
├── main.py              # Executes the full hybrid pipeline
├── fine_tuning.py       # Phase 1: Expert-Informed Fine-tuning (Triplet Loss)
├── topic_modeling.py    # Phase 2: Hierarchical Semi-Supervised BERTopic
├── utils.py             # Preprocessing & Environment setup
├── visualization.py     # Lift Score Analysis & 3D Clustering
└── requirements.txt     # Dependency list
```

## Quick Start
```
# Install dependencies
pip install -r requirements.txt

# Run the hybrid analysis pipeline
python main.py
```

## Citation
If you utilize this methodology in your research, please cite:
```
# To be Appeared
```

# Fake-News-Classifier-with-LLM-Explanation

1. Data Collection and Extraction
* Combined two sources:

* Kaggle Fake News Dataset (CSV format)

* GossipCop Dataset, extracted via article URLs and tweet IDs. The GossipCop articles were cleaned using custom scripts to extract titles, bodies, and metadata from web links and JSON tweet archives.

2. Data Cleaning and Labeling
   
 Removed null entries, dropped irrelevant columns like URLs and tweet IDs, and assigned binary labels:
 
 "real" or "fake". Merged the title and text columns into a single feature.

3. Text Normalization
   
Applied lowercasing, removed punctuation and special characters, and ensured text consistency.
  
Used spaCy and re for text preprocessing and lemmatization.

4. Tokenization Strategy

 For ML: Used TF-IDF on spaCy-processed text.
 
 For DL: Used Tokenizer from Keras and padded sequences to fixed length (300) for consistency across models.

5. Subject Category Mapping
   
The original subject column included 9+ noisy categories. Mapped them into 5 clean classes:

politics, entertainment, general, national, and world.

6. Label Encoding

Used LabelEncoder to convert subject classes and binary labels into numeric targets.

Encoders were saved using joblib for consistent inference.

7. Data Splitting
   
Split data into training and test sets using stratified sampling (train_test_split) to preserve class balance. Used a 70/30 split.

8. Machine Learning Models
   
* Trained 3 baseline models:

* Multinomial Naive Bayes

* Random Forest

* XGBoost
  
XGBoost performed the best on subject prediction (accuracy: ~94.4%) using TF-IDF features.

9. Deep Learning Models
    
Developed and compared 3 Keras models for subject classification:

* CNN

* BiLSTM

* GRU
* CNN achieved the best performance (~94.5% accuracy), especially on imbalanced categories.

10. Real vs Fake News Detection (DL)
    
Built an ensemble model combining BiLSTM and CNN layers. It outperformed traditional models in capturing sequential and semantic patterns in fake vs real news detection.

11. Model Evaluation and Selection

Evaluation metrics included:

* Accuracy

* F1 Score

* Classification Report

* Cross-validation (for ML)
  
CNN (for subject) and BiLSTM+CNN (for fake/real) were selected based on macro-averaged F1 scores.

12. Model and Tokenizer Saving
    
Saved:

* DL models as .h5 files

* Tokenizers and encoders using joblib and pickle
  
These were used during inference and deployment, ensuring full reproducibility.

13. LLM Explanation Layer
    
Integrated flan-t5-large from Hugging Face Transformers to explain predictions. 

Used prompt engineering to generate concise explanations for why an article is fake.

14. Prompt Design
    
Crafted dynamic prompts like:

"Article:\n{news_text}\n\nExplain factually why this article might be considered fake in the context of {subject} news."

Prompts are generated based on model predictions and passed to the LLM.

15. Gradio Web App Deployment
    
Built a user interface using Gradio. The app takes input text and outputs:

* Fake/real prediction

* Subject classification

* LLM-based factual explanation
  
The system is hosted and tested via Google Colab. Ready to migrate to Hugging Face Spaces for public access without Drive dependency.

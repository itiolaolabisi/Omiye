# Omiye
Omiye is a word-based dialect identification model developed to investigate the computational identification of Supare, a Yoruba variety spoken in Àkókò South West, Ondo State, Nigeria.
The project has two complementary components:
Omiye — a language/dialect identification model trained to distinguish Supare from Hausa, Igbo, and Yoruba.
External LID benchmark — an evaluation of pretrained AfroLID and CommonLingua to determine whether existing multilingual language identification systems classify Supare as Yoruba when Supare is not an explicit output class.
The distinction is important: Omiye treats Supare as a separate class, whereas the external models are evaluated in their pretrained form and therefore determine which of their existing language categories best matches Supare.
#Dataset
The final dataset contains four classes:
Hausa
Igbo
Supare
Yoruba
Supare data were manually created for this study. Hausa, Igbo, and Yoruba comparison data were obtained from the Masakhane repository.
Dataset construction pipeline
Manual Supare data
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
Sentence-level segmentation    Comparison data
        │                       from Masakhane
        │                          │
        └──────────────┬───────────┘
                       ▼
              Sentence-level CSV
                       │
                       ▼
                Word tokenization
                       │
                       ▼
             Remove punctuation
                 tokens/artifacts
                       │
                       ▼
              Combine four classes
                       │
                       ▼
              Train / Dev / Test

Sentence-level processing
Comparison-language data were initially read from Word documents and segmented at the sentence level. Each sentence was stored as an individual record in CSV format.
Word-level processing
The sentence-level data were subsequently tokenized into individual words.
This second tokenization step is important because Omiye is a word-based model. Words, rather than characters or complete sentences, constitute the fundamental textual units used for feature representation.
Punctuation tokens introduced during word tokenization were removed before constructing the final dataset.

#Test Set
The held-out test set contains 89 examples:
Class
Samples
Supare
40
Hausa
20
Igbo
17
Yoruba
12
Total
89

The test set is not artificially balanced.
The Supare subset contains 40 examples and is used for the dedicated Supare → Yoruba external-model analysis.
Note: Exact training and development set sizes should be taken from the final dataset/split files used for the experiment.

#Omiye Architecture
Word-level text
      │
      ▼
   TF-IDF
      │
      ▼
Logistic Regression
      │
      ▼
Predicted language/dialect

The classifier uses:
TF-IDF for word-level feature representation
Logistic Regression for classification
max_iter=2000
random_state=42

#Training and Evaluation
The combined dataset is divided into:
Training set
     │
     ├── used to train Omiye
     │
Development set
     │
     └── used during model development
     
Held-out test set
     │
     └── reserved for final evaluation

The test set is not used to train Omiye.
The trained model is subsequently used to generate predictions for the held-out test data.
The resulting predictions are stored separately from the original test labels.

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer
nltk.download("punkt")
nltk.download("punkt_tab")

# Load dataset
df = pd.read_csv("yoruba_tokenized_dataset.csv")

# Tokenize Prompt
df["tokens_tokens"] = df["tokens_tokens"].apply(
    lambda x: word_tokenize(str(x))
)

# Tokenize Completion
#df["Completion_tokens"] = df["Completion"].apply(
   # lambda x: word_tokenize(str(x))
#)

# Save output in the same folder
df.to_csv("Yoruba_tokenized_dataset.csv", index=False)

print("Done! Tokenized dataset saved as Yoruba_tokenized_dataset.csv")

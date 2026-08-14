from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "UBC-NLP/afrolid_1.5"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL
)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL
)

print("Model loaded successfully.")

text = "Mo fe jeun."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True
)

with torch.no_grad():

    outputs = model(
        **inputs
    )

scores = torch.softmax(
    outputs.logits,
    dim=1
)

predicted_id = torch.argmax(
    scores,
    dim=1
).item()

print("\n================================")
print("AFROLID TEST")
print("================================")

print(
    "Predicted class ID:",
    predicted_id
)

print(
    "Confidence:",
    scores[0][predicted_id].item()
)

print(
    "\nModel labels:"
)

print(
    model.config.id2label
)
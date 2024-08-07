import time
import torch
from classes import ReviewDataset
from model import ReviewClassifier
from helpers import preprocess_text
import json
from argparse import Namespace

# Record the start time
start_time = time.time()

args = Namespace(
    frequency_cutoff=25,
    model_state_file="model.pth",
    review_csv="data/processed/test_data_processed.csv",
    save_dir="model_storage/ch3/yelp/",
    vectorizer_file="vectorizer.json",
    cuda=True,
    batch_size=128,
    seed=1337,
)

if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")

with open(args.vectorizer_file, "r") as f:
    vectorizer_data = json.load(f)

dataset = ReviewDataset.load_dataset_and_load_vectorizer(
    args.review_csv, vectorizer_data
)

vectorizer = dataset.get_vectorizer()

model = ReviewClassifier(
    input_dim=len(vectorizer.review_vocab), hidden_dim=100, output_dim=1
).to(args.device)
model.load_state_dict(torch.load(args.model_state_file))
model.eval()


def preprocess_and_vectorize(text, vectorizer):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.vectorize(preprocessed_text)
    return vectorized_text


def predict(text):
    processed_text = preprocess_and_vectorize(text, vectorizer)

    processed_text = torch.tensor(processed_text).view(1, -1).to(args.device)

    model.eval()

    with torch.no_grad():
        probs = model(x_in=processed_text.float(), apply_sigmoid=True)

    prediction = (probs > 0.5).int()

    return probs.item(), prediction.item()


input_strings = [
    "I love this product! It works great.",
    "The service was terrible and the product was awful.",
    "It's Good but not good enough as a product.",
    "Needs a lot of improvements!",
    "A great product just need some improvments!"
]

for input_str in input_strings:
    prob, pred = predict(input_str)
    print(f"Input: '{input_str}'")
    print(f"Probability: {prob:.4f}")
    print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
    print()

# Record the end time
end_time = time.time()

# Compute and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")

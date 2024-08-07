from argparse import Namespace
import torch.optim as optim
import torch
import torch.nn as nn
from classes import ReviewDataset
from model import ReviewClassifier
from helpers import generate_batches, compute_accuracy
import json

args = Namespace(
    # Data and path information
    frequency_cutoff=25,
    model_state_file="model.pth",
    review_csv="data/processed/test_data_processed.csv",
    save_dir="model_storage/ch3/yelp/",
    vectorizer_file="vectorizer.json",
    save_vectorizer=False,
    # No model hyperparameters
    # Training hyperparameters
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    # Runtime options omitted for space
    cuda=True,
)


def make_train_state(args):
    return {
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": 1,
        "test_acc": 1,
    }


train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")

if args.save_vectorizer:
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    vectorizer = dataset.get_vectorizer()
    vectorizer_data = vectorizer.to_serializable()

    with open(args.vectorizer_file, "w") as f:
        json.dump(vectorizer_data, f, indent=4)

else:
    with open(args.vectorizer_file, "r") as f:
        vectorizer_data = json.load(f)
    dataset = ReviewDataset.load_dataset_and_load_vectorizer(
        args.review_csv, vectorizer_data
    )
    vectorizer = dataset.get_vectorizer()


model = ReviewClassifier(
    input_dim=len(vectorizer.review_vocab), hidden_dim=100, output_dim=1
).to(args.device)

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

for epoch_index in range(args.num_epochs):
    train_state["epoch_index"] = epoch_index

    dataset.set_split("train")
    batch_generator = generate_batches(
        dataset, batch_size=args.batch_size, device=args.device
    )

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    model.train()

    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()

        y_pred = model(x_in=batch_dict["x_data"].float())

        loss = loss_func(y_pred, batch_dict["y_target"].float())
        loss_batch = loss.item()
        running_loss += loss_batch

        loss.backward()

        optimizer.step()

        acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
        running_acc += acc_batch

        total_batches += 1

    train_loss = running_loss / total_batches
    train_acc = running_acc / total_batches
    train_state["train_loss"].append(train_loss)
    train_state["train_acc"].append(train_acc)

    print(
        f"Epoch {epoch_index+1}/{args.num_epochs} - "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
    )

    dataset.set_split("val")

    batch_generator = generate_batches(
        dataset, batch_size=args.batch_size, device=args.device
    )

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    model.eval()

    with torch.no_grad():
        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = model(x_in=batch_dict["x_data"].float())

            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_batch = loss.item()
            running_loss += loss_batch

            acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc += acc_batch

            total_batches += 1

        val_loss = running_loss / total_batches
        val_acc = running_acc / total_batches
        train_state["val_loss"].append(val_loss)
        train_state["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch_index+1}/{args.num_epochs} - "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

torch.save(model.state_dict(), args.model_state_file)

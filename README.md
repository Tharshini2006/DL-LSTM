# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Problem Statement:
To develop a deep learning model using LSTM to identify and classify named entities such as persons, organizations, geographical locations, time indicators, artifacts, events, and natural phenomena from text.

Dataset:
The model uses the ner_dataset.csv dataset containing words, sentence numbers, and corresponding NER tags. The dataset is preprocessed and converted into numerical sequences before training the LSTM model.


## DESIGN STEPS
## STEP 1:

Load the NER dataset and preprocess the text data by handling missing values and extracting words and tags.

## STEP 2:

Convert words and NER tags into numerical values using word-to-index and tag-to-index mappings.

## STEP 3:

Group the words according to their sentences and pad all sequences to a fixed maximum length.

## STEP 4:

Split the prepared dataset into training and testing data, then create Dataset and DataLoader objects.

## STEP 5:

Build a BiLSTM model using an embedding layer, bidirectional LSTM layer, and fully connected layer. Train the model using Cross Entropy Loss and Adam optimizer.

## STEP 6:

Evaluate the trained model using classification metrics, plot training and validation loss, and predict NER tags for sample text.





## PROGRAM

### Name: Nikila D

### Register Number: 212224230187

```python

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)



data = pd.read_csv(
    "ner_dataset.csv",
    encoding="latin1"
).ffill()

print("\nFirst 5 rows:")
print(data.head())



words = list(data["Word"].unique())
tags = list(data["Tag"].unique())

# Add padding word
if "ENDPAD" not in words:
    words.append("ENDPAD")

# Word to index
word2idx = {
    w: i + 1
    for i, w in enumerate(words)
}

# Tag to index
tag2idx = {
    t: i
    for i, t in enumerate(tags)
}

# Index to tag
idx2tag = {
    i: t
    for t, i in tag2idx.items()
}



print("\nUnique words in corpus:",
      data["Word"].nunique())

print("Unique tags in corpus:",
      data["Tag"].nunique())

print("Unique tags are:")
print(tags)


# ------------------------------------------------------------
# Step 6: Group Words into Sentences
# ------------------------------------------------------------

class SentenceGetter:

    def __init__(self, data):

        self.grouped = data.groupby(
            "Sentence #",
            group_keys=False
        ).apply(
            lambda s: [
                (w, t)
                for w, t in zip(
                    s["Word"],
                    s["Tag"]
                )
            ]
        )

        self.sentences = list(
            self.grouped
        )


getter = SentenceGetter(data)

sentences = getter.sentences


print("\nTotal number of sentences:",
      len(sentences))

print("\nExample sentence:")
print(sentences[0])



X = [
    [
        word2idx[w]
        for w, t in sentence
    ]
    for sentence in sentences
]

y = [
    [
        tag2idx[t]
        for w, t in sentence
    ]
    for sentence in sentences
]


print("\nEncoded first sentence:")
print(X[0])

print("\nEncoded tags:")
print(y[0])



plt.figure(figsize=(8, 5))

plt.hist(
    [len(s) for s in sentences],
    bins=50
)

plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.title("Sentence Length Distribution")

plt.grid(True)

plt.show()



max_len = 50

# Pad input sentences
X_pad = pad_sequence(
    [
        torch.tensor(seq)
        for seq in X
    ],
    batch_first=True,
    padding_value=word2idx["ENDPAD"]
)

# Pad target tags
y_pad = pad_sequence(
    [
        torch.tensor(seq)
        for seq in y
    ],
    batch_first=True,
    padding_value=tag2idx["O"]
)

# Limit maximum length to 50
X_pad = X_pad[:, :max_len]

y_pad = y_pad[:, :max_len]


print("\nPadded input:")
print(X_pad[0])

print("\nPadded target:")
print(y_pad[0])

print("\nX shape:", X_pad.shape)
print("Y shape:", y_pad.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y_pad,
    test_size=0.2,
    random_state=1
)

print("\nTraining samples:",
      len(X_train))

print("Testing samples:",
      len(X_test))


class NERDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return {
            "input_ids": self.X[idx],
            "labels": self.y[idx]
        }



train_dataset = NERDataset(
    X_train,
    y_train
)

test_dataset = NERDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)



class BiLSTMTagger(nn.Module):

    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=128,
        hidden_dim=128
    ):

        super(BiLSTMTagger, self).__init__()

        # Word Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=word2idx["ENDPAD"]
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Fully Connected Layer
        self.fc = nn.Linear(
            hidden_dim * 2,
            tagset_size
        )

    def forward(self, input_ids):

        # Convert word IDs to embeddings
        x = self.embedding(input_ids)

        # BiLSTM
        x, _ = self.lstm(x)

        # Classification layer
        x = self.fc(x)

        return x



model = BiLSTMTagger(
    vocab_size=len(word2idx) + 1,
    tagset_size=len(tag2idx),
    embedding_dim=128,
    hidden_dim=128
).to(device)


print("\nModel:")
print(model)


loss_fn = nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



def train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):


        model.train()

        total_train_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)

            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids)

            # Reshape output and labels
            outputs = outputs.view(
                -1,
                outputs.shape[-1]
            )

            labels = labels.view(-1)

            # Calculate loss
            loss = loss_fn(
                outputs,
                labels
            )

            # Clear gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_train_loss += loss.item()


        # Average training loss
        avg_train_loss = (
            total_train_loss /
            len(train_loader)
        )

        train_losses.append(
            avg_train_loss
        )



        model.eval()

        total_val_loss = 0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch[
                    "input_ids"
                ].to(device)

                labels = batch[
                    "labels"
                ].to(device)

                outputs = model(
                    input_ids
                )

                outputs = outputs.view(
                    -1,
                    outputs.shape[-1]
                )

                labels = labels.view(-1)

                loss = loss_fn(
                    outputs,
                    labels
                )

                total_val_loss += loss.item()


        # Average validation loss
        avg_val_loss = (
            total_val_loss /
            len(test_loader)
        )

        val_losses.append(
            avg_val_loss
        )


        # Print results
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )


    return train_losses, val_losses



def evaluate_model(
    model,
    test_loader,
    X_test,
    y_test
):

    model.eval()

    true_tags = []
    pred_tags = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            # Model prediction
            outputs = model(
                input_ids
            )

            # Get highest probability tag
            preds = torch.argmax(
                outputs,
                dim=-1
            )

            for i in range(
                len(labels)
            ):

                for j in range(
                    len(labels[i])
                ):

                    # Ignore padding
                    if (
                        labels[i][j].item()
                        != tag2idx["O"]
                    ):

                        true_tags.append(
                            idx2tag[
                                labels[i][j].item()
                            ]
                        )

                        pred_tags.append(
                            idx2tag[
                                preds[i][j].item()
                            ]
                        )



    print(
        classification_report(
            true_tags,
            pred_tags,
            zero_division=0
        )
    )


train_losses, val_losses = train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
)



evaluate_model(
    model,
    test_loader,
    X_test,
    y_test
)


print("\nName:                 ")
print("Register Number:      ")

history_df = pd.DataFrame({
    "loss": train_losses,
    "val_loss": val_losses
})

history_df.plot(
    title="Loss Over Epochs",
    figsize=(8, 5)
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)

plt.show()


# Select one test sentence
i = 125

model.eval()

# Add batch dimension
sample = X_test[i].unsqueeze(0).to(device)

with torch.no_grad():

    output = model(sample)


# Get predicted tags
preds = torch.argmax(
    output,
    dim=-1
).squeeze().cpu().numpy()


# Actual tags
true = y_test[i].numpy()


print("\nName:                 ")
print("Register Number:      ")

print(
    "\n{:<20} {:<15} {}".format(
        "Word",
        "True",
        "Predicted"
    )
)

print("-" * 55)


for w_id, true_tag, pred_tag in zip(
    X_test[i],
    y_test[i],
    preds
):

    # Ignore padding words
    if (
        w_id.item()
        != word2idx["ENDPAD"]
    ):

        # Convert ID back to word
        word = words[
            w_id.item() - 1
        ]

        # Convert ID back to true tag
        true_label = tags[
            true_tag.item()
        ]

        # Convert ID back to predicted tag
        pred_label = tags[
            pred_tag
        ]

        print(
            f"{word:<20} "
            f"{true_label:<15} "
            f"{pred_label}"
        )




```

### OUTPUT

## Loss Vs Epoch Plot
<img width="1227" height="123" alt="image" src="https://github.com/user-attachments/assets/279a18c3-2690-43c8-aacb-52c2230482b7" />
<img width="1212" height="657" alt="image" src="https://github.com/user-attachments/assets/74c5d468-a376-43f2-9342-15ef994fd9e3" />


### Sample Text Prediction
<img width="882" height="582" alt="image" src="https://github.com/user-attachments/assets/0d0238f1-3847-46c7-8596-5ecde51ed108" />



## RESULT
Thus, the LSTM-based NER model was successfully developed and trained to identify named entities from text.

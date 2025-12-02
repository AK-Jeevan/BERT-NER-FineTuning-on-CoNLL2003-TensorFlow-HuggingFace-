# This script Fine-tunes BERT for Named Entity Recognition (NER) on the CoNLL-2003 dataset using TensorFlow

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np

MODEL = "bert-base-cased"
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
LR = 2e-5

conll = load_dataset("conll2003")
label_names = conll["train"].features["ner_tags"].feature.names
num_labels = len(label_names)
print("NER Labels:", label_names)

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # mask out
            elif word_idx != previous_word_idx:
                label_ids.append(label_seq[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_conll = conll.map(tokenize_and_align_labels, batched=True)

# -----------------------------
# FORMAT FOR TENSORFLOW
# -----------------------------
tokenized_conll.set_format(
    type="tensorflow",
    columns=["input_ids", "attention_mask", "labels"]
)

def to_tf_dataset(dataset):
    features = {x: tf.constant(dataset[x]) for x in ["input_ids", "attention_mask"]}
    labels = tf.constant(dataset["labels"])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    return ds.shuffle(1024).batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = to_tf_dataset(tokenized_conll["train"])
val_ds   = to_tf_dataset(tokenized_conll["validation"])

model = TFAutoModelForTokenClassification.from_pretrained(MODEL, num_labels=num_labels)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# -----------------------------
# INFERENCE
# -----------------------------
def predict_ner(sentence):
    tokens = sentence.split()
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="tf", truncation=True)
    logits = model(enc).logits
    preds = tf.argmax(logits, axis=-1).numpy()[0]

    results = []
    for token, pred_id in zip(tokens, preds[1:len(tokens)+1]):  # skip [CLS]
        if pred_id != -100:
            label = label_names[pred_id]
            results.append((token, label))
    return results

example_sentence = "Barack Obama was born in Hawaii ."
print(predict_ner(example_sentence))

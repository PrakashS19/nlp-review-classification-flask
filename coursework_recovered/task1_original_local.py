#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Prakash Sudarsanraj
# #### Student ID: [redacted]
# 
# 
# ## Introduction
# This task focuses on basic natural language preprocessing using clothing review data. The main objective is to clean and prepare the review text so it can be used for feature generation and machine learning tasks in later stages of the assignment.
# 
# Several preprocessing techniques were applied to the `Review Text` column, including lowercasing, tokenization using regular expressions, stopword removal, removal of rare words, and removal of highly frequent words.
# 
# After preprocessing, the cleaned review text was saved into `processed.csv`, and a unigram vocabulary was generated and saved into `vocab.txt`. These outputs will be used in Task 2 and Task 3 for document vector generation and text classification experiments.

# ## Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter


# ### 1.1 Examining and loading data
# - Examine the data and explain your findings
# - Load the data into proper data structures and get it ready for processing.

# In[2]:


#Load dataset
base_path = Path(".")

data_path = base_path / "assignment3.csv"
stopwords_path = base_path / "stopwords_en.txt"

df = pd.read_csv(data_path)

print(df.shape)
print(df.columns.tolist())

df.head()


# The dataset contains customer clothing reviews collected from an online shopping platform.
# 
# For this assignment, the main columns used are:
# - `Review Text` for text preprocessing
# - `Title` for later classification experiments
# - `Recommended IND` as the target label
# 
# The dataset was loaded into a pandas DataFrame to prepare it for preprocessing and feature generation.

# In[3]:


#Inspect missing values
print(df.isna().sum())


# The dataset contains some missing values, mainly in the text-related columns. These missing values will be handled during preprocessing by converting them into empty strings before tokenization.

# ### 1.2 Pre-processing data
# In this section, the `Review Text` column is preprocessed to prepare it for text analysis and machine learning tasks.
# 
# The preprocessing steps include:
# 1. Converting text to lowercase
# 2. Tokenization using the required regular expression
# 3. Removing single-character words
# 4. Removing stopwords
# 5. Removing words appearing only once
# 6. Removing the top 20 most frequent words based on document frequency
# 
# The cleaned reviews will then be used to build the vocabulary and feature representations.

# In[4]:


#Load stopwords
with open(stopwords_path, "r", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())

print("Number of stopwords:", len(stopwords))


# The provided English stopword list was loaded and converted into a set for faster lookup during preprocessing.

# In[5]:


#Tokenization function
token_pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"

def tokenize_text(text):
    text = str(text).lower()
    tokens = re.findall(token_pattern, text)
    return tokens


# The required regular expression was used for tokenization.
# 
# This regex captures alphabetic words and also preserves words containing apostrophes or hyphens such as:
# - don't
# - well-made

# In[6]:


#Tokenize review text
df["tokens_raw"] = df["Review Text"].fillna("").apply(tokenize_text)

df[["Review Text", "tokens_raw"]].head()


# All review text was converted to lowercase and tokenized into individual words using the specified regex pattern.

# In[7]:


#Removing single characters and stopwords
df["tokens_step5"] = df["tokens_raw"].apply(
    lambda tokens: [
        word for word in tokens
        if len(word) > 1 and word not in stopwords
    ]
)

df[["tokens_raw", "tokens_step5"]].head()


# Single-character words and stopwords were removed because they usually provide little meaningful information for text classification tasks.

# In[8]:


#Removing words appearing once
term_freq = Counter()

for tokens in df["tokens_step5"]:
    term_freq.update(tokens)

words_once = {
    word for word, count in term_freq.items()
    if count == 1
}

df["tokens_step6"] = df["tokens_step5"].apply(
    lambda tokens: [
        word for word in tokens
        if word not in words_once
    ]
)

print("Words appearing once:", len(words_once))


# Words that appeared only once in the entire document collection were removed because they are unlikely to contribute meaningful patterns for classification.

# In[9]:


#Remove top 20 document-frequency words
doc_freq = Counter()

for tokens in df["tokens_step6"]:
    doc_freq.update(set(tokens))

top20_df_words = {
    word for word, count in doc_freq.most_common(20)
}

df["processed_tokens"] = df["tokens_step6"].apply(
    lambda tokens: [
        word for word in tokens
        if word not in top20_df_words
    ]
)

print("Top 20 frequent words removed:")
print(top20_df_words)


# The top 20 most frequent words based on document frequency were removed because extremely common words tend to dominate the dataset and provide limited discriminative value.

# In[10]:


#Create processed review text
df["processed_review"] = df["processed_tokens"].apply(
    lambda tokens: " ".join(tokens)
)

df[["Review Text", "processed_review"]].head()


# The cleaned tokens were joined back into processed review text. This processed version will be used for feature generation and classification tasks.

# ## Saving required outputs
# After preprocessing, the cleaned review data and vocabulary were saved in the required output formats.

# In[11]:


#Save processed.csv
processed_df = df.copy()

processed_df.to_csv("processed.csv", index=False)

print("processed.csv saved successfully")


# In[12]:


#Build vocabulary
vocab = sorted(
    set(
        word
        for tokens in df["processed_tokens"]
        for word in tokens
    )
)

word_to_index = {
    word: idx
    for idx, word in enumerate(vocab)
}

print("Vocabulary size:", len(word_to_index))


# The vocabulary was created from the processed review tokens.
# 
# All vocabulary words were sorted alphabetically, and integer indices were assigned starting from 0 as required by the assignment specification.

# In[13]:


#Save vocab.txt
with open("vocab.txt", "w", encoding="utf-8") as f:
    for word, idx in word_to_index.items():
        f.write(f"{word}:{idx}\n")

print("vocab.txt saved successfully")


# In[14]:


#Final validation
print("processed.csv exists:", Path("processed.csv").exists())
print("vocab.txt exists:", Path("vocab.txt").exists())

with open("vocab.txt", "r", encoding="utf-8") as f:
    for _ in range(10):
        print(f.readline().strip())


# ## Summary
# In this task, the clothing review text was preprocessed using multiple NLP preprocessing techniques including tokenization, stopword removal, rare word filtering, and document frequency filtering.
# 
# The processed review data was saved into `processed.csv`, and a cleaned unigram vocabulary was generated and saved into `vocab.txt`.
# 
# These outputs will be used in the next tasks for feature representation and machine learning classification experiments.

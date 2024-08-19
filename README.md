
# NLP Sentence Tokenization and Embedding with SentenceTransformer

This repository contains two Jupyter notebooks that demonstrate the use of the `SentenceTransformer` class from the `sentence_transformers` library to perform tokenization, sentence embedding, and similarity computation on sentences using the `all-MiniLM-L6-v2` model.

## Notebook 1: Tokenization and Sentence Embedding (sentence_tokenizer.ipynb)

### Overview

In this notebook, we import the `SentenceTransformer` class and use the pre-trained `all-MiniLM-L6-v2` model to perform tokenization on a single sentence. We also convert the tokens into their respective IDs, obtain the numerical embeddings, and decode the tokens back to the original sentence.

### Steps

1. **Importing Dependencies:**
   - `SentenceTransformer` class is imported from the `sentence_transformers` library.
   
2. **Model Loading:**
   - Load the pre-trained `all-MiniLM-L6-v2` model using the `SentenceTransformer` class.
   
3. **Tokenization:**
   - Tokenize a given sentence into subwords.
   
4. **Token ID Conversion:**
   - Convert the tokens to their corresponding IDs.
   
5. **Sentence Embedding:**
   - Generate a numerical embedding for the sentence.
   
6. **Decoding:**
   - Decode the token IDs back to the original sentence.

### Key Code Snippets

- Tokenization:
    ```python
    tokens = model.tokenizer.tokenize(sentence)
    ```

- Sentence Embedding:
    ```python
    embeddings = model.encode(sentence)
    ```

- Decoding:
    ```python
    decoded_tokens = model.tokenizer.decode(ids)
    ```

### Output

- The notebook outputs the tokens, token IDs, the first 20 values of the embedding, and the decoded sentence.

## Notebook 2: Sentence Similarity Computation (similarity_measures.ipynb)

### Overview

In this notebook, we extend the previous example to multiple sentences. We tokenize a list of sentences, generate their embeddings, and compute the cosine similarity between each pair of sentences. The similarity scores are then organized in a DataFrame for better visualization.

### Steps

1. **Importing Dependencies:**
   - `SentenceTransformer` class and `util` function are imported from the `sentence_transformers` library.

2. **Model Loading:**
   - Load the pre-trained `all-MiniLM-L6-v2` model using the `SentenceTransformer` class.
   
3. **Tokenization:**
   - Tokenize multiple sentences into subwords.
   
4. **Sentence Embedding:**
   - Generate numerical embeddings for all the sentences.
   
5. **Cosine Similarity Calculation:**
   - Calculate the cosine similarity between each pair of sentence embeddings.
   
6. **Pairwise Similarity Sorting:**
   - Find and sort the sentence pairs based on their cosine similarity scores.

7. **Visualization:**
   - Create a Pandas DataFrame to display the cosine similarity matrix.

### Key Code Snippets

- Cosine Similarity:
    ```python
    cosine_scores = util.cos_sim(embeddings, embeddings)
    ```

- Pairwise Similarity Sorting:
    ```python
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    ```

- DataFrame Creation:
    ```python
    similarity_df = pd.DataFrame(cosine_scores)
    ```

### Output

- The notebook outputs the tokens, embeddings, cosine similarity scores, and the sorted list of sentence pairs based on similarity. Additionally, it displays a DataFrame containing the cosine similarity matrix for all sentence pairs.

## Installation

To run the notebooks, you need to have Python installed along with the necessary packages. You can install the required libraries using pip:

```bash
pip install sentence-transformers pandas
```

## Usage

Clone the repository and open the notebooks in Jupyter Notebook or Jupyter Lab. Execute the cells to see the results.

```bash
git clone <repository-url>
cd <repository-folder>
jupyter notebook
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

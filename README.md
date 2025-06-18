# AbstractLLaMA
LLM for generating an Abstract based on the paper's title

# Metrics

We generated an abstract for 100 random article titles from a dataset test sample using our model, as well as the latest version of ChatGPT and LLaMA 405b. To evaluate the similarity of the generated text with the real abstract from the sample, we use the BERT vectorizer and the cosine distance between the vectors. The smallest distance value corresponds to the best result. Based on the metric values obtained, we can judge that our model is superior to LLaMA 405b and slightly inferior to ChatGPT, while having only 32M parameters, wich is ~12656 times less.

|       Model      |  Value |
|:----------------:|:------:|
| ChatGPT 4 Latest | 0.0617 |
| LLaMA 405b       | 0.0715 |
| Ours             | 0.0626 |

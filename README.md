# AbstractLLaMA

<p align="center">
  <img src="https://github.com/GlebBrykin/AbstractLLaMA/blob/main/.content/Screenshot.jpg" alt="Screenshot.jpg">
</p>

**Title**: _AbstractLLaMA: Title-Based Abstract Generator for arXiv Papers_

**Abstract (from our AbstractLLaMA)**:

> The recent breakthroughs in large language models (LLMs) have led to significant progress in various natural language processing tasks, including generating titles for specific questions. However, the absence of a comprehensive taxonomy, which covers a broad range of questions, has limited the effectiveness of these models. In this paper, we propose AbstractLLaMA, a novel framework that leverages abstract generators to enhance LLMs' performance in title-based abstract generation. AbstractLLaMA leverages the fact-checking capabilities of LLMs to identify and identify critical elements and abstract steps in the title. By applying this abstraction to the generated titles, AbstractLLaMA generates more accurate and diverse titles. To address the inherent issue of the generation process, AbstractLLaMA employs a multi-stage training strategy that iteratively fine-tunes the LLM with abstract and high-quality titles. Our extensive experiments on three real-world datasets demonstrate that AbstractLLaMA significantly outperforms existing baselines, achieving improvements of 2.7% on F1-score.

# Metrics

We generated an abstract for 100 random article titles from a dataset test sample using our model, as well as the latest version of ChatGPT and LLaMA 405b. To evaluate the similarity of the generated text with the real abstract from the sample, we use the BERT vectorizer and the cosine distance between the vectors. The smallest distance value corresponds to the best result. Based on the metric values obtained, we can judge that our model is superior to LLaMA 405b and slightly inferior to ChatGPT, while having only 32M parameters, wich is ~12656 times less.

|       Model      |  Value |
|:----------------:|:------:|
| ChatGPT 4 Latest | 0.0617 |
| LLaMA 405b       | 0.0715 |
| Ours             | 0.0626 |

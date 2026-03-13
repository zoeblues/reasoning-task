# Word2Vec from scratch with SGNS 
Implementation of Word2Vec using only NumPy. This project implements the Skip-gram architecture with Negative Sampling (SGNS) as introduced by Mikolov et al. (2013). Presented solution is one of the tasks specifically for the internship.

# Explanation of my design choices
Following the methodologies in "Efficient Estimation of Word Representations in Vector Space" and "Distributed Representations of Words and Phrases and their Compositionality", I chose Skip-gram with Negative Sampling (SGNS) for the following reasons:
- rare medical terms: skip-gram is much better at representing rare words than CBOW, which smooths over context. In medical datasets specific desease names apper very rarely and it is very important to catch the semantic value,
- computational efficiency: by implementing Negative Sampling, the function simplifies to a binary logistic regression for each sample,
- subsampling: in medical texts there are many frequent words like "patient", "the" and "trial". According to the methodology proposed in the 2013 Mikolov et al. paper, I implemented a subsampling threshold to discard these high-frequency words. This not only accelerates training but also significantly improves the quality of embeddings for rare, high-value clinical terms.

# Features
- No ML frameworks, pure NumPy library used to derive and implement all gradients
- Unigram Table, which implements the 3/4 power noise distribution for negative sampling. It was an empirical discovery that outperformed both unigram and uniform distributions.
- Vectorized Gradients to simulate batch like efficiency

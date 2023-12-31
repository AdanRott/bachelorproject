**Note: The remaining models and code will be added soon. Currently, the best performing model and the main functions are here.**

# The Potential of Data-driven Fingerprints
## With A Focus on Graph and String-Based Representational Methods

### Author: Adan Rotteveel

---
### Abstract
In the field of Artificial Intelligence, one of the key challenges is to accurately convert real-world elements and occurrences, such as molecules into numerical representations for computational analysis. Having the right numerical representation is particularly crucial in environmental chemistry, in which understanding the biological activity of chemicals is vital for human and environmental safety. Computational methods for predicting biological activity rely on converting molecules to numerical representations and then applying predictive models to them. Numerical representations used, such as descriptors, are often unstable and computationally intensive. Fingerprints, another type of numerical representation, are often context-specific and lack generalisability, and in the case of non-hashed fingerprints, they lack interpretability. As such, an alternative approach is suggested, which makes use of unsupervised machine learning for generating new numerical representations: a data-driven fingerprint.

To achieve this aim, this research first examines whether molecules should be viewed as sequential or graph-based data for machine learning. It then identifies the most effective unsupervised learning methods. The research applies these methods to create three unique models capable of generating new molecular fingerprints and evaluates their generalisability across different prediction tasks. The study shows that the sequential SMILES approach using masked language modeling outperforms other methods, suggesting a new direction for generating molecular representations in environmental chemistry and related fields.

### Supervisors
- Dr. Saer Samanipour
- Viktoriia Turkina

### Institution
Analytical Chemistry group (HIMS), Environmental Modeling & Computational Mass Spectrometry, Faculty of Science, University of Amsterdam

### Thesis Details
- **Program**: Bachelor Artificial Intelligence
- **Semester**: 1, 2023-2024
- **Submission Date**: 21st of November 2023
---

### Repository Contents
- Figures, including ICNTS 23 poster
- Data
- Code


### Usage
Currently code is still added to this repository.
But the best performing model can be used by uploading it into google colab and running it! 
Furthermore, are the amide data sets and acute fish toxicity datasets included.
For  validation of the results the fingerprint of the masked language model is already included in: X_train, X_test, y_train, y_test!


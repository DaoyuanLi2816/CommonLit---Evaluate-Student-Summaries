# CommonLit - Evaluate Student Summaries

This repository contains the code and top-scoring notebook from my participation in the Kaggle competition "CommonLit - Evaluate Student Summaries". The goal of the competition was to assess the quality of summaries written by students in grades 3-12. Our models evaluate how well a student represents the main idea and details of a source text, as well as the clarity, precision, and fluency of the language used in the summary.

## Competition Overview

- **Competition Host**: CommonLit, The Learning Agency Lab, Vanderbilt University, Georgia State University
- **Start Date**: July 12, 2023
- **End Date**: October 11, 2023
- **Evaluation Metric**: MCRMSE (Mean Columnwise Root Mean Squared Error)

## Notebook Overview

This notebook includes the following key components:

1. **Libraries and Environment Setup**: Installation of necessary libraries including `autocorrect` and `pyspellchecker`.

2. **Data Loading**: Loading of training and testing datasets, including `prompts_train.csv`, `prompts_test.csv`, `summaries_train.csv`, and `summaries_test.csv`.

3. **Preprocessing**: 
   - **Tokenization**: Tokenizing text data using `nltk` and `spacy`.
   - **N-gram Overlap Calculation**: Counting unigram, bigram, trigram, and four-gram overlaps between student summaries and prompts.
   - **NER Overlap**: Calculating named entity recognition overlaps using `spacy`.
   - **Spelling Correction**: Correcting spelling errors using `autocorrect` and `pyspellchecker`.
   - **Difficult Words Analysis**: Identifying and analyzing difficult words in summaries.
   - **Sentence Type Analysis**: Classifying sentences into simple, compound, and complex types.
   - **Repetition Count**: Counting repetitions of n-grams in summaries.

4. **Feature Engineering**: Creating features based on the preprocessing steps to be used for model training.

5. **Model Training and Evaluation**:
   - **Cross-Validation**: Using Group K-Fold cross-validation to ensure robust evaluation.
   - **Transformer Models**: Utilizing transformer models like `DeBERTa` and `RoBERTa` for sequence classification tasks.
   - **LightGBM Models**: Combining transformer embeddings with LightGBM for enhanced prediction performance.

6. **Prediction**: 
   - **Inference**: Making predictions on the test set using the trained models.
   - **Ensembling**: Combining predictions from multiple models to generate the final output.

## Data

The competition provided a dataset of real student summaries, which was used to train and evaluate the models.

## Main Approach

### Preprocessing and Feature Engineering

- **Tokenization**: Tokenized text using `nltk` and `spacy` to create word tokens and sentence tokens.
- **N-gram Overlap**: Calculated n-gram overlaps between summaries and prompts to measure content similarity.
- **NER Overlap**: Used named entity recognition to identify and count overlapping named entities between summaries and prompts.
- **Spelling Correction**: Corrected spelling errors in summaries to improve text quality.
- **Difficult Words Analysis**: Analyzed the presence and overlap of difficult words in summaries and prompts.
- **Sentence Type Classification**: Classified sentences into simple, compound, and complex types and calculated their ratios.
- **Repetition Count**: Counted repetitions of bigrams and trigrams to measure redundancy in summaries.

### Model Building

- **Transformer Models**: Fine-tuned transformer models such as `DeBERTa` and `RoBERTa` for the task of sequence classification.
- **LightGBM**: Combined transformer embeddings with LightGBM models to enhance prediction accuracy.

### Training and Evaluation

- **Cross-Validation**: Applied Group K-Fold cross-validation to ensure robust model evaluation and prevent overfitting.
- **Evaluation Metric**: Used MCRMSE to evaluate the model performance on both training and test sets.

### Prediction

- **Inference**: Generated predictions on the test set using the trained models.
- **Ensembling**: Combined predictions from multiple folds and models to create a final submission file.

## Results

The models achieved competitive performance, resulting in bronze medals for the submissions.

## Acknowledgments

Special thanks to CommonLit, The Learning Agency Lab, Vanderbilt University, Georgia State University, the Walton Family Foundation, and Schmidt Futures for their support.

For more details about the competition, please visit the [Kaggle competition page](https://kaggle.com/competitions/commonlit-evaluate-student-summaries).

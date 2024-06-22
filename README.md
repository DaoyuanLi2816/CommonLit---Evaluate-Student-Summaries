
# CommonLit - Evaluate Student Summaries

This repository contains the code and top-scoring notebook from my participation in the Kaggle competition "CommonLit - Evaluate Student Summaries". The goal of the competition was to assess the quality of summaries written by students in grades 3-12. Our models evaluate how well a student represents the main idea and details of a source text, as well as the clarity, precision, and fluency of the language used in the summary.

![Intro](./212749168-86d6c7ab-98da-409b-998f-c5b74721badd.gif)

## Competition Overview

- **Competition Host**: CommonLit, The Learning Agency Lab, Vanderbilt University, Georgia State University
- **Start Date**: July 12, 2023
- **End Date**: October 11, 2023
- **Evaluation Metric**: MCRMSE (Mean Columnwise Root Mean Squared Error)

## Main Approach

### Preprocessing and Feature Engineering

**Tokenization and Text Cleaning**:
- We used `transformers` and `nltk` libraries to tokenize the text data. The `AutoTokenizer` from the `transformers` library handled different transformer models effectively.
- Spelling corrections were performed using `autocorrect` and `pyspellchecker` to ensure the quality of the text.

**N-gram Overlap Calculation**:
- We calculated unigram, bigram, and trigram overlaps between the student summaries and prompts. This involved generating n-grams from tokens and counting the overlaps, which helped measure the content similarity.

**Additional Features**:
- We computed additional features such as length ratios, word overlap counts, and n-gram overlap ratios to enrich the input data for the models.

### Model Building

**Transformer Models**:
- Fine-tuned transformer models such as `DeBERTa` and `RoBERTa` for the sequence classification tasks. These models leveraged pre-trained embeddings and were fine-tuned on our specific dataset to predict the quality of summaries based on the processed features.

**LightGBM**:
- We combined transformer model embeddings with LightGBM for regression tasks. This hybrid approach enhanced the prediction accuracy by leveraging both deep learning and gradient boosting techniques.

### Training and Evaluation

**Cross-Validation**:
- Applied Group K-Fold cross-validation to ensure robust model evaluation and prevent overfitting. This method involved splitting the dataset into multiple folds and iteratively training the model on each fold.

**Evaluation Metric**:
- Used MCRMSE (Mean Columnwise Root Mean Squared Error) to evaluate the model performance on both training and test sets. MCRMSE is particularly suitable for multi-output regression tasks like ours, where we predict multiple quality metrics for each summary.

### Prediction and Ensembling

**Inference**:
- Generated predictions on the test set using the trained models. Predictions from multiple models and folds were combined to improve robustness and accuracy.

**Ensembling**:
- Combined the predictions from different models and folds to create the final submission file. This involved averaging the predictions to get the final scores for each summary.

## Detailed Code Explanation

### Preprocessing

In the preprocessing stage, we defined a `Preprocessor` class to handle tasks such as text tokenization, spelling correction, and n-gram overlap calculations.

```python
class Preprocessor:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.twd = TreebankWordDetokenizer()
        self.STOP_WORDS = set(stopwords.words('english'))
        
        self.spacy_ner_model = spacy.load('en_core_web_sm')
        self.speller = Speller(lang='en')
        self.spellchecker = SpellChecker()
```
In the initialization method, we load the tokenizer for the specified model and necessary tools like stop words, NER model, and spelling correction tools.

```python
    def word_overlap_count(self, row):
        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(lambda word: word in the STOP_WORDS, prompt_words))
            summary_words = list(filter(lambda word: word in the STOP_WORDS, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))
```
This method calculates the word overlap between prompt and summary tokens by filtering out stop words and computing the intersection.

```python
    def preprocess(self, input_df):
        input_df['summary_tokens'] = input_df['summary'].apply(self.tokenizer.tokenize)
        input_df['prompt_tokens'] = input_df['prompt'].apply(self.tokenizer.tokenize)
        input_df['summary_length'] = input_df['summary_tokens'].apply(len)
        input_df['prompt_length'] = input_df['prompt_tokens'].apply(len)
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']
        input_df['word_overlap_count'] = input_df.apply(self.word_overlap_count, axis=1)
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])
```
The `preprocess` method tokenizes the input data, calculates lengths, length ratios, and word overlap counts, then drops the temporary token columns.

### Model Training and Evaluation

In the model training and evaluation stage, we defined a `train_and_predict` function that performs model training, validation, and prediction.

```python
def train_and_predict(train_df, test_df, target, save_each_model, model_name, hidden_dropout_prob, attention_probs_dropout_prob, max_length):
    train_df["fold"] = -1
    gkf = GroupKFold(n_splits=CFG.n_splits)
    for fold, (_, val_idx) in enumerate(gkf.split(train_df, groups=train_df["prompt_id"])):
        train_df.loc[val_idx, "fold"] = fold
```
First, we use Group K-Fold cross-validation to split the dataset, ensuring that data from the same group does not appear in both training and validation sets simultaneously.

```python
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        train_data = train_df[train_df.fold != fold]
        valid_data = train_df[train_df.fold == fold]
        
        if save_each_model:
            model_dir = f"/kaggle/input/{model_name}-{target}-fold-{fold}/{target}/"
        else:
            model_dir = f"/kaggle/input/commonlistdebertatuned/{model_name}/fold_{fold}"
        
        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )
        
        csr.model.train()
        train_dataset = Dataset.from_pandas(train_data)
        tokenized_train = train_dataset.map(csr.preprocess_function, batched=True)
        valid_dataset = Dataset.from_pandas(valid_data)
        tokenized_valid = valid_dataset.map(csr.preprocess_function, batched=True)
```
Next, for each fold, we create and configure a `ContentScoreRegressor` object, load the corresponding pre-trained model and configuration, and preprocess the training and validation datasets.

```python
        training_args = TrainingArguments(
            output_dir=f"./results/{model_name}-{target}-fold-{fold}",
            evaluation_strategy="epoch",
            learning_rate=CFG.learning_rate,
            per_device_train_batch_size=CFG.batch_size,
            per_device_eval_batch_size=CFG.batch_size,
            num_train_epochs=CFG.num_train_epochs,
            weight_decay=CFG.weight_decay,
            logging_dir=f"./logs/{model_name}-{target}-fold-{fold}",
            logging_steps=10,
            save_steps=CFG.save_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
        )
        
        trainer = Trainer(
            model=csr.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            tokenizer=csr.tokenizer,
            compute_metrics=csr.compute_metrics,
        )
        
        trainer.train()
        trainer.save_model(f"./{model_name}-{target}-fold-{fold}-model")
        
        pred = csr.predict(test_df=valid_data, fold=fold)
        train_df.loc[valid_data.index, f"{target}_{model_name}"] = pred

    return train_df
```
We then configure training arguments and create a `Trainer` object to manage the training process. After training, we save the best model and predict on the validation set.

### Prediction and Ensembling

In the prediction stage, we define a `predict` function to generate predictions for the test set.

```python
def predict(test_df, target, save_each_model, model_name, hidden_dropout_prob, attention_probs_dropout_prob, max_length):
    for fold in range(CFG.n_splits):
        model_dir = f"/kaggle/input/{model_name}-{target}-fold-{fold}/{target}/"
        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )
        pred = csr.predict(test_df=test_df, fold=fold)
        test_df[f"{target}_{model_name}_{fold}"] = pred
    
    test_df[f"{target}_{model_name}"] = test_df[[f"{target}_{model_name}_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)
    return test_df
```
This function loads the model for each fold, predicts on the test data, stores the predictions in the test dataframe, and averages the predictions across all folds to get the final result.


## Results

The models achieved competitive performance, resulting in a bronze medal for the submissions.

## Acknowledgments

Special thanks to CommonLit, The Learning Agency Lab, Vanderbilt University, Georgia State University, the Walton Family Foundation, and Schmidt Futures for their support.

## Author

Daoyuan Li - [Kaggle Profile](https://www.kaggle.com/distiller)

I am a Competitions Expert on Kaggle, with multiple competition participations and several bronze medals.

For more details about the competition, please visit the [Kaggle competition page](https://kaggle.com/competitions/commonlit-evaluate-student-summaries).

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

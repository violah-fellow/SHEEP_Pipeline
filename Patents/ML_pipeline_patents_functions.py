## This file contains all functions necessary to run the ML pipeline for publication classification

## Necessary packages
import pandas as pd
import numpy as np
import os
import warnings

import torch
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
import joblib
from tqdm import tqdm

## load_patentsberta
## loads the embedding model SPETER2 with the classification adapter
## this model needs a different loading process than simple models from SentenceTransformers
## two outputs --> call like this: model, tokenizer = load_specter(...)
def load_patentsberta():
    # load model
    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

    return model

## check_token_size 
## counts number of tokens per entry and saves a 'truncated' = True/False column to the provided dataframe if add_column = True
def check_token_size(data, text_column, model=None, add_column=True):
    # data = dataframe as pandas.DataFrame
    # text_column = column in data containing the concatenated string for embedding
    # model = model to use (must be loaded before)
    # add_column = whether to add a 'truncated' column to data

    tokenizer = model.tokenizer

    # Count tokens for each text (title + abstract combined); treat NaN as empty string
    token_count = data[text_column].fillna('').apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=True))
    )

    # Summary statistics
    print(f"\nTexts exceeding 512 tokens: {(token_count > 512).sum()} "
        f"({(token_count > 512).mean():.1%})")
    
    # add column
    if add_column == True:
        data['truncated'] = token_count > 512

## get_embeddings
## converts the provided text to embeddings using the specified model
## runs on GPU if available, otherwise on CPU
def get_embeddings(data, text_column, file_path, model=None, batch_size=32, checkpoint=True):
    # data = dataframe as pandas.DataFrame
    # text_column = column in data containing the concatenated string for embedding
    # model = model to use (must be loaded before)
    # batch_size = size of batches going into model
    # checkpoint = whether to save checkpoint during embedding to enable continuous embedding in case of crash
    # file_path = path to save the embedding file

    if model is None:
        raise ValueError("model must be provided")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    texts = data[text_column]
    total_batches = (len(texts) + batch_size - 1) // batch_size

    if checkpoint == True:
        if os.path.exists(file_path):
            embeddings = list(np.load(file_path, allow_pickle=True))
            if len(embeddings) > len(texts):
                print(f"Checkpoint has {len(embeddings)} embeddings but only {len(texts)} texts — discarding stale checkpoint.")
                embeddings = []
            start_idx = len(embeddings)
            start_batch = start_idx // batch_size
            if start_idx > 0:
                print(f"Resuming from {start_idx}")
        else:
            embeddings = []
            start_idx = 0
            start_batch = 0

        texts_idx = texts.tolist()[start_idx:]

        for i in tqdm(range(0, len(texts_idx), batch_size),
                      desc='Embedding', total=total_batches,
                      initial=start_batch, unit='batch'):
            batch = texts_idx[i:i+batch_size]
            embeddings.extend(model.encode(batch, batch_size=batch_size))
            np.save(file_path, embeddings)

        embeddings = np.array(embeddings)

    else:
        embeddings = model.encode(texts.tolist(), batch_size=batch_size)
        np.save(file_path, embeddings)

    return embeddings

## train_scope
## trains the cope classifier
## two outputs --> call like this: classifier, threshold = train_scope(...)
def train_scope(embeddings, labels, model_path, model=SVC,
                test=False, test_size=0.2, stratify_by=None, max_fn=0.01, **model_kwargs):
    # embeddings = output from sentence transfomer, emebedded text
    # labels = labels that will be predicted, e.g. data['scope']. Make sure the indeces of embeddings and labels are the same and labels are binary (0 and 1).
    # model = which classifier to use, default = SVC, other options include: Logistic Regression, MLPClassifier (not built-in!)
    # model_kwargs = arguments for the specific model. For SVC: C, class_weight, kernel, gamma
    # test = whether to split the provided data into training and test data. If True, test probabilities will be used to determine the threshold, if False, cross-validation will be used
    # test_size = if data is split in training and test data, what size should the test set be
    # stratify_by = which variable should be used to stratify the data in a balanced way, if None, data will be split randomly
    # max_fm = maximum % of false negative cases, will be used to set the threshold
    # model_path = path to save the model

    if model == SVC:
        defaults = {'C': 100, 'class_weight': 'balanced', 'max_iter': 1000, 'kernel': 'rbf', 'gamma': 0.01}
        defaults.update(model_kwargs)
        model_kwargs = defaults

    # define classifier and wrap with calibration
    base_classifier = model(**model_kwargs, random_state=42)
    classifier = CalibratedClassifierCV(base_classifier, cv=5, method='isotonic')

    # split data into train and test if test = True
    # train classifier with training data and validate with test data
    if test == True:
        if stratify_by is None:
            warnings.warn('Data is split randomly because no stratification variable was provided.')

            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels,
                test_size=test_size,
                random_state=42
            )

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels,
                test_size=test_size,
                random_state=42,
                stratify=stratify_by
            )

        # train model
        classifier.fit(X_train, y_train)

        # validate on test data
        proba = classifier.predict_proba(X_test)[:, 1]

        # determine threshold where FN < 1%
        thresholds = np.linspace(1, 0, 1001)
        n = len(y_test)

        for T in thresholds:
            preds = (proba >= T).astype(int)
            fn_rate = ((np.array(y_test) == 1) & (preds == 0)).sum() / n
            if fn_rate < max_fn:
                threshold = T
                print(f"Determined threshold: {T:.3f}, FN rate: {fn_rate:.3%}")
                break

    else:
        # train model
        classifier.fit(embeddings, labels)

        # get probabilities with cross-validation
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba = cross_val_predict(classifier, embeddings, labels, 
                              cv=cv, method='predict_proba')[:, 1]
        
    
        # determine threshold where FN < 1%
        thresholds = np.linspace(1, 0, 1001)
        n = len(labels)

        for T in thresholds:
            preds = (proba >= T).astype(int)
            fn_rate = ((np.array(labels) == 1) & (preds == 0)).sum() / n
            if fn_rate < max_fn:
                threshold = T
                print(f"Determined threshold: {T:.3f}, FN rate: {fn_rate:.3%}")
                break

    # save model
    joblib.dump(classifier, model_path)

    return classifier, threshold

## train_pillar
## classifier to predict the AP pillar
def train_pillar(embeddings, labels, model_path, model=SVC, **model_kwargs):
    # embeddings = output from sentence transfomer, emebedded text
    # labels = labels that will be predicted, e.g. data['pillar']. Make sure the indeces of embeddings and labels are the same! 
    # model = which classifier to use, default = SVC, other options include: LogisticRegression, MLPClassifier (not built-in!)
    # model_kwargs = arguments for the specific model. For SVC: C, class_weight, kernel, gamma
    # model_path = path to save the model

    if model == SVC:
        defaults = {'C': 1000, 'class_weight': 'balanced', 'max_iter': 1000, 'kernel': 'rbf', 'gamma': 0.01}
        defaults.update(model_kwargs)
        model_kwargs = defaults

    # define classifier and wrap with calibration
    base_classifier = model(**model_kwargs, random_state=42)
    classifier = CalibratedClassifierCV(base_classifier, cv=5, method='isotonic')

    # train model
    classifier.fit(embeddings, labels)

    # save model
    joblib.dump(classifier, model_path)

    return classifier

## scope_classification
## use scope classifier to predict scope
## two outputs --> call like this: proba, preds = scope_classification(...)
def scope_classification(embeddings, model_path, threshold=0.1):
    # embeddings = output from sentence transfomer, emebedded text
    # model_path = path to saved scope classifier
    # threshold = treshold for in scope prediction as determined during training

    # load model
    classifier = joblib.load(model_path)

    # predict probabilities and scope using the threshold
    proba = classifier.predict_proba(embeddings)[:, 1]
    preds = (proba >= threshold).astype(int)

    # output: probabilities and predictions as arrays
    # can be added to data with: data['proba_scope'] = proba
    return proba, preds

## pillar_classification
## use pillar classifier to predict pillar
## two outputs --> call like this: proba, preds = pillar_classification(...)
def pillar_classification(embeddings, model_path):
    # load model
    classifier = joblib.load(model_path)

    # predict probabilities and pillar
    proba = classifier.predict_proba(embeddings).max(axis=1)
    preds = classifier.predict(embeddings)

    # output: probabilities and predictions as arrays
    # can be added to data with: data['proba_pillar'] = proba
    return proba, preds

## combine_classifications
## combine scope and pillar predictions to only exclude entries that were predicted out of scope and not assigned to an AP pillar
## one output --> call like this: preds_combined = combine_classification(...)
def combine_classifications(preds_scope, preds_pillar):
    # preds_scope = scope predictions from scope_classification
    # preds_pillar = pillar predictions from pillar_classification

    # combine the two predictions
    preds_combined = ((preds_pillar != 'NA') | (preds_scope == 1)).astype(int)

    # return combined predictions
    # can be added to data with: data['pred_final'] = preds_combined
    return preds_combined











   
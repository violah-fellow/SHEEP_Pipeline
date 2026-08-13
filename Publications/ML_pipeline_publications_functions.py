## This file contains all functions necessary to run the ML pipeline for publication classification
##
## Function library, not a script - nothing runs when this file is imported. It is loaded by
## S0_ML_training.py (training) and S2_ML_classification.py (applying the models), which both import
## it as `mlf` and only call the functions below. The idea is that all the ML mechanics live here and
## the pipeline scripts stay short and readable.
##
## Order in which the functions are used:
##   load_specter()          -> load embedding model (once per script run)
##   check_token_size()      -> how many texts are too long for the model? (informational)
##   get_embeddings()        -> text -> vectors (the slow step, with checkpointing)
##   train_scope()           -> train binary in/out classifier + determine threshold   (S0 only)
##   train_pillar()          -> train multiclass PB/F/CM/CC/NA classifier              (S0 only)
##   scope_classification()  -> apply saved scope model to new embeddings              (S2 only)
##   pillar_classification() -> apply saved pillar model to new embeddings             (S2 only)
##   combine_classifications() -> combine both predictions into the final in/out decision
##
## The Patents pipeline has a parallel file (ML_pipeline_patents_functions.py) with the same function
## names but a different model (PatentSBERTa instead of SPECTER2) and different classifiers (SVC
## instead of LogisticRegression). If you change the logic here, check whether the same applies there.

## Necessary packages
import pandas as pd
import numpy as np
import os
import warnings

import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
import joblib
from tqdm import tqdm

## load_specter
## loads the embedding model SPETER2 with the classification adapter
## this model needs a different loading process than simple models from SentenceTransformers
## two outputs --> call like this: model, tokenizer = load_specter(...)
## on the first call the model is downloaded from Hugging Face (a few hundred MB) and cached locally,
## so an internet connection is only needed once
def load_specter():
    # load model
    # base model and tokenizer must come from the same checkpoint - the tokenizer decides how text is
    # split into tokens, and a mismatch would silently produce worse embeddings
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

    # Load classification adapter
    # SPECTER2 offers several adapters for different purposes; the 'classification' adapter is the
    # right one for embeddings that feed into a classifier (others are meant for search or citations).
    # model.eval() below switches off training behaviour (dropout) - required for inference.
    model.load_adapter("allenai/specter2_classification", 
                    source="hf", 
                    load_as="specter2_cls", 
                    set_active=True)
    model.eval()

    return model, tokenizer

## check_token_size
## counts number of tokens per entry and saves a 'truncated' = True/False column to the provided dataframe if add_column = True
## purely informational - nothing is shortened or dropped here. Texts longer than 512 tokens are cut
## off by the model itself when embedding, i.e. the end of a long abstract is not taken into account.
## The printed percentage is a useful sanity check: if it is very high, the classification quality
## for those records depends only on their first ~2 paragraphs.
def check_token_size(data, text_column, tokenizer=None, add_column=True):
    # data = dataframe as pandas.DataFrame
    # text_column = column in data containing the concatenated string for embedding
    # model = model to use (must be loaded before)
    # add_column = whether to add a 'truncated' column to data

    if tokenizer is None:
            raise ValueError("tokenizer must be provided")

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
## the slowest part of the whole pipeline. With checkpoint=True the file at file_path is rewritten
## after every batch, so an interrupted run continues instead of starting over.
## IMPORTANT for reuse: the checkpoint only records HOW MANY texts have been embedded, not WHICH ones.
## It therefore only fits the exact same input in the exact same order - give every run its own
## file_path (the pipeline does), otherwise embeddings from a previous run may be reused. The
## start_idx >= len(texts) case below catches the most obvious version of that mistake.
def get_embeddings(data, text_column, file_path, model=None, tokenizer=None, batch_size=32, checkpoint=True):
    # data = dataframe as pandas.DataFrame
    # text_column = column in data containing the concatenated string for embedding
    # model = model to use (must be loaded before)
    # batch_size = size of batches going into model
    # checkpoint = whether to save checkpoint during embedding to enable continuous embedding in case of crash
    # file_path = path to save the embedding file

    if model is None:
        raise ValueError("model must be provided")

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    # 'cuda' = NVIDIA GPU. Without a GPU everything still works, just considerably slower - which is
    # exactly why the checkpointing below matters on a laptop.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    texts = data[text_column]
    total_batches = (len(texts) + batch_size - 1) // batch_size

    # inner helper: embeds a list of texts in batches and returns one matrix
    def get_embeddings_helper(texts, model, batch_size):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # truncation=True + max_length=512 is where over-long texts are actually cut off
            # (see check_token_size); padding=True pads shorter texts so the batch has one shape
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # no_grad: no gradients needed for inference, saves memory and time
            with torch.no_grad():
                outputs = model(**inputs)
            # [:, 0, :] takes the vector of the first token (CLS) as the embedding of the whole text -
            # the standard approach for SPECTER2 - and moves it back to the CPU as a numpy array
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    if checkpoint == True:
        if os.path.exists(file_path):
            embeddings = list(np.load(file_path, allow_pickle=True))
            start_idx = len(embeddings)
            if start_idx >= len(texts):
                # checkpoint is stale (e.g. from a previous run whose rows were already saved
                # to the embeddings table and filtered out of the current batch)
                print(f"Checkpoint has {start_idx} entries but only {len(texts)} texts — starting fresh.")
                embeddings = []
                start_idx = 0
            start_batch = start_idx // batch_size
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
            embeddings.extend(get_embeddings_helper(batch, model=model, batch_size=batch_size))
            # saved after every batch: costs a little time, but at most one batch is lost on a crash
            np.save(file_path, embeddings)

        embeddings = np.array(embeddings)

    else:
        embeddings = get_embeddings_helper(texts.tolist(), model=model, batch_size=batch_size)
        np.save(file_path, embeddings)

    return embeddings

## train_scope
## trains the cope classifier   [scope classifier]
## two outputs --> call like this: classifier, threshold = train_scope(...)
##
## Two things happen here, and the second one is the important one:
##  1. a classifier is trained on the labelled embeddings
##  2. a decision threshold is determined - the probability above which a publication counts as in
##     scope. Not 0.5: the threshold is pushed down until fewer than max_fn of the labelled records
##     would be missed (false negatives). The rationale is that a false positive is cheap (the LLM
##     step discards it) while a false negative is lost from the dataset for good.
## The threshold is NOT stored in the model file - the caller (S0_ML_training.py) writes it to
## Models/LR_scope_threshold.txt, and S2_ML_classification.py reads it back from there. Model file and
## threshold file therefore always belong together.
##
## Caveats worth knowing when interpreting the printed FN rate:
##  - the rate is computed over ALL records, not over the in-scope ones only, so it is not the usual
##    "recall loss" figure; with few in-scope examples it looks better than it is
##  - if no threshold satisfies max_fn, `threshold` is never assigned and the function raises an
##    UnboundLocalError instead of reporting the problem
def train_scope(embeddings, labels, model_path, model=LogisticRegression,
                test=False, test_size=0.2, stratify_by=None, max_fn=0.01, **model_kwargs):
    # embeddings = output from sentence transfomer, emebedded text
    # labels = labels that will be predicted, e.g. data['scope']. Make sure the indeces of embeddings and labels are the same and labels are binary (0 and 1).
    # model = which classifier to use, default = LogisticRegression, other options include: MLPClassifier, SVC
    # model_kwargs = arguments for the specific model. For LogisticRegression: C, class_weight
    # test = whether to split the provided data into training and test data. If True, test probabilities will be used to determine the threshold, if False, cross-validation will be used
    # test_size = if data is split in training and test data, what size should the test set be
    # stratify_by = which variable should be used to stratify the data in a balanced way, if None, data will be split randomly
    # max_fn = maximum % of false negative cases, will be used to set the threshold
    # model_path = path to save the model

    # sensible defaults for LogisticRegression, only applied if the caller passes nothing else;
    # max_iter=1000 avoids the convergence warnings that occur with the default of 100
    if model == LogisticRegression:
        defaults = {'C': 0.1, 'class_weight':'balanced', 'max_iter': 1000}
        defaults.update(model_kwargs)
        model_kwargs = defaults

    # define classifier
    # random_state=42 keeps the result reproducible: the same training data yields the same model
    classifier = model(**model_kwargs, random_state=42)

    # split data into train and test if test = True
    # train classifier with training data and validate with test data
    # two ways of arriving at the threshold:
    #   test=True  -> a single train/test split, threshold determined on the held-out test set
    #   test=False -> 5-fold cross-validation over all data (the default, and what S0 uses): every
    #                 record gets a probability from a model that did not see it, so the threshold
    #                 rests on the whole labelled dataset instead of one split
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
        # the model that is saved is trained on ALL data; the cross-validation below is only used to
        # estimate the threshold, not to produce the final model
        classifier.fit(embeddings, labels)

        # get probabilities with cross-validation
        # StratifiedKFold keeps the in/out ratio the same in every fold - important with an
        # imbalanced dataset. cross_val_predict gives each record a probability from the fold in
        # which it was NOT part of the training data.
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba = cross_val_predict(classifier, embeddings, labels,
                              cv=cv, method='predict_proba')[:, 1]


        # determine threshold where FN < 1%
        # thresholds are walked from 1 down to 0 in steps of 0.001, and the FIRST value at which the
        # false-negative rate drops below max_fn wins - i.e. the highest (strictest) threshold that
        # still meets the requirement
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
    # overwrites the existing file without warning - keep a copy if you want to compare models
    joblib.dump(classifier, model_path)

    return classifier, threshold

## train_pillar
## classifier to predict the AP pillar
## AP = alternative protein; classes are PB (plant-based), F (fermentation), CM (cultivated meat),
## CC (cross-cutting) and NA (no pillar). No threshold here - the predicted class is simply the one
## with the highest probability, which is why this function has only one return value.
def train_pillar(embeddings, labels, model_path, model=LogisticRegression, **model_kwargs):
    # embeddings = output from sentence transfomer, emebedded text
    # labels = labels that will be predicted, e.g. data['pillar']. Make sure the indeces of embeddings and labels are the same! 
    # model = which classifier to use, default = LogisticRegression, other options include: MLPClassifier, SVC
    # model_kwargs = arguments for the specific model. For LogisticRegression: C, class_weight
    # model_path = path to save the model

    if model == LogisticRegression:
        defaults = {'C': 0.1, 'class_weight':'balanced', 'max_iter': 1000}
        defaults.update(model_kwargs)
        model_kwargs = defaults

    # define classifier
    classifier = model(**model_kwargs, random_state=42)

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
    # [:, 1] = probability of class 1 (in scope). The default threshold=0.1 in the signature is only
    # a fallback; S2_ML_classification.py always passes the value from the training run.
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
    # proba is only the probability of the winning class (max), i.e. a measure of how certain the
    # model is about that pillar - not the probability of a specific pillar
    proba = classifier.predict_proba(embeddings).max(axis=1)
    preds = classifier.predict(embeddings)

    # output: probabilities and predictions as arrays
    # can be added to data with: data['proba_pillar'] = proba
    return proba, preds

## combine_classifications
## combine scope and pillar predictions to only exclude entries that were predicted out of scope and not assigned to an AP pillar
## one output --> call like this: preds_combined = combine_classification(...)
## the deliberately generous OR rule of the pre-filter: a publication is only dropped if BOTH models
## reject it. Whatever survives here goes to the LLM (S3), which is far better at judging scope - so
## this function should err on the side of letting too much through rather than too little.
def combine_classifications(preds_scope, preds_pillar):
    # preds_scope = scope predictions from scope_classification
    # preds_pillar = pillar predictions from pillar_classification

    # combine the two predictions
    preds_combined = ((preds_pillar != 'NA') | (preds_scope == 1)).astype(int)

    # return combined predictions
    # can be added to data with: data['pred_final'] = preds_combined
    return preds_combined











   
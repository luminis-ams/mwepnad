# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # M-WePNaD task
# 
# In this notebook we'll take a first look at the data for the M-WePNaD task; we'll sample ten random training queries and develop a simple preprocessing pipeline.

# <codecell>

from itertools import islice
import nltk
import numpy as np
import os
import pandas as pd
import re
import sys
import xml.etree.ElementTree as ET

# <codecell>

# check how np is configured (do we have fast linear algebra?)
# NB: it's not that easy to find just how to interpret this output.
np.show_config()

# <codecell>

training_data_dir = os.path.join('..', '..', 'data', 'training_data', 'MWePNaDTraining')
exp_dir = os.path.join('..','..','exp')
src_dir = os.path.join('..')

# <codecell>

def log(msg):
    print(msg)

# <codecell>

prng = np.random.RandomState(seed=42)
ten_queries = prng.choice(os.listdir(training_data_dir), 10)

# <markdowncell>

# # A first peek into some of the data

# <codecell>

walk_first_query = os.walk(os.path.join(training_data_dir, ten_queries[0]))

# <codecell>

[i for i in islice(walk_first_query, 1, 3)]

# <codecell>

first_query_dir = os.path.join(training_data_dir, ten_queries[0])
first_metadata_path = os.path.join(first_query_dir, os.listdir(first_query_dir)[0], 'metadata.xml')

# <codecell>

print(open(first_metadata_path).read())

# <codecell>

tree = ET.parse(first_metadata_path)
root = tree.getroot()

# <codecell>

[i for i in root]

# <codecell>

root.tag

# <codecell>

language_e = root.find('{http://www.example.org/metadata-corpus}language')

# <codecell>

root.find('{http://www.example.org/metadata-corpus}url').text

# <codecell>

language_e.text

# <markdowncell>

# # Some useful ad-hoc functions and classes

# <codecell>

# NB: fix annoying spelling mistake in gold standard and dirname in training corpus
def correct_paul_erhlich(query):
    return 'paul ehrlich' if query == 'paul erhlich' else query

# <codecell>

def load_gold_standard():
    try:
        f = open(os.path.join(training_data_dir, '..', 'GoldStandardTraining.txt'))
    except IOError:
        pass
    else:
        with f:
            return frozenset([(correct_paul_erhlich(q), d, c) for q, d, c in [
                    tuple(l.rstrip('\n').split('\t')) for l in f]])

# <codecell>

gold_standard = load_gold_standard()

# <codecell>

"""
Parse document dir

Load data and metadata in memory

Just meant as a minimal parse of the files on disk, not meant to hold many features.
"""
class Document:
    XMLNS = '{http://www.example.org/metadata-corpus}'
    def __init__(self, dir_path, file_paths):
        self.path = dir_path
        self.file_paths = file_paths
        self._parse_metadata()
        self._get_text()
        
    def _parse_metadata(self):
        tree = ET.parse(os.path.join(self.path, 'metadata.xml'))
        root = tree.getroot()
        self.url_ = root.find('{}url'.format(Document.XMLNS)).text
        self.languages = frozenset(root.find('{}language'.format(Document.XMLNS))
                                   .text.strip().split(','))
         
    def _get_text(self):
        ids = [re.sub(r'\.txt$', '', p) for p in self.file_paths if p.endswith('.txt')]
        assert len(ids) == 1
        self.id = ids[0]
        with open(os.path.join(self.path,'{}.txt'.format(self.id))) as f:
            self.text = f.read()

# <codecell>

"""
Parse query dir

Load data and metadata in memory
"""
class Query:
    def __init__(self, path):
        self.path = path
        self.query = correct_paul_erhlich(os.path.basename(path).replace('_', ' '))
        try:
            self._load()
        except Exception as e:
            log('Exception: {}'.format(e))
            raise ValueError('Could not parse everything in query dir')
    
    def _load(self):
        self.docs = []
        walker = os.walk(self.path)
        for path, _, files in walker:
            if 'metadata.xml' in files:
                self.docs.append(Document(path, files))

# <markdowncell>

# # A closer look at our first query

# <codecell>

first_query = Query(first_query_dir)

# <codecell>

first_query.path

# <codecell>

first_query.query

# <codecell>

s_docs_first_query = pd.Series([doc.languages for doc in first_query.docs])

# <codecell>

s_docs_first_query.unique()

# <codecell>

print(re.sub(r'\s+', ' ', first_query.docs[11].text.lower())[:250])

# <codecell>

s_first_query_urls = pd.Series([doc.url_ for doc in first_query.docs])

# <codecell>

s_first_query_urls.sample(5, random_state=prng)

# <markdowncell>

# # Features for classification into 'NR' and 'relevant'

# <codecell>

"""
Compute a feature

Accepts a Document and a Query object. 
Normally you would use this function with a Document that is in query.docs.

Returns a numeric value
"""
def compute_n_exact_name_matches(query, doc):
    first_name, last_name = query.query.split(' ')
    re_ = re.compile(r'' + re.escape(first_name) + r'\s+' + re.escape(last_name))
    return len(re.findall(re_, doc.text.lower()))
    # TODO: use URL, too

# <codecell>

"""
Compute a feature

Accepts a Document and a Query object. 
Normally you would use this function with a Document that is in query.docs.

Returns a numeric value
"""
def compute_n_name_matches_with_optional_word_or_initial_in_between(query, doc):
    first_name, last_name = query.query.split(' ')
    re_ = re.compile(r'' + re.escape(first_name) + r'\s+' + r'[\w]*\.?\s*' + re.escape(last_name))
    return len(re.findall(re_, doc.text.lower()))
    # TODO: use URL, too
    # TODO: compute other features, e.g., 1 / (1 + number_of_chars_between_first_and_last_name)
    # TODO: possibly preprocess and tokenise text / URL prior to computing features

# <codecell>

compute_n_exact_name_matches(first_query, first_query.docs[11])

# <codecell>

compute_n_name_matches_with_optional_word_or_initial_in_between(first_query, first_query.docs[11])

# <markdowncell>

# Now, let's get an idea for how well these two features correlate with NR in the ground truth for our ten randomly sampled development queries. Or rather, first, just for our 'first_query'.

# <codecell>

s_feature = pd.Series({
        doc.id : compute_n_name_matches_with_optional_word_or_initial_in_between(
                first_query, doc) for doc in first_query.docs})

# <codecell>

s_gold_standard = pd.Series({a[1] : 1 if a[2] == 'NR' else 0 for a in gold_standard if a[0] == first_query.query})

# <codecell>

df = pd.concat([s_gold_standard, s_feature], axis=1, join_axes=[s_gold_standard.index], keys=['NR', 'x'])

# <codecell>

df.shape[0]

# <codecell>

df.groupby('NR').agg('mean')

# <markdowncell>

# Interestingly, above, the number of name mention matches is higher for pages annotated as NR. What reasons would annotators use to make this decision? Let's look into some of these pages, where we do have name matches, but the pages are annotated as 'NR' (not relevant).

# <codecell>

df.loc[df['NR'] == 1, :]

# <codecell>

print(re.sub(r'\n\n+', '\n\n', [doc for doc in first_query.docs if doc.id == '029'][0].text)[:500])

# <markdowncell>

# The above page seems to be a LinkedIn listing of public profiles that came up for the search 'Paul Ehrlich'.

# <codecell>

print(re.sub(r'\n\n+', '\n\n', [doc for doc in first_query.docs if doc.id == '024'][0].text)[:500])

# <markdowncell>

# The above page seems to refer mainly to an event / network that is named after one Paul Ehrlich.

# <markdowncell>

# # Some ideas for this task
# 
# As a first step, it would seem a nice experiment to see if we can correctly classify pages as 'NR' (binary classification). Because of the way this task will be evaluated, it is a good idea to put all of these pages together in a single cluster. We could engineer some features (see ideas in the TODO's above) and then train a simple classifier.
# 
# As a second step, we could try to find some generic way to tokenise text, regardless of language. So, no stemming or anything like that yet, just tokenisation. Then, an often used representation of documents could be a TF-IDF vector.
# 
# As a third step, we could calculate cosine distances between documents.
# 
# As a fourth step, we could use hierarchical agglomerative clustering, it has performed well in previous editions of WePS campaigns.
# 
# If we want, we can use the simple idea from Berendsen et al, ECIR 2012, where social media profiles were not included in the clustering. Instead, adding them as singleton clusters after clustering the rest of the pages boosted the score considerably.
# 
# If we want to improve further, we could add custom tokenisation for each language.
# 
# We can also investigate how common it is that an individual will be referred to from pages with different languages. If it happens a lot, another way to improve scores would be to use some kind of translation machinery.

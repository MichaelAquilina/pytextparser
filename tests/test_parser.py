# -*- encoding: utf8 -*-
from __future__ import unicode_literals, print_function, division

import unittest

import pytextparser


def test_loads_stopwords():
    assert isinstance(pytextparser.load_aggressive_stopwords(), list)


class TfidfTestCase(unittest.TestCase):

    def test_zero_term_frequency(self):
        assert pytextparser.tfidf(tf=0, df=10, corpus_size=1) == 0

    def test_zero_document_frequency(self):
        assert pytextparser.tfidf(tf=10, df=0, corpus_size=1) == 0


class IsNumericTestCase(unittest.TestCase):

    def test_integer(self):
        assert pytextparser.isnumeric('23')
        assert pytextparser.isnumeric('8431')

    def test_float(self):
        assert pytextparser.isnumeric('23.480')
        assert pytextparser.isnumeric('9.6502')

    def test_scientific_notation(self):
        assert pytextparser.isnumeric('1e-10')
        assert pytextparser.isnumeric('2e+54')

    def test_no_numeric(self):
        assert not pytextparser.isnumeric('foo')
        assert not pytextparser.isnumeric('10 foo')


class GetNGramsTestCase(unittest.TestCase):

    def test_bigram_token_list(self):
        assert list(pytextparser.get_ngrams(
            token_list=['one', 'two', 'three', 'four'],
        )) == [['one', 'two'], ['two', 'three'], ['three', 'four']]

    def test_trigram_token_list(self):
        assert list(pytextparser.get_ngrams(
            token_list=['one', 'two', 'three', 'four'],
            n=3,
        )) == [
            ['one', 'two', 'three'],
            ['two', 'three', 'four'],
        ]


class WordTokenizeTestCase(unittest.TestCase):

    def test_sentence(self):
        assert list(pytextparser.word_tokenize(
            text='hello cruel world',
        )) == [('hello', ), ('cruel', ), ('world', )]

    def test_splits_punctuation(self):
        assert list(pytextparser.word_tokenize(
            text='first. second',
        )) == [('first', ), ('second', )]

    def test_ignores_stopwords(self):
        assert list(pytextparser.word_tokenize(
            text='The first rule of python is',
            stopwords=set(['the', 'of', 'is']),
            min_length=1,
        )) == [('first', ), ('rule', ), ('python', )]

    def test_min_length(self):
        assert list(pytextparser.word_tokenize(
            text='one for the money two for the go',
            min_length=4,
        )) == [('money', )]

    def test_ignores_numeric(self):
        assert list(pytextparser.word_tokenize(
            text='one two 3 four',
        )) == [('one', ), ('two', ), ('four', )]

    def test_ngrams(self):
        assert list(pytextparser.word_tokenize(
            text='foo bar bomb blar',
            ngrams=2,
        )) == [('foo', 'bar'), ('bar', 'bomb'), ('bomb', 'blar')]


class IsUrlTestCase(unittest.TestCase):

    def test_http_url(self):
        assert pytextparser.is_url('http://www.google.com')

    def test_https_url(self):
        assert pytextparser.is_url('https://www.google.com')

    def test_url_with_path(self):
        assert pytextparser.is_url('https://www.facebook.com/some/path/here')

    def test_url_with_query_string(self):
        assert pytextparser.is_url('https://www.yplanapp.com?foo=1&bar=2')

    def test_not_a_url(self):
        assert not pytextparser.is_url('foo')
        assert not pytextparser.is_url('bar')
        assert not pytextparser.is_url('waterboat')

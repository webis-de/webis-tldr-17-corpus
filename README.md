# webis-tldr-17-corpus
This repository contains code for constructing TLDR corpus from Reddit Corpus as described in [TL;DR: Mining Reddit to Learn Automatic Summarization, EMNLP 2017 - New Frontiers in Summarization workshop](https://aclanthology.org/W17-4508)

## About this code

This code is intended to be run using Spark framework for working with large Reddit dumps directly. It consists of two scripts:

`make_reddit.py` - Reads the raw dumps and creates content-summary pairs in the form of Spark dataframe.

`clean_reddit.py` - Reads the result of the previous script and applies some normalization for improving precision of the final corpus.

The resources folder contains an exhaustive list of Reddit bots which we use to filter automatic postings.

## Usage

`spark-submit --master yarn make_tldr.py --input_comments input-comments-path --input_submissions input-submissions-path --output_comments tldr-comments-raw --output_submissions tldr-submissions-raw`

__We use Mistune library to remove markdown, which should be passed to Spark using `--py-files`__

`spark-submit --master yarn --py-files /usr/local/lib/python3.5/dist-packages/mistune.py clean_tldr.py --input_comments tldr-comments-raw --input_submissions tldr-submissions-raw --output_comments tldr-comments-cleaned --output_submissions tldr-submissions-cleaned`

## Released corpus

The current version of the corpus can be found here on [Zenodo](https://zenodo.org/record/1043504#.Wzt7PbhXryo)

# Machine Learning Engineer Nanodegree Capstone Project

## Overview
This project attempts to extract useful signal from the text of public company quarterly earnings transcripts. Although the proposed approach does not prove to be successful, this notebook contains interesting analysis and a few potential developments that could improve this model to a useful (and profitable) level. 

Privacy notice: Please do not distribute this notebook.

## Data
##### Stock price
Price data is downloaded from Google Finance using the Pandas DataReader. See notebook to run the command to download all necessary data.

##### Quarterly earnings call transcripts
Transcripts are scraped from [Seeking Alpha](https://seekingalpha.com/) using the Python library [Scrapy](https://docs.scrapy.org/en/latest/).

To fetch a company transcript, complete the following steps.

```
cd data/
scrapy crawl transcripts -a symbol=$SYM
```

This will download all of the posted earnings call transcripts for company `SYM` and store it as a JSON lines file in `data/company_transcripts/SYM.json`.

## Environment
Create a new Anaconda environment using the command below to ensure your workspace has all necessary dependencies.

```
conda env create -f requirements/environment.yml
```

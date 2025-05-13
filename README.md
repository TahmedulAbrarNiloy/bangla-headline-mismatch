# bangla-headline-mismatch
This project aims to detect whether a Bangla news article’s headline accurately reflects its body content or is misleading. It is part of my MIT Final and research project under the supervision of Dr. Ahmedul Kabir.

## Objectives
- Build a labelled dataset of Bangla news headline-body pairs
- Train machine learning models to classify matched vs. mismatched articles
- Develop a web application to test headline-body pairs
- Integrate with a browser extension for automatic news scanning

## Project Timeline
Week 1 (Current):
- Set up repository
- Begin collecting dataset (30 headline-body samples)
- Train a baseline ML model

## Folder Structure
bangla-headline-mismatch/
│
├── README.md
├── log.md # Daily progress log
├── data/
│ └── dataset.csv # Headline-body-label dataset
└── scripts/
├── preprocessing.py # Data cleaning, TF-IDF
└── baseline_model.py # First model training and testing

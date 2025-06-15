# Sentiment Analysis & Clustering

This project analyses company reviews from ITviec using two main tasks:

1. **Sentiment Analysis** – counts positive and negative words/emoji to label each review.
2. **Clustering** – groups companies based on aggregated review text and LDA topic vectors.

The repository contains a Streamlit dashboard (`streamlit_app.py`) and a notebook with the full workflow.

The data files located in the `data/` folder originate from the original project repository.

To install the required packages, run:

```bash
pip install -r requirements.txt
```

Launch the Streamlit dashboard with:

```bash
streamlit run streamlit_app.py
```

Outputs referenced by the dashboard (`outputs/` folder) are not included here. Re-run the notebook or your own pipeline to generate them.

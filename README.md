# US Election Sentiment Analysis

This project analyzes Twitter sentiment for the 2020 US Presidential Election using Python. It compares public sentiment towards Donald Trump and Joe Biden by analyzing tweets from their official Twitter handles.

## Project Overview
- **Goal:** Predict the preferred candidate based on sentiment analysis of tweets.
- **Method:** Uses TextBlob for sentiment polarity, cleans and balances the data, and visualizes the results with Plotly.

## Dataset
- `Trumpall2.csv`: Tweets mentioning Donald Trump
- `Bidenall2.csv`: Tweets mentioning Joe Biden
- Place both CSV files in the project directory before running the script.

## Setup Instructions
1. **Clone the repository or download the script.**
2. **Install dependencies:**
   ```sh
   pip install pandas numpy seaborn matplotlib textblob wordcloud plotly
   ```
   Or, if using a virtual environment:
   ```sh
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   pip install pandas numpy seaborn matplotlib textblob wordcloud plotly
   ```
3. **Add the datasets** (`Trumpall2.csv` and `Bidenall2.csv`) to the project folder.

## Usage
Run the script with:
```sh
python us_election_sentiment.py
```
- The script will print sample data, sentiment counts, and analysis steps to the terminal.
- An interactive bar chart will open in your browser and be saved as `us_election_sentiment_analysis.html`.

## Result Interpretation
- The bar chart compares the number of positive and negative tweets for each candidate.
- The candidate with more positive and fewer negative tweets is considered more favored by public sentiment in this analysis.



## Notes
- This analysis is for educational purposes and is based solely on the provided Twitter data.
- Sentiment analysis is a simple approach and may not capture all nuances of public opinion.

## License
MIT License

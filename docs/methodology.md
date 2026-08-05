# Methodology — Brand Sentiment Analysis

## Sentiment Scoring Approach

This project uses a **hybrid sentiment analysis** approach combining two NLP engines:

### TextBlob
- Lexicon-based polarity scoring
- Range: -1.0 (most negative) to +1.0 (most positive)
- Best for: General English text, formal language
- Limitation: Struggles with slang, emoji, and informal social media text

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Rule-based sentiment analysis optimized for social media
- Range: -1.0 to +1.0 (compound score)
- Best for: Twitter, Reddit, Instagram comments
- Strengths: Handles capitalization intensity, punctuation, emoji, and slang

### Hybrid Score
The final sentiment score averages both engines:

```
hybrid_score = (textblob_polarity + vader_compound) / 2
```

Classification thresholds:
- Positive: score > +0.05
- Neutral: -0.05 ≤ score ≤ +0.05
- Negative: score < -0.05

## Why Hybrid?

Single-engine approaches have known weaknesses. TextBlob underestimates sentiment in informal text. VADER can overreact to single strong words. Averaging both produces more stable, defensible results — important when presenting findings to stakeholders.

## Data Pipeline

1. **Collection** — Social media posts are gathered via API or export
2. **Preprocessing** — URLs, handles, and stop words removed
3. **Scoring** — Both engines score each post independently
4. **Classification** — Hybrid scores are binned into positive/neutral/negative
5. **Aggregation** — Scores are grouped weekly and by platform
6. **Visualization** — Charts are generated for trend analysis

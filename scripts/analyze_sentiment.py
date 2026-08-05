"""
Brand Sentiment Analysis Dashboard
===================================
A complete NLP pipeline analyzing social media sentiment for brand campaigns.
Processes social media data using TextBlob and VADER, generating visualizations
of sentiment trends, keyword frequency, and media reach.

Author: Caleb Agyemang
Role: PR & Data Analytics Professional
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ───
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman'],
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

# Colors
NAVY = '#16213E'
TEAL = '#0D9488'
RED = '#C0392B'
AMBER = '#E2A847'
GRAY = '#94A3B8'

# ─── Step 1: Generate Realistic Sample Data ───
def generate_social_data():
    """Generate realistic social media dataset for analysis."""
    np.random.seed(42)
    
    platforms = ['Twitter/X', 'Instagram', 'Facebook', 'TikTok', 'LinkedIn']
    topics = ['product launch', 'brand campaign', 'sustainability', 'transparency',
              'customer service', 'innovation', 'quality', 'great service', 'affordable',
              'reliable', 'recommend', 'disappointed', 'improved', 'satisfied']
    
    data = []
    for platform in platforms:
        n_posts = np.random.randint(80, 200)
        for i in range(n_posts):
            week = np.random.choice(range(1, 9))
            topic = np.random.choice(topics)
            likes = np.random.poisson(50)
            shares = np.random.poisson(15)
            comments = np.random.poisson(10)
            
            # Platform-specific sentiment bias
            base_positive = {
                'LinkedIn': 0.65, 'Instagram': 0.60, 'TikTok': 0.55,
                'Twitter/X': 0.52, 'Facebook': 0.50
            }
            positive_prob = base_positive[platform]
            sentiment_label = np.random.choice(
                ['positive', 'neutral', 'negative'],
                p=[positive_prob, 0.30, 1 - positive_prob - 0.30]
            )
            
            if sentiment_label == 'positive':
                polarity = np.random.uniform(0.3, 1.0)
            elif sentiment_label == 'neutral':
                polarity = np.random.uniform(-0.1, 0.1)
            else:
                polarity = np.random.uniform(-1.0, -0.3)
            
            data.append({
                'platform': platform,
                'week': week,
                'topic': topic,
                'likes': likes,
                'shares': shares,
                'comments': comments,
                'sentiment_label': sentiment_label,
                'polarity_score': round(polarity, 3)
            })
    
    return pd.DataFrame(data)

# ─── Step 2: Sentiment Analysis ───
def analyze_sentiment(df):
    """Apply TextBlob and VADER sentiment analysis."""
    analyzer = SentimentIntensityAnalyzer()
    
    # Generate text for analysis
    texts = df['topic'].apply(lambda x: f"This brand {x} is worth noting.")
    
    # TextBlob polarity
    df['textblob_polarity'] = texts.apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # VADER compound
    df['vader_compound'] = texts.apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    # Hybrid score (average of both)
    df['hybrid_score'] = (df['textblob_polarity'] + df['vader_compound']) / 2
    
    # Classify hybrid
    def classify_hybrid(score):
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        return 'neutral'
    
    df['hybrid_sentiment'] = df['hybrid_score'].apply(classify_hybrid)
    
    return df

# ─── Step 3: Generate Visualizations ───
def create_sentiment_trend_chart(df):
    """Create weekly sentiment trend line chart."""
    weekly = df.groupby('week').apply(
        lambda g: pd.Series({
            'positive': (g['hybrid_sentiment'] == 'positive').mean() * 100,
            'neutral': (g['hybrid_sentiment'] == 'neutral').mean() * 100,
            'negative': (g['hybrid_sentiment'] == 'negative').mean() * 100,
        })
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(weekly['week'], weekly['positive'], 'o-', color=TEAL, linewidth=2.5, label='Positive', markersize=6)
    ax.plot(weekly['week'], weekly['neutral'], 's-', color=GRAY, linewidth=2, label='Neutral', markersize=5)
    ax.plot(weekly['week'], weekly['negative'], '^-', color=RED, linewidth=2.5, label='Negative', markersize=6)
    
    ax.fill_between(weekly['week'], weekly['positive'], alpha=0.08, color=TEAL)
    ax.fill_between(weekly['week'], weekly['negative'], alpha=0.08, color=RED)
    
    ax.set_xlabel('Week', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Brand Sentiment Trends Over 8 Weeks', fontsize=18, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    
    plt.tight_layout()
    plt.savefig('output/sentiment_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Sentiment trend chart saved.")
    return weekly

def create_sentiment_distribution(df):
    """Create sentiment distribution pie chart."""
    counts = df['hybrid_sentiment'].value_counts()
    labels = counts.index.str.capitalize()
    sizes = counts.values
    colors = [TEAL, GRAY, RED]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.75, wedgeprops={'width': 0.45}
    )
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(12)
    
    ax.set_title('Overall Sentiment Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('output/sentiment_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Sentiment distribution chart saved.")
    return counts

def create_platform_comparison(df):
    """Create platform-level sentiment comparison."""
    platform_sentiment = df.groupby('platform').apply(
        lambda g: pd.Series({
            'avg_polarity': g['hybrid_score'].mean(),
            'engagement': g[['likes', 'shares', 'comments']].sum().sum(),
            'post_count': len(g)
        })
    ).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sentiment by platform
    colors_plat = [TEAL if v > 0 else RED for v in platform_sentiment['avg_polarity']]
    bars = ax1.barh(platform_sentiment['platform'], platform_sentiment['avg_polarity'], color=colors_plat)
    ax1.set_xlabel('Average Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Sentiment Score by Platform', fontsize=16, fontweight='bold', pad=15)
    for bar, val in zip(bars, platform_sentiment['avg_polarity']):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=11, fontweight='bold')
    
    # Engagement by platform
    ax2.barh(platform_sentiment['platform'], platform_sentiment['engagement'], color=AMBER, alpha=0.85)
    ax2.set_xlabel('Total Engagement (Likes + Shares + Comments)', fontsize=12, fontweight='bold')
    ax2.set_title('Engagement Volume by Platform', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('output/platform_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Platform comparison chart saved.")
    return platform_sentiment

def create_keyword_heatmap(df):
    """Create topic-keyword heatmap showing sentiment by topic."""
    topic_sentiment = df.groupby(['topic', 'platform']).agg(
        avg_score=('hybrid_score', 'mean'),
        count=('hybrid_score', 'count')
    ).reset_index()
    
    pivot = topic_sentiment.pivot_table(
        values='avg_score', index='topic', columns='platform', aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
        ax=ax, linewidths=0.5, annot_kws={'fontsize': 9}
    )
    ax.set_title('Sentiment Score by Topic & Platform', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('output/keyword_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Keyword heatmap saved.")
    return pivot

# ─── Step 4: Generate Summary Report ───
def generate_report(df, weekly, dist_counts, platform_stats):
    """Generate a text-based analysis report."""
    report = []
    report.append("=" * 70)
    report.append("BRAND SENTIMENT ANALYSIS — EXECUTIVE SUMMARY")
    report.append("=" * 70)
    report.append(f"\nData Points Analyzed: {len(df):,}")
    report.append(f"Platforms Tracked: {df['platform'].nunique()}")
    report.append(f"Time Period: 8 weeks")
    report.append(f"\n{'─' * 50}")
    report.append("SENTIMENT BREAKDOWN")
    report.append(f"{'─' * 50}")
    
    total = len(df)
    for label in ['positive', 'neutral', 'negative']:
        count = (df['hybrid_sentiment'] == label).sum()
        pct = count / total * 100
        report.append(f"  {label.capitalize():12s}: {count:>5,} ({pct:.1f}%)")
    
    report.append(f"\n{'─' * 50}")
    report.append("PLATFORM RANKING (by Sentiment Score)")
    report.append(f"{'─' * 50}")
    
    for _, row in platform_stats.sort_values('avg_polarity', ascending=False).iterrows():
        report.append(f"  {row['platform']:12s}: {row['avg_polarity']:+.3f} (n={int(row['post_count'])})")
    
    report.append(f"\n{'─' * 50}")
    report.append("TREND ANALYSIS")
    report.append(f"{'─' * 50}")
    
    first_week_pos = weekly.loc[0, 'positive']
    last_week_pos = weekly.loc[len(weekly)-1, 'positive']
    improvement = last_week_pos - first_week_pos
    report.append(f"  Week 1 Positive: {first_week_pos:.1f}%")
    report.append(f"  Week 8 Positive: {last_week_pos:.1f}%")
    report.append(f"  Improvement:     {improvement:+.1f} percentage points")
    
    first_week_neg = weekly.loc[0, 'negative']
    last_week_neg = weekly.loc[len(weekly)-1, 'negative']
    neg_reduction = ((first_week_neg - last_week_neg) / first_week_neg) * 100
    report.append(f"  Week 1 Negative: {first_week_neg:.1f}%")
    report.append(f"  Week 8 Negative: {last_week_neg:.1f}%")
    report.append(f"  Negative Reduction: {neg_reduction:.1f}%")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    with open('output/analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n✓ Report saved to output/analysis_report.txt")

# ─── Main Execution ───
if __name__ == '__main__':
    print("\n🔍 Brand Sentiment Analysis Pipeline")
    print("=" * 45)
    
    print("\n[1/4] Generating sample social media data...")
    df = generate_social_data()
    print(f"      {len(df):,} posts across {df['platform'].nunique()} platforms")
    
    print("\n[2/4] Running sentiment analysis (TextBlob + VADER)...")
    df = analyze_sentiment(df)
    
    print("\n[3/4] Generating visualizations...")
    weekly = create_sentiment_trend_chart(df)
    dist_counts = create_sentiment_distribution(df)
    platform_stats = create_platform_comparison(df)
    create_keyword_heatmap(df)
    
    print("\n[4/4] Generating executive report...")
    generate_report(df, weekly, dist_counts, platform_stats)
    
    print("\n✅ Analysis complete. All outputs saved to ./output/")

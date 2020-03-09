---
title: "NoMoney - Gathering Data"
date: 2020-03-01
tags: [data-collection, nomoney]
categories: ds

excerpt: "Gathering data for informing cryptocurrency trading"
usemathjax: "true"
---

# Collecting Data
When I first started looking into how I'd like to use machine learning for trading, I thought about what data was out there for me to look at that could *potentially* help me with informed trading. It's possible that data I collect could be informative or uninformative, but I needed to collect the data first and perform some analysis before I could arrive at any conclusion. There were three sources of data that I wanted to take a look at when I started:  
1. Traditional price data (open, close, volume, etc.)
2. Wikipedia pageviews
3. Reddit submissions

This notebook goes over how I gathered that data and how I stored them as simple .csv's. (Note that as data grows in volume and variety, using .csv's is non-ideal)

# Price Data
At the end of the day, my expectations of future prices is what is going to drive my trading bot. It's pretty obvious that price data will be needed for any informed trading I wish to do. The ccxt library offers a convenient way to gather OHLCV (open, high, low, close, volume) data for trading cryptocurrency.


```python
import pandas as pd

from datetime import datetime
import ccxt
import plotly.graph_objects as go
```


```python
binance = ccxt.binance()
pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
```


```python
dfs = []

for p in pairs:
    candles = binance.fetch_ohlcv(p, '1d')
    
    pair = []
    date = []
    open_rate = []
    high_rate = []
    low_rate = []
    close_rate = []
    volume = []
    
    for candle in candles:
        pair.append(p)
        date.append(datetime.fromtimestamp(candle[0] / 1000.0).strftime('%Y-%m-%d'))
        open_rate.append(candle[1])
        high_rate.append(candle[2])
        low_rate.append(candle[3])
        close_rate.append(candle[4])
        volume.append(candle[5])
    
    df = pd.DataFrame({
        'pair' : pair,
        'date' : date,
        'open' : open_rate,
        'high' : high_rate,
        'low' : low_rate,
        'close' : close_rate,
        'volume' : volume
    })
    
    dfs.append(df)
    
crypto_prices = pd.concat(dfs).reset_index(drop=True)
```


```python
crypto_prices.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pair</th>
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BTC/USDT</td>
      <td>2018-10-26</td>
      <td>6538.63</td>
      <td>6558.35</td>
      <td>6463.04</td>
      <td>6505.60</td>
      <td>7776.907864</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BTC/USDT</td>
      <td>2018-10-27</td>
      <td>6505.60</td>
      <td>6514.17</td>
      <td>6453.00</td>
      <td>6489.93</td>
      <td>5743.933728</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BTC/USDT</td>
      <td>2018-10-28</td>
      <td>6489.93</td>
      <td>6505.01</td>
      <td>6315.00</td>
      <td>6344.50</td>
      <td>13827.224067</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BTC/USDT</td>
      <td>2018-10-29</td>
      <td>6344.50</td>
      <td>6395.00</td>
      <td>6317.01</td>
      <td>6330.87</td>
      <td>8877.175385</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BTC/USDT</td>
      <td>2018-10-30</td>
      <td>6330.01</td>
      <td>6428.00</td>
      <td>6245.02</td>
      <td>6371.93</td>
      <td>12148.888216</td>
    </tr>
  </tbody>
</table>
</div>




```python
crypto_prices.to_csv('../data/crypto_prices.csv')
```

# Wikipedia Pageviews
Another variable that interested me is seeing how many people are looking up topics such as cryptocurrency, Bitcoin, etc. on Wikipedia. The Wikimedia provides the mwviews library to collect this information. My intuition tells me that higher page views would correlate with higher interest in a certain cryptocurrency, but it could also be because they were mentioned a lot in the news recently (whether good or bad). 


```python
from mwviews.api import PageviewsClient
```


```python
start_date = crypto_prices.date.min().replace('-','')
end_date = crypto_prices.date.max().replace('-','')
```


```python
pv = PageviewsClient(user_agent='Gathering cryptocurrency pageview information')
```


```python
views = pv.article_views('en.wikipedia', ['Bitcoin', 'Ripple_(payment_protocol)', 'Ethereum', 'Cryptocurrency'],
                granularity='daily', start=start_date, end=end_date)
```


```python
wiki_views = pd.DataFrame.from_dict(views, orient='index')
wiki_views = wiki_views.rename(str.lower, axis='columns')
wiki_views = wiki_views.reset_index().rename(columns={'index':'date', 'ripple_(payment_protocol)':'ripple'})
```


```python
wiki_views.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>bitcoin</th>
      <th>ripple</th>
      <th>ethereum</th>
      <th>cryptocurrency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-10-26</td>
      <td>9548</td>
      <td>952</td>
      <td>1788</td>
      <td>5563</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-10-27</td>
      <td>9565</td>
      <td>717</td>
      <td>1543</td>
      <td>3802</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-10-28</td>
      <td>8465</td>
      <td>802</td>
      <td>1842</td>
      <td>4097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-10-29</td>
      <td>11526</td>
      <td>973</td>
      <td>2391</td>
      <td>5107</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-10-30</td>
      <td>12133</td>
      <td>1473</td>
      <td>2269</td>
      <td>5018</td>
    </tr>
  </tbody>
</table>
</div>




```python
wiki_views.to_csv('../data/wiki_views.csv')
```

# Reddit Submissions
There are countless subreddits on Reddit relating to cryptocurrency, whether they are to discuss specific coins (/r/bitcoin, /r/ethereum, /r/ripple, and more), trading (/r/cryptocurrencytrading), general cryptocurrency (/r/cryptocurrency), or more. They provide countless variables that can be looked at for trading insights: comment and submission count, subreddit growth, or text data in the form of comments and submissions. For this example specifically, I'm only going to be saving the titles of popular submissions that I can later perform sentiment analysis on.

Unlike the data for exchange rates or Wikipedia pageviews, retrieving data from Reddit requires signing up for an account and using your own API key. Reddit provides the conditions and instructions to use their API [here](https://www.reddit.com/wiki/api).

I'm not going to be sharing my Reddit keys here, but once you have your own CLIENT_ID and CLIENT_SECRET for Reddit, you can look at [here](https://towardsdatascience.com/how-to-hide-your-api-keys-in-python-fb2e1a61b0a0) to see how to set them as environment variables like I did.


```python
import praw
import os
```


```python
CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
```


```python
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                    user_agent='gathering cryptocurrency community data - /u/crafting_vh')
```


```python
subreddits = ['bitcoin', 'ethereum', 'ripple', 'cryptocurrency']

dfs = []
for sub in subreddits:
    subreddit = []
    title = []
    score = []
    created_utc = []
    
    for s in reddit.subreddit(sub).top('all', limit=100):
        subreddit.append(sub)
        title.append(s.title)
        score.append(s.score)
        created_utc.append(s.created_utc)
        
    df = pd.DataFrame({
        'subreddit' : subreddit,
        'title' : title,
        'score' : score,
        'created_utc' : created_utc
    })
    
    dfs.append(df)
        
reddit_submissions = pd.concat(dfs).reset_index(drop=True)
```


```python
reddit_submissions.to_csv('../data/reddit_submissions.csv')
```

# References
[1] https://medium.com/coinmonks/python-scripts-for-ccxt-crypto-candlestick-ohlcv-charting-data-83926fa16a13  
[2] https://blog.wikimedia.org/2015/12/14/pageview-data-easily-accessible/  
[3] https://towardsdatascience.com/how-to-hide-your-api-keys-in-python-fb2e1a61b0a0


# Day 1 - Time series forecasting with fbprophet

A quick practice on using Facebook's prophet library for forecasting time series data.


```
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```
df = pd.read_csv('https://raw.githubusercontent.com/andy-vh/ds-portfolio/master/100days/data/suicide_trends.csv', index_col=0)
```


```
df.head(5)
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
      <th>language</th>
      <th>article</th>
      <th>date</th>
      <th>views</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>en</td>
      <td>mental_health</td>
      <td>2016-01-01</td>
      <td>525</td>
    </tr>
    <tr>
      <th>2</th>
      <td>en</td>
      <td>suicide</td>
      <td>2016-01-01</td>
      <td>3845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>en</td>
      <td>suicide_methods</td>
      <td>2016-01-01</td>
      <td>3282</td>
    </tr>
    <tr>
      <th>4</th>
      <td>en</td>
      <td>mental_health</td>
      <td>2016-01-02</td>
      <td>763</td>
    </tr>
    <tr>
      <th>5</th>
      <td>en</td>
      <td>suicide</td>
      <td>2016-01-02</td>
      <td>4121</td>
    </tr>
  </tbody>
</table>
</div>




```
_ = df.pivot_table(index='date', columns='article', values='views').plot(rot=45, title='Article Views (2016 to 2019)')
_ = df[df.date>='2019-01-01'].pivot_table(index='date', columns='article', values='views').plot(rot=45, title='Article Views (2019 to present)')
_ = df[(df.date>='2018-01-01') & (df.date<='2018-12-31')].pivot_table(index='date', columns='article', values='views').plot(rot=45, title='Article Views (2018)')
_ = df[(df.date>='2017-01-01') & (df.date<='2017-12-31')].pivot_table(index='date', columns='article', values='views').plot(rot=45, title='Article Views (2017)')
_ = df[(df.date>='2016-01-01') & (df.date<='2016-12-31')].pivot_table(index='date', columns='article', values='views').plot(rot=45, title='Article Views (2016)')
```


![png](day1_prophet_files/day1_prophet_4_0.png)



![png](day1_prophet_files/day1_prophet_4_1.png)



![png](day1_prophet_files/day1_prophet_4_2.png)



![png](day1_prophet_files/day1_prophet_4_3.png)



![png](day1_prophet_files/day1_prophet_4_4.png)


Not directly related to any analysis I'll be doing in this post, but it's interesting that the page on "suicide methods" started becoming more popular than "suicide" in 2018.


```
df_1 = df[df.article=='suicide'][['date', 'views']].rename(columns={'date':'ds', 'views':'y'})
```


```
m_1 = Prophet()
m_1.fit(df_1)
```

    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    




    <fbprophet.forecaster.Prophet at 0x7f6e442aa080>




```
ft_1 = m_1.make_future_dataframe(periods=365)
```


```
fc_1 = m_1.predict(ft_1)
```


```
_ = m_1.plot(fc_1)
```


![png](day1_prophet_files/day1_prophet_10_0.png)



```
_ = m_1.plot_components(fc_1)
```


![png](day1_prophet_files/day1_prophet_11_0.png)


The model seems to expect an increase in views for the "suicide" article for the next year. Another interesting thing of note is that Saturdays tend to have less views, which reminds me of [Crisistrends showing less texters experiencing anxiety/stress on Fridays and Saturdays.](https://crisistrends.org/) These trends could be because people are less stressed out on weekends.


```
df_2 = df[df.article=='suicide_methods'][['date', 'views']].rename(columns={'date':'ds', 'views':'y'})
m_2 = Prophet()
m_2.fit(df_2)
ft_2 = m_2.make_future_dataframe(periods=365)
fc_2 = m_2.predict(ft_2)
_ = m_2.plot(fc_2)
_ = m_2.plot_components(fc_2)
```

    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    


![png](day1_prophet_files/day1_prophet_13_1.png)



![png](day1_prophet_files/day1_prophet_13_2.png)


# Summary and Findings
* I went into the data expecting that there would be more views for suicide and suicide methods in the winter months (i.e. when holidays such as Christmas and New Years happen). However that is not the case, as for both "suicide" and "suicide methods", the average views tend to be lower during the holiday season. Further Googling led me to find that [suicide rates spike in spring, not winter.](https://www.hopkinsmedicine.org/news/articles/suicide-rates-spike-in-spring-not-winter)
* Fbprophet allows for really quick and easy analysis of time series data, without requiring the user to understand the math behind the tool. Understanding what's happening behind the scenes and making adjustments would be useful to improve the model, but being able to easily get quick and interpretable results easily makes fbprophet one of my favorite tools for time series analysis.

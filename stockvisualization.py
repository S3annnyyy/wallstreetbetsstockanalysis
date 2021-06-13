import sys
import time
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as stat
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from sklearn import preprocessing


def main():
    # check command line argument
    if len(sys.argv) != 2:
        sys.exit("Usage: stockvisualization.py corpus")

    # initiate time
    start_time = time.time()

    # read dataset
    corpus = pd.read_csv(r"C:\Users\Sean\PycharmProjects\wallstreetbetsanalysis\{}".format(sys.argv[1]), encoding="latin-1")

    # Form hypotheses
    """I will be determining two things, 
    1) Is there a correlation between the number of mentions vs the volume of stock being traded
    2) Is there a correlation between the sentiment polarity scores of posts vs the price of the stock"""

    # split corpus into 3 df for the 3 stocks that I am focusing on (save it to reduce processing time on further code)
    """Based on NER's evaluation, the top 3 stocks are GameStop, Tesla and AMC. Thus I shall focus primarily on this
    3 stocks"""
    tesla_df, gme_df, amc_df = split_dataframe(corpus)

    # get stock tickers data along with end and start date of stock data based on the 3 df collected
    tesla_stock = get_stock_data(tesla_df, "TSLA")
    gme_stock = get_stock_data(gme_df, "GME")
    amc_stock = get_stock_data(amc_df, "AMC")

    # correlation plot for num of mentions vs vol of stock traded
    tesla_data = mention_against_vol_plot(tesla_df, tesla_stock, "Tesla")
    gme_data = mention_against_vol_plot(gme_df, gme_stock, "Gamestop")
    amc_data = mention_against_vol_plot(amc_df, amc_stock, "AMC")

    # correlation plot for sentiment polarity scores vs price of the stock
    sentiment_against_price_plot(tesla_data, tesla_df, "Tesla")
    sentiment_against_price_plot(gme_data, gme_df, "Gamestop")
    sentiment_against_price_plot(amc_data, amc_df, "AMC")
    print("{} minutes | {} seconds".format(round(((time.time() - start_time) / 60), 1), round((time.time() - start_time), 3)))


def split_dataframe(dataframe):
    # entity variations
    tesla_entities = ["tsla", "tesla"]
    gme_entities = ["gme", "gamestop"]
    amc_entities = ["amc"]
    # prepping dataframes
    tesla_df = pd.DataFrame()
    gme_df = pd.DataFrame()
    amc_df = pd.DataFrame()

    for index, row in tqdm(dataframe.iterrows()):
        if row["organizations"] == "[]":
            continue
        else:
            tickers = row["organizations"].strip('[]').split(', ')
            for i in tickers:
                if i.lower() in tesla_entities:
                    tesla_df = tesla_df.append(dataframe.iloc[index], ignore_index=True)
                    break
                elif i.lower() in gme_entities:
                    gme_df = gme_df.append(dataframe.iloc[index], ignore_index=True)
                    break
                elif i.lower() in amc_entities:
                    amc_df = amc_df.append(dataframe.iloc[index], ignore_index=True)
                    break

    return tesla_df, gme_df, amc_df


def get_stock_data(stock_df, stock_ticker):
    """Purpose of this function is to:
     1) get the start_date and end_date from stock ticker
     2) input start_date and end_date along with stock ticker to get price of stock within given date
     3) get volume data of stock"""
    # get start_date and end_date
    start_date = datetime.utcfromtimestamp(stock_df["utc"][0])
    end_date = datetime.utcfromtimestamp(stock_df["utc"][len(stock_df) - 1])
    # get stock data
    stock = yf.Ticker(stock_ticker).history(start=start_date, end=end_date, interval="1d")
    # modify data (date is parked under index and not in the dataframe itself)
    df_date = stock.index.to_frame().reset_index(drop=True)
    stock_data = df_date.merge(stock, on='Date')
    stock_data = stock_data.rename({'Date': 'date', 'Volume': 'volume', 'Close': 'close'}, axis=1)
    return stock_data


def utc_to_datetime(utc):
    x = datetime.strftime(datetime.utcfromtimestamp(utc), "%Y-%m-%d")
    return x


def mention_against_vol_plot(posts, stock, name):
    """Purpose of this function is to: posts = df, stock = ticker stock prices, volume, etc
        1) get vol from stock data
        2) reindex stock vol data and reddit posts data
        3) Use spearman correlation test to determine hypothesis
        4) plot the numbers of mentions of stock and the volume of stock traded each day
        5) include stock close price and polarity for next hypotheses"""

    def date_iteration(df):
        x = [utc_to_datetime(i) for i in df["utc"].values.tolist()]
        date_count = Counter(x)
        # convert to df
        y = (pd.DataFrame.from_dict(date_count, orient='index').reset_index()).rename(columns={'index': 'date', 0: 'count'})
        y["date"] = y["date"].apply(pd.to_datetime)  # convert type object to datetime64[ns] to prevent merging error
        return y

    # convert reddit post's utc to dates and count occurrences per date
    dates = date_iteration(posts)
    # get volume of stock data with corresponding date
    stock_data = stock[["date", "volume", "close"]]
    # merge stock_vol with dates with corresponding counts
    final = pd.merge(left=dates, right=stock_data, on='date', how='left')
    final = final.set_index('date')
    # spearman correlation
    corr, pval = stat.spearmanr(final['count'].values, final['volume'].values, nan_policy='omit')

    # kendall correlation
    tau, pval1 = stat.kendalltau(final['count'].values, final['volume'].values, nan_policy='omit')

    # plot
    # eliminate nan values
    final.dropna(inplace=True)
    # rolling mean
    stockvol_rol_mean = final["volume"].dropna().rolling(10).mean()
    mention_rol_mean = final["count"].dropna().rolling(10).mean()
    # plot
    fig, ax = plt.subplots()
    ax.plot(stockvol_rol_mean.index, stockvol_rol_mean, label="Stock trading volume")
    ax.ticklabel_format(style='plain', axis='y')  # eliminates scientific notification
    ax.set_xlabel("Year/Month")
    ax.set_ylabel("Trading volume")
    ax2 = ax.twinx()
    ax2.plot(mention_rol_mean.index, mention_rol_mean, label="Number of mentions", color="sandybrown")
    ax2.set_ylabel("No. of mentions")
    if pval < 0.0005:
        ax2.set_title("Effects of {} posts to trading volume".format(name) + "\n Spearman correlation: {}, p-value: {}".format(round(corr, 5), format(pval, '.2e')))
        ax2.set_title("Effects of {} posts to trading volume".format(name) + "\n Kendall's tau: {}, p-value: {}".format(round(tau, 5), format(pval1, '.2e')))

    else:
        ax2.set_title("Effects of {} posts to trading volume".format(name) + "\n Spearman correlation: {}, p-value: {}".format(round(corr, 5), round(pval, 4)))
        ax2.set_title("Effects of {} posts to trading volume".format(name) + "\n Kendall's tau: {}, p-value: {}".format(round(tau, 5), round(pval1, 4)))

    handle, label = ax.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    ax2.legend(handle + handle2, label + label2, loc=0)
    plt.show()
    return final


def sentiment_against_price_plot(data, df, name):
    """Purpose of this function is to collate overall sentiments of posts w.r.t stock and determine correlation
    with price and plot results out"""

    def posneg_count(dataframe):
        positive = []
        negative = []
        x = dataframe.groupby('date')["polarity"]
        y = dict(list(x))
        for i in y.keys():
            pos_count = 0
            neg_count = 0
            for j in y[i]:
                if j > 0.05:
                    pos_count += 1
                elif j < - 0.05:
                    neg_count += 1

            positive.append(pos_count)
            negative.append(neg_count)
        return positive, negative, list(y.keys())
    # convert utc to datetime
    df["date"] = df["utc"].apply(utc_to_datetime)
    df["date"] = df["date"].apply(pd.to_datetime)

    pos, neg, index = posneg_count(df)

    # merge df
    polarity = df.groupby('date')["polarity"].mean().reset_index()
    final = pd.merge(left=data, right=polarity, how='left', on='date')
    final = final.set_index('date')

    # spearman correlation
    corr, pval = stat.spearmanr(final["close"].values, final["polarity"].values, nan_policy='omit')

    # plot
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(final["close"].index, final["close"], label="Stock price")
    ax[0].set_xlabel("Month/Year")
    ax[0].set_ylabel("Stock price")
    if pval < 0.05:
        ax[0].set_title("Effects of {} posts to stock price".format(name) + "\n Spearman correlation: {}, p-value: {}".format(round(corr, 5), format(pval, '.2e')))
    else:
        ax[0].set_title("Effects of {} posts to stock price".format(name) + "\n Spearman correlation: {}, p-value: {}".format(round(corr, 5), round(pval, 4)))

    ax[1].set_xlabel("Month/Year")
    ax[1].bar(index, neg, label="Negative posts", color="crimson")
    ax[1].bar(index, pos, bottom=neg, label="Postive posts", color="limegreen")
    handle, label = ax[1].get_legend_handles_labels()
    ax[1].legend(handle, label, loc=0)
    ax[1].set_ylabel("Number of posts")
    plt.show()


if __name__ == "__main__":
    main()

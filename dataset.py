import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm


def main():
    # Time measurement
    start_time = time.time()

    # datetime period for collecting data
    start_date = 1585670400  # April 1st 2020, time where GME was around $3 a share
    end_date = 1618110903  # Today

    # get dataset
    dataset = pull_request("wallstreetbets", end_date, start_date)

    # convert to dataframe
    df = pd.DataFrame(dataset)

    # data preprocessing
    # removed removed posts and posts with empty texts and any blank rows
    df["text"].replace(to_replace={"": np.nan, "[removed]": np.nan, "[deleted]": np.nan}, value=None, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # export dataframe into csv file
    df.to_csv(r"C:\\Users\Sean\PycharmProjects\wallstreetbetsanalysis\datasettesting.csv")

    print("{}mins|{}seconds".format(round(((time.time() - start_time) / 60), 1), round((time.time() - start_time))))


def get_request(url):
    """Get http request"""
    response = requests.get(url)
    assert response.status_code == 200
    return response.json()


def make_request(url, max_tries=5):
    """This function is to add in logic to request for more post submissions"""
    tries = 1
    while tries < max_tries:
        try:
            time.sleep(0.5)
            json_response = get_request(url)
            return json_response
        except:
            tries += 1
            print("Error.... trying to get request attempt {}".format(tries))
    return get_request(url)


def pull_request(subreddit, before_date, after_date):
    SIZE = 100

    def add_submissions(submission):
        temp_collection = []
        for i in range(SIZE):
            try:
                temp = dict(header=submission["data"][i]["title"], text=submission["data"][i]["selftext"],
                            utc=submission["data"][i]["created_utc"])
                # removes any posts with text removed
                temp_collection.append(temp)
            except (KeyError, IndexError):  # ignoring submissions without text, only title
                continue
        return temp_collection

    url_template = "https://api.pushshift.io/reddit/search/submission/?subreddit={}&after={}&before={}&size={}"
    database = add_submissions(make_request(url_template.format(subreddit, after_date, before_date, SIZE)))

    # iteration to get posts till
    last_utc = database[-1]["utc"]
    posts = 100
    while last_utc != 1618110903:
        print("HTTP request injection ongoing...")
        next_database = add_submissions(make_request(url_template.format(subreddit, last_utc, before_date, SIZE)))
        try:
            last_utc = next_database[-1]["utc"]
            for j in tqdm(next_database):
                database.append(j)
                posts += 1
        except IndexError:
            last_utc = 1618110903

    print("Current date: {}".format(datetime.fromtimestamp(last_utc)))
    print("Total number of posts: {}".format(posts))

    return database


if __name__ == "__main__":
    main()

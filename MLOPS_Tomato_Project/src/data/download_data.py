import opendatasets as od


def download_data(): 
        od.download("https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources", data_dir = "./data/raw")

download_data()
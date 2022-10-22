from dataclasses import replace
import numpy as np
import datetime
import pandas as pd
from sklearn.cluster import DBSCAN
from utils import (#get_dates, 
                   #get_files, 
                   extract_mta_data,
                   #resample_counts,
                   get_entries_exits,
                   anomaly_replacement
                  )
# add logging
# add arg parse for chossing start date and number of weeks


def main(start_date, weeks=1, replace_anomalies=False):
    df = extract_mta_data(start_date, weeks=weeks)
    df_ent, df_ex = get_entries_exits(df, replace_anomalies)
    df_ent.columns = [x+"_ent" for x in df_ent.columns]
    df_ex.columns = [x+"_ex" for x in df_ex.columns]
    df_out = pd.concat([df_ent,df_ex], axis=1)
    return df_out[sorted(df_out.columns)]
        
        
if __name__ == "__main__":
    #start_date = "220108"
    #weeks = 39
    start_date = "210109"
    weeks = 90
    replace_anomalies = True

    df = main(start_date,
              weeks=weeks, 
              replace_anomalies=replace_anomalies)
    if replace_anomalies == True:
        file_name = f"../data/mta_subway_{start_date}_{weeks}wk_dbscan.parquet"
    elif replace_anomalies == False:
        file_name = f"../data/mta_subway_{start_date}_{weeks}wk.parquet"
    df.to_parquet(file_name)
    
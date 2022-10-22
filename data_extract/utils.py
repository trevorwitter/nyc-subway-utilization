import numpy as np
import datetime
import pandas as pd
from sklearn.cluster import DBSCAN


def convert_start_date(date):
    """converts selected date to saturday of same week"""

    pass


def get_dates(start_date, weeks=1):
    # update to change start date to last saturday if date is not saturday
    #start_date = convert_start_date(date)
    dates = []

    date_1 = datetime.datetime.strptime(start_date, "%y%m%d")
    dates.append(date_1.strftime("%y%m%d"))
    for x in range(1,weeks):
        next_date = (date_1 + datetime.timedelta(weeks=x)).strftime("%y%m%d")
        dates.append(next_date)
    return dates


def get_files(start_date, weeks):
    files = []
    dates = get_dates(start_date, weeks)
    for d in dates:
        files.append(f'http://web.mta.info/developers/data/nyct/turnstile/turnstile_{d}.txt')
    return files


def extract_mta_data(start_date, weeks=1):
    """Extracts weekly data files from MTA for range of weeks.
       To do: Need to hardcode columns so they stay consistent 
       for model input"""
    file_locs = get_files(start_date, weeks)
    df = pd.DataFrame()
    errs = []
    for file in file_locs:
        try:
            df_ = pd.read_csv(file)
        except:
            errs.append(file)
            df_ = pd.read_csv(file, 
                              engine='python', 
                              sep=',', 
                              #lineterminator='\n'
                             )
        df = df.append(df_)
    df = df.reset_index(drop=True)
    df.columns = [x.strip() for x in df.columns]
    df['TIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df = df.sort_values(['STATION','C/A','UNIT','SCP','TIME'])
    df = df[df['TIME'].dt.minute==0]
    #print(f"{len(errs)} files with errors:")
    #print(errs)
    return df

def anomaly_replacement(df):
    #df_out = pd.DataFrame()
    df_out = df.copy()
    #for x in df.columns:
    for x in ['ent','ex']:
        # Find way to not fillna until after dbscan; may be filling with outliers sometimes
        # Step 1: Label where nan are located
        # Step 2: Fill na with mean value
        # Step 4: dbscan 
        # Step 5: revert nan's back to nan
        # Step 5: outliers set to nan
        df[[x]] = df[[x]].fillna(method='bfill').fillna(method='ffill')
        _df = df[[x]]
        #previously used eps 220
        eps = int(_df[x].quantile(0.99)*1.5)
        if eps < 220:
            eps = 220
        dbscan = DBSCAN(eps=220, 
                        min_samples=9)
        dbscan.fit(_df)
        outliers = np.argwhere(dbscan.labels_ == -1).flatten()
        _df.iloc[outliers,:] = np.nan
        _df = _df.fillna(method='bfill').fillna(method='ffill')
        df_out[x] = _df
    return df_out


def resample_counts(df):
    """Resamples data to hourly; raw data is pulled every four 
       hours or so at different times for different stations"""
    df2 = df.copy()
    df2['diff'] = df['TIME'].diff()/np.timedelta64(1,'h')
    df2 = df2.set_index('TIME')
    df2['ent'] = df2['ent']/df2['diff']
    df2['ex'] = df2['ex']/df2['diff']
    return df2[['ent','ex']].resample('H').mean().bfill().reset_index()


def get_entries_exits(df, replace_anomalies=False):
    """Returns dfs for entires and exits by station"""
    df['ent'] = df[['ENTRIES',
                    'STATION',
                    'UNIT',
                    'SCP',
                    'C/A']].groupby(['STATION','UNIT','SCP','C/A']).diff()#.abs()
    df['ex'] = df[['EXITS',
                   'STATION',
                   'UNIT',
                   'SCP',
                   'C/A']].groupby(['STATION','UNIT','SCP','C/A']).diff()#.abs()
    #df['ent'].loc[df['ent']< -100] = df['ENTRIES'].loc[df['ent']< -100]
    #df['ex'].loc[df['ex']< -100] = df['EXITS'].loc[df['ex']< -100]
    df['ent'].loc[df['ent']< 0] = np.nan
    df['ex'].loc[df['ex']< 0] = np.nan
    #df['ent'] = df['ent'].abs()
    #df['ex'] = df['ex'].abs()
    df = df[['TIME','STATION','ent','ex']]
    
    #consider moving anomaly detection to here, prior to resample
    # REMOVE ALL ROWS HAVING INDEX TIME NOT ON THE TOP OF THE HOUR
    # ex: df[df.index.minute==0]
    df1 = df[['TIME','STATION','ent','ex']].groupby(['STATION','TIME'])\
                                          .sum()\
                                          .reset_index()
    #Anomaly detection, prior to resample
    if replace_anomalies == True:                        
        df1 = df1[['TIME','STATION','ent','ex']].groupby('STATION')\
                                                .apply(anomaly_replacement)

    df1 = df1[['TIME','STATION','ent','ex']].groupby('STATION')\
                                            .apply(resample_counts)\
                                            .reset_index()
    
    df_out_ent = df1.pivot(index='TIME',
                           columns='STATION',
                           values='ent')

    df_out_ex = df1.pivot(index='TIME',
                          columns='STATION',
                          values='ex')
    return df_out_ent.ffill().bfill(), df_out_ex.ffill().bfill()
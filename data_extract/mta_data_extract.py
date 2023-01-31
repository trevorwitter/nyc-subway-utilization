from dataclasses import replace
import numpy as np
import datetime
import pandas as pd
from sklearn.cluster import DBSCAN
#from utils import (#get_dates, 
#                   #get_files, 
#                   extract_mta_data,
#                   #resample_counts,
#                   get_entries_exits,
#                   anomaly_replacement
#                  )
# add logging
# add arg parse for chossing start date and number of weeks


class turnstile_data_extractor():
    def __init__(self):
        pass
    def _convert_start_date(self,date):
        date = datetime.datetime.strptime(date, "%y%m%d")
        idx = (date.weekday() + 1) % 7
        sat = date - datetime.timedelta(7+idx-6)
        return sat
    
    def _get_dates(self, start_date, weeks=1):
        start_date = self._convert_start_date(start_date)
        dates = []
        dates.append(start_date.strftime("%y%m%d"))
        for x in range(1,weeks):
            next_date = (start_date - datetime.timedelta(weeks=x)).strftime("%y%m%d")
            dates.append(next_date)
        return sorted(dates)

    def _get_files(self, start_date, weeks):
        files = []
        dates = self._get_dates(start_date, weeks)
        for d in dates:
            files.append(f'http://web.mta.info/developers/data/nyct/turnstile/turnstile_{d}.txt')
        return files

    def _extract_mta_data(self,start_date, weeks=1):
        """Extracts weekly data files from MTA for range of weeks.
        To do: Need to hardcode columns so they stay consistent 
        for model input"""
        file_locs = self._get_files(start_date, weeks)
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
        return df
    
    def _anomaly_replacement(self, df):
        df_out = df.copy()
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
    
    def _resample_counts(self,df):
        """Resamples data to hourly; raw data is pulled every four 
        hours or so at different times for different stations"""
        df2 = df.copy()
        df2['diff'] = df['TIME'].diff()/np.timedelta64(1,'h')
        df2 = df2.set_index('TIME')
        df2['ent'] = df2['ent']/df2['diff']
        df2['ex'] = df2['ex']/df2['diff']
        return df2[['ent','ex']].resample('H').mean().bfill().reset_index()
    
    def get_entries_exits(self, df, replace_anomalies=False):
        """Returns dfs for entires and exits by station"""
        df['ent'] = df[['ENTRIES',
                        'STATION',
                        'UNIT',
                        'SCP',
                        'C/A']].groupby(['STATION','UNIT','SCP','C/A']).diff()
        df['ex'] = df[['EXITS',
                    'STATION',
                    'UNIT',
                    'SCP',
                    'C/A']].groupby(['STATION','UNIT','SCP','C/A']).diff()
        df['ent'].loc[df['ent']< 0] = np.nan
        df['ex'].loc[df['ex']< 0] = np.nan
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
                                                    .apply(self._anomaly_replacement)

        df1 = df1[['TIME','STATION','ent','ex']].groupby('STATION')\
                                                .apply(self._resample_counts)\
                                                .reset_index()
        
        df_out_ent = df1.pivot(index='TIME',
                            columns='STATION',
                            values='ent')

        df_out_ex = df1.pivot(index='TIME',
                            columns='STATION',
                            values='ex')
        return df_out_ent.ffill().bfill(), df_out_ex.ffill().bfill()

    def get_data(self, start_date, weeks=1, replace_anomalies=False):
        df = self._extract_mta_data(start_date, weeks=weeks)
        df_ent, df_ex = self.get_entries_exits(df, replace_anomalies)
        df_ent.columns = [x+"_ent" for x in df_ent.columns]
        df_ex.columns = [x+"_ex" for x in df_ex.columns]
        df_out = pd.concat([df_ent,df_ex], axis=1)
        return df_out[sorted(df_out.columns)]  




if __name__ == "__main__":
    start_date = "221231"
    weeks = 100
    replace_anomalies = True

    df = turnstile_data_extractor().get_data(start_date=start_date,
                                             weeks=weeks, 
                                             replace_anomalies=replace_anomalies) 

    if replace_anomalies == True:
        file_name = f"../data/mta_subway_{start_date}_{weeks}wk_dbscan.parquet"
    elif replace_anomalies == False:
        file_name = f"../data/mta_subway_{start_date}_{weeks}wk.parquet"
    df.to_parquet(file_name)
    

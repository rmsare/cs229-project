import os
import sys
import pandas as pd
import subprocess


def load_hourly(code, year, state, county):
    fn = 'hourly_{}_{}.zip'.format(code, year)
    command = ['wget', 'https://aqs.epa.gov/aqsweb/airdata/' + fn]
    subprocess.check_output(command)
    df = pd.read_csv(fn)
    df = df.loc[df['State Name'].str.contains(state)]
    df = df.loc[df['County Name'].str.contains(county)]
    return df

if __name__ == "__main__":
    codes ={'pm25': 88101,
            }
    key = 'pm25'
    code = codes[key]

    print(sys.argv)

    state = sys.argv[1]
    county = sys.argv[2]

    os.chdir('data/pm25/')
    dfs = []
    for year in range(2008, 2019):
        print('loading {}...'.format(year))
        df = load_hourly(code, year, state, county)
        nobs = df.shape[0]
        if nobs > 0:
            print('got {} records'.format(nobs))
            dfs.append(df)
    df = pd.concat(dfs, axis='rows')
    df.to_csv(county.lower().replace(' ', '_') + '_' + key + '_2008_2018.csv')

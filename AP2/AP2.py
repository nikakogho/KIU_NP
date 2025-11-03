import pandas as pd
import numpy as np

# Step 0: View each indicator at https://data.worldbank.org/indicator/{indicator_code}
# Step 1: Download CSV file for each indicator from https://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=csv
# Step 2: Load and merge on 'Country Name'
df_list = []
for code in ['IT.NET.USER.ZS','IT.NET.BBND.P2','SE.TER.ENRR','SP.POP.SCIE.RD.P6',
             'EG.USE.ELEC.KH.PC','EG.FEC.RNEW.ZS','NY.GDP.PCAP.CD',
             'TX.VAL.TECH.MF.ZS','GB.XPD.RSDV.GD.ZS']:
    temp = pd.read_csv(f'WorldBankData/API_{code}.csv')
    temp = temp[['Country Name','2021']].rename(columns={'2021':code})
    df_list.append(temp)

data = df_list[0]
for d in df_list[1:]:
    data = pd.merge(data,d,on='Country Name',how='inner')

print(data.head())
#!/bin/sh
head -n 10000 'en.openfoodfacts.org.products.csv' >> 'truncated.csv'


#import pandas as pd
#
#df = pd.read_csv("./truncated.csv", sep='\t', )
#df = df.loc[:, df.dtypes == float]
#cols = df.isnull().sum().sort_values()[:12]\
#    .drop(labels=['last_updated_t', 'last_image_t'])\
#    .index.to_list()
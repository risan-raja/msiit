import pandas as pd
from dataclasses import dataclass
orig_columns = ['binary__offer_expiration',
 'ordinal__income_range',
 'ordinal__no_visited_cold_drinks',
 'binary__travelled_more_than_15mins_for_offer',
 'ordinal__restaur_spend_less_than20',
 'nominal__marital_status',
 'nominal__restaurant_type',
 'ordinal__age',
 'binary__prefer_western_over_chinese',
 'binary__travelled_more_than_25mins_for_offer',
 'ordinal__no_visited_bars',
 'binary__gender',
 'nominal__car',
 'binary__restuarant_same_direction_house',
 'binary__cooks_regularly',
 'nominal__customer_type',
 'ordinal__qualif',
 'binary__is_foodie',
 'ordinal__no_take_aways',
 'nominal__job_industry',
 'binary__restuarant_opposite_direction_house',
 'binary__has_children',
 'ordinal__type_of_rest_rating',
 'interval__temperature',
 'ordinal__restaur_spend_greater_than20',
 'interval__travel_time',
 'interval__season',
 'ordinal__dest_distance',
 'binary__prefer_home_food',
 ]
@dataclass
class MyDB(object):
    def __init__(self):
        self.data = pd.read_parquet('kaggle/input/gen-data/data.parquet')
        self.eval_data = pd.read_parquet('kaggle/input/gen-data/eval_data.parquet')
        self.nr_data = pd.read_parquet('kaggle/input/gen-data/nr_data.parquet')
        self.nr_eval_data = pd.read_parquet('kaggle/input/gen-data/nr_eval_data.parquet')
        self.orig_columns = orig_columns
        self.y  = self.data.target
        
    def getX(self,df):
        cols = list(df.columns)
        if 'target' in cols:
            cols.remove('target')
        check1 =  all(item in list(self.eval_data.columns) for item in cols)
        check2 =  all(item in list(self.nr_eval_data.columns) for item in cols)
        if check1:
            return df.loc[:,self.eval_data.columns]
        elif check2:
            return df.loc[:,self.nr_eval_data.columns]
        else:
            raise('Data Not Familiar')
    
    def getorig(self,df):
        return df.loc[:,self.orig_columns]
    
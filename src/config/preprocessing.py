import numpy  as np
import pandas as pd
from enum import Enum

#####################
## Types
#####################

Data_Types = Enum('Data_Types', ['unsigned int','unsigned big int','unsigned small int', 'numeric', 'string'])
Fill_Strategy = Enum('Fill_Strategy', ['REMOVE','AVERAGE','COMMON'])
#####################
## Constants
#####################

DATA_MAPPER =\
  { Data_Types['unsigned int']       : pd.UInt32Dtype()
  , Data_Types['unsigned big int']   : pd.UInt64Dtype()
  , Data_Types['unsigned small int'] : pd.UInt8Dtype()
  , Data_Types['string']             : pd.StringDtype()
  , Data_Types['numeric']            : pd.Float32Dtype()
  }



#####################
## Aux utilities
#####################

def remove_strategy(df, column):
  df = df[df[column].notna()]
  return df

def common_strategy(df, column):
  df[column].fillna(df[column].mode()[0],inplace=True)
  return df

def average_strategy(df, column):
  df[column].fillna(df[column].mean(),inplace=True)
  return df

def mean_normalize(df):
  df = (df - df.mean()) / df.std()
  return df

def strategy(s):
  match s:
    case Fill_Strategy.REMOVE: 
      return remove_strategy
    case Fill_Strategy.AVERAGE:
      return average_strategy
    case Fill_Strategy.COMMON:
      return common_strategy

def min_max_normalize(df):
  df_max = df.max()
  df_min = df.min()
  df = (df - df_min) / (df_max - df_min)
  return df

def build_category_mapping(category_columns, df):
  mapping = {}
  for column in category_columns:
    mapping[column] = df[column].unique()
    mapping[column] = sorted(mapping[column])

  return mapping

def categorize(category_mapping, df):
  new = df.copy()
  dfs = []
  for column in category_mapping:
    for value in category_mapping[column]:
      #new[f"{column}:{value}"] = df[column].map(lambda x: 1 if x == value else 0)
      dfs.append(pd.DataFrame({f"{column}:{value}":df[column].map(lambda x: 1 if x == value else 0)}))
    new = new.drop(column,axis=1)
  new = pd.concat([new] + dfs,axis=1,sort=False)
  return new
    

#####################
## Primary utilities
#####################

def type_mapper(category):
  try:
    return DATA_MAPPER[Data_Types[category]]
  except (KeyError):
    raise TypeError("Data type not found")

def read_dataset(path,col_dict):
  dtypes = {}
  for row in col_dict:
    dtypes[row["column"]] = type_mapper(row["type"])
  return pd.read_csv(path,usecols=list(dtypes),dtype=dtypes,engine='c',na_values=[""])

def apply_strategy(col_dict,df):
  for row in col_dict:
    column = row["column"]
    f = strategy(Fill_Strategy[row["fill_strategy"]])
    df = f(df,column)
  return df


def builder(path,target,normalize,split,col_dict):
  category_columns = list(map(lambda x: x["column"], filter(lambda x: "categoric" in x and x["categoric"], col_dict)))
  naive_df = read_dataset(path,col_dict)
  not_nulls = apply_strategy(col_dict,naive_df.copy())
  category_mapping = build_category_mapping(category_columns,not_nulls)
  categorized = categorize(category_mapping,not_nulls)
  normalized = min_max_normalize(categorized) if normalize else categorized
  final = normalized
  final[target] = naive_df[target]
  return final




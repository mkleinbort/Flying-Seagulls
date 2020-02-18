from notebook_init import *

def guesstimate_stories(mssubclass):
    if '1-STORY' in mssubclass:
        return 1
    elif '2-STORY' in mssubclass:
        return 2
    elif '1-1/2' in mssubclass:
        return 1.5
    elif '2-1/2' in mssubclass:
        return 2.5
    elif 'SPLIT' in mssubclass:
        return 2
    elif 'DUPLEX' in mssubclass:
        return 2
    elif '2 FAMILY CONVERSION' in mssubclass:
        return 2
    else:
        return np.nan
    
def set_date_from_row(row):
    return pd.to_datetime(f'{row["YrSold"]}-{row["MoSold"]}')

def add_features_from_sale_date(X):
    X_extended = X.assign(SALE_DATE = lambda x: x.apply(set_date_from_row, axis=1))
    return X_extended
                          
def add_featues_from_MSSubClass(X):
    X_extended = (X
                  .assign(MSSubClassHeight = lambda x: x['MSSubClass'].str.split().str.get(0))
                  .assign(PUD = lambda x: x['MSSubClass'].str.contains('PUD'))
                  .assign(UNFINISHED = lambda x: x['MSSubClass'].str.contains('UNFINISHED'))
                  .assign(FINISHED = lambda x: x['MSSubClass'].str.contains(r'\bFINISHED'))
                  .assign(POST1945 = lambda x: x['YearBuilt'] > 1945)
                  .assign(STORIES = lambda x: x['MSSubClass'].apply(guesstimate_stories))
                 )
    return X_extended
                          
def add_features_from_SaleType(X):
    X_extended = (X
                  .assign(SaleType_WarrantyDeed = lambda x: x['SaleType'].isin(['WD','CWD','VWD']))
                  .assign(SaleType_Conventional = lambda x: x['SaleType'].isin(['WD','CWD','VWD']))
                  .assign(SaleType_Cash = lambda x: x['SaleType'].isin(['CWD']))
                  #.assign(SaleType_VALoan = lambda x: x['SaleType'].isin(['VWD']))
                  .assign(SaleType_New = lambda x: x['SaleType'].isin(['New']))
                  .assign(SaleType_CourtOfficer = lambda x: x['SaleType'].isin(['COD']))
                  .assign(SaleType_RegularTerms = lambda x: x['SaleType'].isin(['Con']))
                  .assign(SaleType_Contract = lambda x: x['SaleType'].isin(['Con','ConLw','ConLI','ConLD']))
                  .assign(SaleType_LowDownpayment = lambda x: x['SaleType'].isin(['ConLw','ConLD']))
                  .assign(SaleType_LowInterest = lambda x: x['SaleType'].isin(['ConLw','ConLI']))
                  .assign(SaleType_Other = lambda x: x['SaleType'].isin(['Oth','']))
                 )
    
    return X_extended
                          
def add_features(X):
    X_extended = (X
                  .pipe(add_features_from_sale_date)
                  .pipe(add_featues_from_MSSubClass)
                  .pipe(add_features_from_SaleType)
                 )
    return X_extended
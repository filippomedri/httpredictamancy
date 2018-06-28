import pandas as pd
import os
import pickle
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

ax_dims = (20,12)

sns.set()

# load time series
df = pd.read_csv('requests_ts.csv', parse_dates=['timestamp'], index_col='timestamp')


ar = 0
i = 1
ma = 0
sar = 0
si = 1
sma = 0
sep = '_'
model_type_name = 'sarimax'
extension = '.pkl'
save_dir = 'arima_models'

def load(df, ar, i, ma, sar, si, sma, how_many_hours, save_model= False):
    mod = sm.tsa.statespace.SARIMAX(df.requests[:-how_many_hours], trend='n', order=(ar,i,ma), seasonal_order=(sar,si,sma,24))
    results = mod.fit()
    serialized_name = str(ar)+ sep + str(i) + sep + str(ma) +sep + model_type_name \
                  + sep + str(sar) + sep + str(si) + sep + str(ma) +sep +str(how_many_hours)+ extension
    # save model
    if save_model:
        results.save(os.path.join(save_dir,serialized_name))
    return serialized_name , results

def predict(df,model,how_many_hours, forecast_store, plot_it = False):
    no_records = df['requests'].count()

    series = model.predict(start=no_records - how_many_hours, end=no_records, dynamic=True)
    df[forecast_store] = np.NAN

    df[forecast_store][-how_many_hours:] = series.values[:how_many_hours]


    df[['requests', forecast_store]][-how_many_hours:].plot(figsize=(20, 12))
    plt.title(forecast_store)
    if plot_it:
        plt.show()
    plt.savefig(os.path.join(save_dir,(forecast_store + '.png')))

    return series.values[:how_many_hours]

def evaluate_model(df,ar,i,ma,sar,si,sma,how_many_hours):

    test = df['requests'][-how_many_hours:]
    serialized_name, results = load(df, ar, i, ma, sar, si, sma, how_many_hours)
    predictions = predict(df,results,how_many_hours, serialized_name[:-4])

    # calculate out of sample error
    error = mean_squared_error(test, predictions)

    return error

def evaluate_models(df,how_many_hours,p_values,d_values,q_values,sp_values,sd_values,sq_values):

    best_score, best_cfg = float("inf"), None

    error_dict = {}
    for ar in p_values:
        for i in d_values:
            for ma in q_values:
                for sar in sp_values:
                    for si in sd_values:
                        for sma in sq_values:
                            order = (ar,i,ma,sar,si,sma)
                            try:
                                mse = evaluate_model(df, ar, i, ma, sar, si, sma, how_many_hours)
                                if mse < best_score:
                                    best_score, best_cfg = mse, order
                                print('ARIMA%s MSE=%.3f' % (order,mse))
                                error_dict[mse]= order
                            except:
                                continue
    result = OrderedDict(sorted(error_dict.items()))

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return result


p_values = [x for x in range(2)]
d_values = [1]
q_values = [x for x in range(3)]
sp_values  = [x for x in range(2)]
sd_values = [1]
sq_values  = [x for x in range(3)]
'''

p_values = [x for x in range(1)]
d_values = [1]
q_values = [x for x in range(1)]
sp_values  = [x for x in range(1)]
sd_values = [1]
sq_values  = [x for x in range(1)]

'''

result_24 = evaluate_models(df,24,p_values,d_values,q_values,sp_values,sd_values,sq_values)
with open(os.path.join(save_dir,'24_errors.pkl'),'wb') as f:
    pickle.dump(result_24,f)

result_168 = evaluate_models(df,168,p_values,d_values,q_values,sp_values,sd_values,sq_values)
with open(os.path.join(save_dir,'168_errors.pkl'),'wb') as f:
    pickle.dump(result_168,f)

print('result_24')
print(result_24)
print('result_168')
print(result_168)

df.tail(168).to_csv('plot.csv')

new_df = pd.read_csv('plot.csv')

print(new_df.head())
'''

error = evaluate_model(df,ar,i,ma,sar,si,sma,24)

print('error = ', error)

# create and save model
serialized_name, results = load(df,ar,i,ma,sar,si,sma,168)

# load model
loaded = statsmodels.tsa.statespace.mlemodel.MLEResults.load(os.path.join(save_dir,serialized_name))

print('... \t Loaded Model \t ...')
print(loaded.summary())

print('plot')

predict(df,loaded,168,serialized_name,plot_it=True)
#predict(df,results,24,serialized_name,plot_it=True)

'''
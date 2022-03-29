# import libraries
import argparse;
import numpy as np;
import pandas as pd;
import prophet;
import csv, argparse;
from datetime import datetime;
import matplotlib.pyplot as plt;

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def loadData(trainingDataFName_1, trainingDataFName_2):
    df_data = pd.read_csv(trainingDataFName_1, header=0);
    df_data['日期'] = pd.Series(df_data['日期'], dtype="string");
    df_data['日期'] = pd.to_datetime(df_data['日期'], format='%Y-%m-%d');
    df_data = df_data.drop(df_data.columns[-67:], axis=1);
    df_data = df_data.drop(df_data.columns[1:3], axis=1);
    
    df_data1 = pd.read_csv(trainingDataFName_2, header=0);
    df_data1['日期'] = pd.to_datetime(df_data1['日期'], format='%Y-%m-%d');
    df_data1['備轉容量(MW)'] = df_data1['備轉容量(萬瓩)'].apply(lambda x: 10*int(x));
    df_data1 = df_data1.drop(['備轉容量(萬瓩)','備轉容量率(%)'], axis=1);
    df_data1 = df_data1[-28:];
    
    df_training = pd.concat([df_data,df_data1[-27:]]);
    return df_training;

def makeHolidayData(dataPath_1, dataPath_2):
    h2021 = pd.read_csv(dataPath_1, header=0);
    h2022 = pd.read_csv(dataPath_2, header=0);
    h = pd.concat([h2021, h2022], ignore_index=True);
    h = h.loc[h['是否放假']==2];
    h['西元日期'] = pd.Series(h['西元日期'], dtype="string");
    h['西元日期'] = pd.to_datetime(h['西元日期'], format='%Y-%m-%d')
    superH = h.dropna();
    
    playoffs = pd.DataFrame({
      'holiday': 'playoff',
      'ds': h['西元日期'],
      'lower_window': 0,
      'upper_window': 1,
    });
    superbowls = pd.DataFrame({
      'holiday': 'superbowl',
      'ds': superH['西元日期'],
      'lower_window': 0,
      'upper_window': 1,
    });
    holidays = pd.concat((playoffs, superbowls));
    return holidays;

def buildModel(data, holidays):
    d = {'ds':data['日期'],'y':data['備轉容量(MW)']};
    train = pd.DataFrame(data=d);
    model = prophet.Prophet(holidays = holidays,yearly_seasonality=True,daily_seasonality=True);
    model.fit(train);
    return model;

def definePredictData():
    predict_date = list()
    for i in range(30,32):
        predict_date.append(datetime.strptime(f'2021-03-{i}', '%Y-%m-%d'))
    for i in range(1,14):
        predict_date.append(datetime.strptime(f'2021-04-{i}', '%Y-%m-%d'))
    predict_date = pd.DataFrame(predict_date, columns=['ds'])
    predict_date['ds']= pd.to_datetime(predict_date['ds'])
    return predict_date;

def main(args):
    # load data
    df_training = loadData(args.training, args.training1);
    
    # preprocessing
    min_max_scaler = MinMaxScaler();
    df_training['備轉容量(MW)'] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df_training['備轉容量(MW)'])));
    
    # make holiday data
    holidays = makeHolidayData(args.holiday2021, args.holiday2022);
    
    # build model
    model = buildModel(df_training, holidays);
    
    # define predict data
    predict_date = definePredictData();
    
    # predict
    forecast = model.predict(predict_date);
    
    # inverse normolization
    forecast['yhat'] = pd.DataFrame(min_max_scaler.inverse_transform(pd.DataFrame(forecast['yhat'])));
    forecast = forecast.drop(forecast.columns[1:30], axis=1);
    
    return forecast;

def outputCSV(outputFName, forecast):
    f = open(outputFName,'w',newline='');
    w = csv.writer(f);
    w.writerow(['date','operating_reserve(MW)']);
    for data in forecast.values:
        date = data[0].strftime("%Y%m%d");
        w.writerow([date, np.round(data[1])]);

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',default='training_data.csv',help='input training data file name');
    parser.add_argument('--training1',default='training_data1.csv',help='input training data file name');
    parser.add_argument('--holiday2021',default='2021holiday.csv',help='prophet model argument-1');
    parser.add_argument('--holiday2022',default='2022holiday.csv',help='prophet model argument-2');
    parser.add_argument('--output',default='submission.csv',help='output file name');
    args = parser.parse_args();
    #
    
    forecast = main(args);
    
    # output submission.csv
    outputCSV(args.output, forecast);
    
    # draw
    plt.plot(forecast['ds'], forecast['yhat'], color='red', label='predicted_value');
    plt.legend();
    plt.show();
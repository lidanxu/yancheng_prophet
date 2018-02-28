#-*- coding: utf-8 -*-
from  __future__ import division

'''
尝试使用prophet模块，预测盐城上牌数据
'''
import pandas as pd 
from fbprophet import Prophet
import numpy as np 
import datetime 
'''
m.make_future_dataframe(periods=) #帮助生成待预测日期
'''
def get_data(path):
    df = pd.read_csv(path,sep='\t')

    df['ds'] = pd.to_datetime(df['date'],unit='D')
    df['y'] = df['cnt']
    df = df.drop(['date','day_of_week','brand','cnt'],axis=1)
    return df 

def prophet_model(df,interval=365):
    #instantiating Prophet object
    m = Prophet() 
    #m = Prophet(weekly_seasonality=False)
    #m.add_seasonality(name='monthly',period=30.5,fourier_order=5)
    #m.add_seasonality(name='weekly',period=7,fourier_order=3,prior_scale=0.1)
    m.fit(df)

    future = m.make_future_dataframe(periods=interval)
    #future.tail()
    forecast = m.predict(future)
    return forecast[['ds','yhat']]

def get_data_group(path):
    '''
    不同品牌上牌量预测再合并
    '''
    forecast_result = []
    train = []
    df = pd.read_csv(path,sep='\t')
    df_group = df.groupby('brand')
    for num,d in df_group: 
        #d = fill_nan(d) #填充缺失日期
        d['ds'] = d['date']
        d['y'] = d['cnt']
        d = d.drop(['date','day_of_week','brand','cnt'],axis=1)

        d = fill_nan(d) #填充缺失日期
        d['ds'] = pd.to_datetime(d['ds'],unit='D')
        d.fillna(method='ffill',inplace=True)
        
        #=============================================
        #测试
        #划分数据为训练数据和测试数据
        divided = datetime.datetime(1972,10,1)
        d_train = d[d.ds<divided]
        d_test = d[d.ds>= divided]
        d_test.reset_index(drop=True,inplace=True)

        test.append(d_test)
        forecast = prophet_model(d_train,29)[-29:]
        #预测结果重新编号,pandas.DataFrame比较是按index比较的
        forecast.reset_index(drop=True,inplace=True)
        mse = evaluate_model(forecast,d_test)
        print '%d mse'%num,mse 
        forecast_result.append(forecast)
        #+++++++++++++++++++++++++++++++++++++++++++++
        #预测
        forecast = prophet_model(d_train)
        forecast_result.append(forecast)
    #================================================
    #合计预测结果和test的mse
    forecast_total = forecast_result[1]
    test_total = test[1]
    forecast_total.yhat = sum(x.yhat for x in forecast_result)
    test_total.y = sum(x.y for x in test)
    mse_total = evaluate_model(forecast_total,test_total)
    print 'mse_total:',mse_total
    #++++++++++++++++++++++++++++++++++++++++++++++++
    #预测
    save(forecast_result)

    #return forecast_result,test 

def get_data_group1(path):
    '''
    不同品牌车辆上牌量合并再预测
    '''
    train = []
    test = []
    df = pd.read_csv(path,sep='\t')
    df_group = df.groupby('brand')
    for num,d in df_group: 
        #d = fill_nan(d) #填充缺失日期
        d['ds'] = d['date']
        d['y'] = d['cnt']
        d = d.drop(['date','day_of_week','brand','cnt'],axis=1)

        d = fill_nan(d) #填充缺失日期
        d['ds'] = pd.to_datetime(d['ds'],unit='D')
        d.fillna(method='ffill',inplace=True)
        
        #划分数据为训练数据和测试数据
        divided = datetime.datetime(1972,10,1)
        d_train = d[d.ds<divided]
        d_test = d[d.ds>= divided]
        d_test.reset_index(drop=True,inplace=True)

        test.append(d_test)
        train.append(d_train)
        

        #forecast = prophet_model(d_train,29)[-29:]
        #forecast.reset_index(drop=True,inplace=True)
        #mse = evaluate_model(forecast,d_test)
        #print '%d mse:'%num ,mse

        #forecast_result.append(forecast)
        
        #============================================
        #合计预测结果和test的mse
    train_total = train[1]
    #forecast_total = forecast_result[1]
    test_total = test[1]
    train_total.y = sum(x.y for x in train)
    forecast = prophet_model(train_total,29)[-29:]
    #m.plot_components(forecast)
    forecast.reset_index(drop=True,inplace=True)
    #forecast_total.yhat = sum(x.yhat for x in forecast_result)
    test_total.y = sum(x.y for x in test)

    mse_total = evaluate_model(forecast,test_total)
    print 'mse_total:',mse_total

def fill_nan(d,interval=[1,1032]):
    '''
    弥补日期差距
    '''
    i = 0
    while i < interval[1]:
        try:
            if i+1 == d.iloc[i].ds:
                i = i + 1
            else:
                #print '缺失',i
                above = d.iloc[:i]
                below = d.iloc[i:]
                insertRow = pd.DataFrame([[i+1,np.nan]],columns=['ds','y'])
                d = above.append(insertRow,ignore_index=True).append(below,ignore_index=True)
                #print len(d)
        except:
            insertRow = pd.DataFrame([[i+1,np.nan]],columns=['ds','y'])
            i = i+1

            d = d.append(insertRow,ignore_index=True)
    return d

def save(result):
    '''
    分开预测再合并
    '''
    start_time = datetime.datetime(1970,1,1)
    dict1 = {}
    for brand in result:
        for i in range(len(brand)):
            key = (brand.iloc[i].ds-start_time).days 
            value = brand.iloc[i].yhat 

            dict1.setdefault(key,0)
            dict1[key] += value 
    predict = sorted(dict1.items(),key=lambda item:item[0])
    #评估预测结果

    #存储结果
    with open('predict_result','w') as f_in:
        for line in predict:
            str1 = str(line[0])+'\t'+str(int(line[1])) +'\n'
            f_in.write(str1)

def save1(result):
    '''
    合并预测结果
    '''
    start_time = datetime.datetime(1970,1,1)
    with open('predict_result1','w') as f_in:
        for i in range(len(result)):
            key = (result.iloc[i].ds-start_time).days
            value = result.iloc[i].yhat
            str1 = str(key)+'\t'+str(int(value))+'\n'
            f_in.write(str1)

        
def evaluate_model(forecast,test):
    '''
    MSE  
    '''
    #print test
    #print forecast
    #确认预测和测试时间相同
    assert (test.ds == forecast.ds).all()
    mse = sum((test.y - forecast.yhat) ** 2)
    mse = mse/len(test) 
    return mse

if __name__ == '__main__':
    get_data_group('./train_20171215.txt')
    #save(result)

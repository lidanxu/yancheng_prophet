#-*- coding: utf-8 -*-
'''
尝试使用svr模型，预测盐城上牌数据
'''
import pandas as pd 
from sklearn.svm import SVR 
import numpy as np 

def run(path):

    df = pd.read_csv(path,sep='\t')
    #训练数据缺失值填充
    df_group = df.groupby('brand')
    df_list = []
    result = []
    for num,d in df_group:
        d = fill_nan(num,d)
        d.fillna(method='ffill',inplace=True)
        d.fillna(method='bfill',inplace=True)
        
        #mse = test(d)
    #train.cnt = sum(x.cnt for x in df_list)
    #=====================================
    #test(train)
        #+++++++++++++++++++++++++++++++++++++
        #predict 
        
        train_x = d[['date','day_of_week','brand']]
        train_y = d.cnt

        #构建svr模型
        svr_model = SVR(kernel='rbf',C=1e1,gamma=0.1)
        svr_model.fit(train_x[['day_of_week','brand']],train_y)
    
        #构造预测数据
        predict_x = constructed_data(train_x)[-700:]
        predict_y = svr_model.predict(predict_x[['day_of_week','brand']])
        #print 'y:',len(predict_y)
        result.append(predict_y)
        save2(result,predict_x.date)
    #print result
    #save(result)
def save2(result,date):
    dict1 = {}
    for brand in result:
        n = 0
        for i in brand:
            key =  date.iloc[n]
            value = i

            dict1.setdefault(key,0)
            dict1[key] += value
            n += 1
    predict = sorted(dict1.items(),key=lambda item:item[0])
    #评估预测结果

    #存储结果
    with open('predict_result_3','w') as f_in:
        for line in predict:
            str1 = str(line[0])+'\t'+str(int(line[1])) +'\n'
            f_in.write(str1)



def constructed_data(train_x,interval=700):
    '''
    
    '''
    predict_x = train_x
    start_date = train_x.date.max()
    start_day = train_x[train_x.date == start_date].day_of_week
    start_day = start_day.values[0]
    date = start_date 
    day = start_day 
    brand = train_x.brand.values[0]
    num = 0
    for i in range(interval):
        date = date + num
        day = (day + num)%7
        if day == 0:
            day = 7
        insertRow = pd.DataFrame([[date,day,brand]],columns=['date','day_of_week','brand'])
        predict_x = predict_x.append(insertRow,ignore_index=True)
        if num == 0:
            num =1 
    return predict_x

def test(d):
    d_train = d[:1000]
    d_test = d[1000:]
    
    train_x = d_train[['date','day_of_week','brand']]
    train_y = d_train.cnt

    svr_model = SVR(kernel = 'rbf',C=1e1,gamma=0.1)
    svr_model.fit(train_x.values,train_y.values)

    predict_x = d_test[['date','day_of_week','brand']]
    predict_y = svr_model.predict(predict_x)

    test_y = d_test.cnt
    mse = sum((test_y - predict_y) ** 2)
    mse = mse/len(test_y) 
    print 'test: mse',mse
    return mse

def fill_nan(brand,d,interval=[1,1032]):
    '''
    弥补日期差距
    '''
    i = 0
    while i < interval[1]:
        try:
            if i+1 == d.iloc[i].date:
                i = i + 1
            else:
                #print '缺失',i
                above = d.iloc[:i]
                below = d.iloc[i:]
                day_of_week = (d.iloc[i]['day_of_week'] + 1)%7 
                if day_of_week == 0:
                    day_of_week = 7
                insertRow = pd.DataFrame([[i+1,day_of_week,brand,np.nan]],columns=['date','day_of_week','brand','cnt'])
                d = above.append(insertRow,ignore_index=True).append(below,ignore_index=True)
                #print len(d)
        except:
            day_of_week = (d.iloc[i-1]['day_of_week']+1)%7 
            if day_of_week == 0:
                day_of_week = 7
            insertRow = pd.DataFrame([[i+1,day_of_week,brand,np.nan]],columns=['date','day_of_week','brand','cnt'])
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
    run('./train_20171215.txt')

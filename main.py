from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import LeaveOneOut, KFold

from LSSVMRegression import LSSVMRegression
##################数据处理#####################
def loadDataSet(fileName): 
    # 读取数据
    df = pd.read_excel(fileName)
    #显示前几行数据
    display(df.head())
    #因为时间列不作为输入，所以去掉时间列
    df = df.drop('时间', axis=1)
    #找到每一列的最大值和最小值，以便将数据全部转换为0-1之间的数字
    max_ = df.max(axis=0)
    min_ = df.min(axis=0)
    #归一化
    df = (df - min_) / (max_ - min_)
    #划分训练集和验证集，其中训练集占总数据的70%，验证集占总数据的30%
    df_train = df.sample(frac=0.7, random_state=0)
    df_valid = df.drop(df_train.index)
    #X_train 为训练集的输入，X_valid 为验证集的输入，y_train为训练集的输出， y_valid为验证集的输出
    X_train = np.array(df_train.drop(columns=['SO2原始浓度\t(mg/m3)','NOx原始浓度(mg/m3)'], axis=1))
    X_valid = np.array(df_valid.drop(columns=['SO2原始浓度\t(mg/m3)','NOx原始浓度(mg/m3)'], axis=1))
    y_train = np.array(df_train[['SO2原始浓度\t(mg/m3)','NOx原始浓度(mg/m3)']])
    y_valid = np.array(df_valid[['SO2原始浓度\t(mg/m3)','NOx原始浓度(mg/m3)']])
    return X_train, X_valid, y_train, y_valid
####################I.Neural Network Model#######################
    #建立模型并训练模型
def neuralNetwork(X_train, y_train, X_valid, y_valid):
    #输入数据的维度
    input_shape = X_train.shape[1]
    #输出数据的维度
    output_shape = y_train.shape[1]

    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[input_shape]),	

        layers.Dense(units=1024, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),	

        layers.Dense(units=1024, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),	

        layers.Dense(units=1024, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),	

        layers.Dense(units=output_shape),
        ])

    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size= 256,
        epochs=100,
        callbacks=[early_stopping],
        verbose=0,# hide the output because we have so many epochs
    )

    #训练结果
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
################II.Random Forest Model#######################
    #建立模型并训练模型
def randomForestRegressor(X_train, y_train, X_valid, y_valid): 
    rf=RandomForestRegressor()
    model = rf.fit(X_train, y_train)
    #在训练集上的拟合结果
    y_train_predict=model.predict(X_train)
    #展示在训练集上的表现
    plt.figure(1)
    drawSO2=pd.concat([pd.DataFrame(y_train[:, 0]),pd.DataFrame(y_train_predict[:, 0])],axis=1)
    drawSO2.iloc[:,0].plot(figsize=(12,6))
    drawSO2.iloc[:,1].plot(figsize=(12,6))
    plt.legend(('realSO2', 'predictSO2'),fontsize='15')
    plt.title("Train Data SO2 ",fontsize='30') #添加标题

    plt.figure(2)
    drawNOx=pd.concat([pd.DataFrame(y_train[:, 1]),pd.DataFrame(y_train_predict[:, 1])],axis=1)
    drawNOx.iloc[:,0].plot(figsize=(12,6))
    drawNOx.iloc[:,1].plot(figsize=(12,6))
    plt.legend(('realNOx', 'predictNOx'),fontsize='15')
    plt.title("Train Data NOx ",fontsize='30') #添加标题
 
    #在验证集上的预测
    y_valid_predict=model.predict(X_valid)
    #展示在验证集上的表现 
    plt.figure(3) 
    drawSO2=pd.concat([pd.DataFrame(y_valid[:, 0]),pd.DataFrame(y_valid_predict[:, 0])],axis=1);
    drawSO2.iloc[:,0].plot(figsize=(12,6))
    drawSO2.iloc[:,1].plot(figsize=(12,6))
    plt.legend(('realS02', 'predictSO2'),loc='upper right',fontsize='15')
    plt.title("Valid Data SO2",fontsize='30') #添加标 

    plt.figure(4)
    drawNOx=pd.concat([pd.DataFrame(y_valid[:, 1]),pd.DataFrame(y_valid_predict[:, 1])],axis=1);
    drawNOx.iloc[:,0].plot(figsize=(12,6))
    drawNOx.iloc[:,1].plot(figsize=(12,6))
    plt.legend(('realN0x', 'predictNOx'),loc='upper right',fontsize='15')
    plt.title("Valid Data NOx",fontsize='30') #添加标 
    
    print('训练集上的MAE/MSE')
    print(mean_absolute_error(y_train_predict, y_train))
    print(mean_squared_error(y_train_predict, y_train))
    print('验证集上的MAE/MSE')
    print(mean_absolute_error(y_valid_predict, y_valid))
    print(mean_squared_error(y_valid_predict, y_valid)) 
################III.LSSVM#######################
    #建立模型并训练模型
def LSSVMRegressor(X_train, y_train, X_valid, y_valid): 
		#rbf-model
	clfrbf=LSSVMRegression(
        gamma=1,       #set the gamma-hyper parameter equal to 1
        kernel='poly', #use the linear kernel
        sigma=1.0,
        c=1.0,
        d=3,
            )
	x = X_train
	y_0 = y_train[:,0] # SO2
	y_1 = y_train[:,1] # NOx
	# train model and predict for SO2
	clfrbf.fit(x, y_0)
	xPred=X_valid
	yExact=y_valid[:, 0]
	yPred=clfrbf.predict(xPred)
	# The mean squared error
	print('Mean squared error: %.6f' % mean_squared_error(yExact, yPred))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.6f' % r2_score(yExact, yPred))
	
	# train model and predict for NOx
	clfrbf.fit(x, y_1)
	yExact=y_valid[:, 1]
	yPred=clfrbf.predict(xPred)
	# The mean squared error
	print('Mean squared error: %.6f' % mean_squared_error(yExact, yPred))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.6f' % r2_score(yExact, yPred))

# The mean squared error
print('Mean squared error: %.6f' % mean_squared_error(yExact, yPred)
if __name__ == '__main__':
    X_train, X_valid, y_train, y_valid = loadDataSet('总数据整理.xls')
    #neuralNetwork(X_train, y_train, X_valid, y_valid)
    #randomForestRegressor(X_train, y_train, X_valid, y_valid)
	LSSVMRegressor(X_train, y_train, X_valid, y_valid)
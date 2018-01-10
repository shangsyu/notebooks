import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
#################### 模型实验 ####################
def testModel(model, test_x, test_y):
    pred_y = model.predict(test_x)
    #print('\n######################\n',model)
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    #    print('explained_variance_score =',explained_variance_score(test_y, pred_y))
    #    print('mean_absolute_error =',mean_absolute_error(test_y, pred_y))
    #    print('mean_squared_error',mean_squared_error(test_y, pred_y))
    #    print('r2_score',r2_score(test_y, pred_y))
    return(explained_variance_score(test_y, pred_y), mean_absolute_error(test_y, pred_y),
           mean_squared_error(test_y, pred_y), r2_score(test_y, pred_y))

def LR(train_x,test_x,train_y,test_y): #线性回归
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(train_x, train_y)
    #print(model.coef_,model.intercept_)
    return testModel(model, test_x, test_y)

def ridge(train_x,test_x,train_y,test_y):   # 岭回归
    from sklearn.linear_model import Ridge
    model = Ridge(alpha = 20,fit_intercept=True)
    model.fit(train_x, train_y)

    return testModel(model, test_x, test_y)

def lasso(train_x,test_x,train_y,test_y):   # Lasso
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=27.8,fit_intercept=True)
    model.fit(train_x, train_y)
    return testModel(model, test_x, test_y)

def bayes(train_x,test_x,train_y,test_y):   # 贝叶斯回归
    from sklearn.linear_model import BayesianRidge
    model = BayesianRidge()
    model.fit(train_x, train_y)
    return testModel(model, test_x, test_y)

def DT(train_x,test_x,train_y,test_y):  # 决策树回归
    from sklearn import tree
    model = tree.DecisionTreeRegressor()
    model.fit(train_x, train_y)
    return testModel(model, test_x, test_y)


def RF(train_x,test_x,train_y,test_y):  #随机森林回归
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(oob_score=True,n_estimators=88)
    model.fit(train_x, train_y)
    return testModel(model, test_x, test_y)


def GB(train_x,test_x,train_y,test_y):  # 梯度提升树回归
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)
    return testModel(model, test_x, test_y)

def explore(data_x,data_y):
    #print('\n####################')
    train_x,test_x,train_y,test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False)
    print('train_shape=',train_x.shape,'test_shape=',test_x.shape)
    LR(train_x,test_x,train_y,test_y)
    ridge(train_x,test_x,train_y,test_y)
    lasso(train_x,test_x,train_y,test_y)
    bayes(train_x,test_x,train_y,test_y)
    DT(train_x,test_x,train_y,test_y)
    RF(train_x,test_x,train_y,test_y)
    GB(train_x,test_x,train_y,test_y)

NON_X_COLS = ['时间','规模以上工业增加值（亿元）','固定资产投资额（亿元）','工业投资（亿元）','dateMonth']
Y_COLS = ['规模以上工业增加值（亿元）']
dataAll = pd.read_csv('/Users/princetechs/Downloads/hefei/data/dataAll.csv',encoding='gb18030')
dataAll = dataAll.sort_values('dateMonth')
dataAll = dataAll.dropna(subset=Y_COLS)
dataX = dataAll.drop(NON_X_COLS,1)
dataY = dataAll[Y_COLS]
data_x = dataX.values
data_y = dataY.values.reshape((1,-1))[0]

################### 计算所有特征的f_regression和mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
# 先计算所有特征的单变量描述信息并保存
featureAll = pd.DataFrame({'featureName':dataX.columns})
featureAll['f_regression']=f_regression(data_x,data_y)[0]
featureAll['mutual_info_regression']=mutual_info_regression(data_x,data_y)
featureAll.to_csv('/Users/princetechs/Downloads/hefei/data/feature_scores.csv',encoding='gb18030',index=False)

########################### 一些人为加入的筛选特征机制 #####################
# 名称中包含‘合肥’的一定保留，剩余的删掉包含‘安徽’的
featureNames=dataX.columns
featureNames = [i for i in featureNames if '合肥' in i or '安徽' not in i]
dataX = dataX[featureNames]
# make data_x and data_y
data_x = dataX.values
data_y = dataY.values.reshape((1,-1))[0]

# 使用f_regression选择
selectModel1 = SelectKBest(mutual_info_regression, k=300)
selectModel1.fit(data_x, data_y)
featureMask = selectModel1.get_support(indices=True)
featureChosen = dataX.columns[featureMask]
dataX = dataX[featureChosen]
data_x = dataX.values
selectModel2 = SelectKBest(f_regression, k=100)
selectModel2.fit(data_x, data_y)
featureMask = selectModel2.get_support(indices=True)
featureChosen = dataX.columns[featureMask]
dataX = dataX[featureChosen]
data_x = dataX.values
data_y=[np.log(i) for i in data_y]
del featureMask,featureChosen
#normalization#
# StandardScaler(copy=True, with_mean=True, with_std=True)
# scaler = StandardScaler(copy=True, with_mean=True, with_std=False)
# scaler.fit(data_x)
# data_x=scaler.transform(data_x)
#explore(data_x, data_y)
data_y=[np.sqrt(i) for i in data_y]
print('\n####################')
train_x,test_x,train_y,test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False)
print('train_shape=',train_x.shape,'test_shape=',test_x.shape)

# featureChosen=pd.DataFrame({'featureName':dataX.columns})
# model = DT(train_x,test_x,train_y,test_y)
# featureChosen['feature_importance_DT']=model.feature_importances_
# model = RF(train_x,test_x,train_y,test_y)
# featureChosen['feature_importance_RF']=model.feature_importances_
# model = GB(train_x,test_x,train_y,test_y)
# featureChosen['feature_importance_GB']=model.feature_importances_
# print('finish calculating feature_importance')

def results(modelname,train_x=train_x,test_x=test_x,train_y=train_y,test_y=test_y):
    evs,mae,mse,r2=[],[],[],[]
    for i in range(1000):
        if modelname=='LR':
            model=LR(train_x,test_x,train_y,test_y)
        if modelname=='ridge':
            model=ridge(train_x,test_x,train_y,test_y)
        if modelname=='lasso':
            model=lasso(train_x,test_x,train_y,test_y)
        if modelname=='bayes':
            model=bayes(train_x,test_x,train_y,test_y)
        if modelname=='DT':
            model=DT(train_x,test_x,train_y,test_y)
        if modelname=='RF':
            model=RF(train_x,test_x,train_y,test_y)
        if modelname=='GB':
            model=GB(train_x,test_x,train_y,test_y)
        evs.append(model[0])
        mae.append(model[1])
        mse.append(model[2])
        r2.append(model[3])

    print (modelname)
    print ('mean_evs: ',np.mean(evs),'max_evs: ',np.max(evs),'min_evs: ',np.min(evs),'most_evs: ', mode(evs)[0][0])
    print ('mean_mae: ',np.mean(mae),'max_mae: ',np.max(mae),'min_mae: ',np.min(mae),'most_mae: ', mode(mae)[0][0])
    print ('mean_mse: ',np.mean(mse),'max_mse: ',np.max(mse),'min_mse: ',np.min(mse),'most_mse: ', mode(mse)[0][0])
    print ('mean_r2: ',np.mean(r2),'max_r2: ',np.max(r2),'min_r2: ',np.min(r2),'most_r2: ', mode(r2)[0][0])

results("RF")
results("DT")
results("GB")

















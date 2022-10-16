import numpy as np
import pandas as pd
from scipy.interpolate import lagrange
from scipy.interpolate import PchipInterpolator as PCHIP
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats

def chazhi(fit_data,mode='interpolate'):
    #可选方法有：interpolate、pad、backfill
    if mode=='interpolate':
        # interpolate()插值法，缺失值前后数值的均值，但是若缺失值前后也存在缺失，则不进行计算插补。
        fit_data = fit_data.interpolate()
    if mode=='pad':
        # 用前面的值替换, 当第一行有缺失值时，该行利用向前替换无值可取，仍缺失
        fit_data=fit_data.fillna(method='pad')
    if mode=='backfill':
        # 用后面的值替换，当最后一行有缺失值时，该行利用向后替换无值可取，仍缺失
        fit_data=fit_data.fillna(method='backfill') # 用后面的值替换
    return fit_data

def newton(s, n, x_j, k=3):
    if n < k:
        y = s[list(range(0, n)) + list(range(n + 1, n + k + 1))]
    elif n > len(s) - k - 1:
        y = s[list(range(n - k, n)) + list(range(n + 1, len(s)))]
    else:
        y = s[list(range(n - k, n)) + list(range(n + 1, n + k + 1))]  # 取空值处的前后5个数
    y = y[y.notnull()]  # 剔除空值
    """
        牛顿差值多项式
        """
    """
        计算插商
        """
    x=y.index
    y=list(y)
    f0 = np.zeros((len(x), len(y)))  # 定义一个存储插商的数组
    for k in range(len(y) + 1):  # 遍历列
        for i in range(k, len(x)):  # 遍历行
            if k == 0:
                f0[i, k] = y[i]
            else:
                f0[i, k] = (f0[i, k - 1] - f0[i - 1, k - 1]) / (x[i] - x[i - 1])
    f0 = f0.diagonal()
    # 与w相乘
    f1 = 0
    for i in range(len(f0)):
        s = 1
        k = 0
        while k < i:
            s = s * (x_j - x[k])
            k += 1
        f1 = f1 + f0[i] * s
    return f1

def mul_interpolation(fit_data,k=5,mode='lagrange'):
    '''
    可用方法：牛顿插值法、分段三次埃米尔特插值法、拉格朗日插值(常用）
    k为取前后的数据个数，默认为5
    '''
    for i in fit_data.columns:  # 获取data的列名
        for j in range(len(fit_data)):
            if (fit_data[i].isnull())[j]:  # 判断data的i列第j个位置的数据是否为空，如果为空即插值
                x_j = fit_data.index[j]
                if mode=='lagrange':
                    s=fit_data[i]
                    n=j
                    if n < k:
                        y = s[list(range(0, n)) + list(range(n + 1, n + k + 1))]
                    elif n > len(s) - k - 1:
                        y = s[list(range(n - k, n)) + list(range(n + 1, len(s)))]
                    else:
                        y = s[list(range(n - k, n)) + list(range(n + 1, n + k + 1))]  # 取空值处的前后5个数
                    y = y[y.notnull()]  # 剔除空值
                    fit_data.loc[j, i]=lagrange(y.index, list(y))(j)
                elif mode=='newton':
                    fit_data.loc[j, i] = newton(fit_data[i],j,x_j,k)
                elif mode=='pchip':
                    s = fit_data[i]
                    n = j
                    if n < k:
                        y = s[list(range(0, n)) + list(range(n + 1, n + k + 1))]
                    elif n > len(s) - k - 1:
                        y = s[list(range(n - k, n)) + list(range(n + 1, len(s)))]
                    else:
                        y = s[list(range(n - k, n)) + list(range(n + 1, n + k + 1))]  # 取空值处的前后5个数
                    y = y[y.notnull()]  # 剔除空值
                    fit_data.loc[j, i]=PCHIP(y.index,list(y)).__call__(x_j)
    return fit_data

def machine_fit(fit_data,mode='random_forest',dispersed=False,k=5):
    for i in fit_data.columns:  # 获取data的列名
        if True in list(fit_data[i].isnull()):
            x_fit=[]
            for j in range(len(fit_data[i])):
                if list(fit_data[i].isnull())[j]==True:
                    x_fit.append(j)
            x_fit=np.array(x_fit)
            x=np.array([i for i in range(len(fit_data[i])) if i not in x_fit])
            y = np.array(fit_data[i][x].values).reshape(-1,1)
            x_test=x_fit.reshape(-1,1)
            x=x.reshape(-1,1)
            #X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=0) 填补先不用训练测试集划分
            #算法没调参，后续更新遗传算法参数最优实现
            if dispersed:
                if mode=='knn':
                    KNN = KNeighborsClassifier(n_neighbors=k, weights="distance")
                    KNN.fit(x, y)
                    y_fit=KNN.predict(x_test)
                    fit_data[i][x_fit]=y_fit
                elif mode=='random_forest':
                    rf = RandomForestClassifier()
                    rf.fit(x,y)
                    y_fit = rf.predict(x_test)
                    fit_data[i][x_fit] = y_fit
            else:
                if mode=='knn':
                    KNN = KNeighborsRegressor(n_neighbors=k, weights="distance")
                    KNN.fit(x, y)
                    y_fit = KNN.predict(x_test)
                    fit_data[i][x_fit] = y_fit
                elif mode=='random_forest':
                    rf = RandomForestRegressor()
                    rf.fit(x, y)
                    y_fit = rf.predict(x_test)
                    print(y_fit)
                    fit_data[i][x_fit] = y_fit
    return fit_data

def math_fill(fit_data,mode='means'):
    for i in fit_data.columns:  # 获取data的列名
        if True in list(fit_data[i].isnull()):
            if mode=='means':
                fit_data[i]=fit_data[i].fillna(fit_data[i].means())
            elif mode=='median':
                fit_data[i] = fit_data[i].fillna(fit_data[i].median())
            elif mode=='most':
                fit_data[i] = fit_data[i].fillna(stats.mode(fit_data[i])[0][0])
    return fit_data

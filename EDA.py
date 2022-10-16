import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
import missingno as msno
import os
from functions import *

class EDA(): #目前仅支持pandas的数据形式
    '''
    数据质量分析
    '''
    #缺失值
    def miss_condition(self,data,fig_savepath=os.path.abspath('.')):
        if isinstance(data,pd.DataFrame) | isinstance(data,pd.Series):
            # STEP 1:大致情况
            condition=data.describe()

            # STEP 2:缺失值处理
            # fig1,矩阵图
            fig1=plt.figure()
            msno.matrix(data, labels=True)
            fig1.savefig(f'{fig_savepath}\\fig1.jpg')
            plt.show()
            # fig2，条形图
            fig2=plt.figure()
            msno.bar(data)
            fig2.savefig(f'{fig_savepath}\\fig2.jpg')
            #缺失统计
            missing = data.isnull().sum().reset_index().rename(columns={0: 'missNum'})
            missing['missRate'] = missing['missNum'] / data.shape[0]
            if True in (missing.missRate>0).values:
                miss_sort = missing[missing.missRate > 0].sort_values(by='missRate', ascending=False)
                #fig3 for missNum
                colors = ['DeepSkyBlue', 'DeepPink', 'Yellow', 'LawnGreen', 'Aqua', 'DarkSlateGray']
                fig3 = plt.figure()
                plt.bar(np.arange(miss_sort.shape[0]), list(miss_sort.missRate.values), align='center', color=colors)
                font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 23, }
                plt.title('Histogram of missing value of variables', fontsize=20)
                plt.xlabel('variables names', font)
                plt.ylabel('missing rate', font)
                # 添加x轴标签，并旋转90度
                plt.xticks(np.arange(miss_sort.shape[0]), list(miss_sort['index']))
                plt.xticks(rotation=90)
                # 添加数值显示
                for x, y in enumerate(list(miss_sort.missRate.values)):
                    plt.text(x, y + 0.08, '{:.2%}'.format(y), ha='center', rotation=90)
                    plt.ylim([0, 1.2])
                # 保存图片
                fig3.savefig(f'{fig_savepath}\\fig3.jpg')
                plt.show()
                #fig4，热图
                fig4=plt.figure()
                msno.heatmap(data)
                fig4.savefig(f'{fig_savepath}\\fig3.jpg')
                plt.show()
                #fig5，树状图
                fig5=plt.figure()
                msno.dendrogram(data)
                fig5.savefig(f'{fig_savepath}\\fig5.jpg')
                plt.show()
            return condition,missing
        else:
            print('data format is not a Pandas supported type')
            return None

    def miss_del(self,data,datadel_mode='fill',fill_mode='lagrange',k=5,dispersed=False):
        # 缺失处理
        if datadel_mode == 'del':
            data.dropna()  # 删除空值
            return data
        elif datadel_mode == 'del':
            if fill_mode=='interpolate' | fill_mode=='pad' | fill_mode=='backfill' :
                del_data=chazhi(data,mode=fill_mode)
            elif fill_mode=='lagrange' | fill_mode=='newton' | fill_mode=='pchip':
                del_data=mul_interpolation(data,k=k,mode=fill_mode)
            elif fill_mode=='knn' | fill_mode=='random_forest' :
                del_data=machine_fit(data,mode=fill_mode,dispersed=dispersed,k=k)
            elif fill_mode=='means' | fill_mode=='median' | fill_mode=='most':
                del_data=math_fill(data,mode=fill_mode)
            return del_data

    # 重复值
    def dup(self,data):
        duplicate = data.duplicated().any()
        if duplicate == True:
            data = data.drop_duplicates()
            return data

    #异常值
    def abnormal(self,data,fig_savepath=os.path.abspath('.')):
        for i in data.columns:
            #箱型图
            fig=plt.figure()
            data.loc[:,[i]].boxplot()
            plt.title(f'{i} box plot ')
            fig.savefig(f'{fig_savepath}\\{i}_box_fig.jpg')
            plt.show()
            #小提琴图
            fig2=plt.figure()
            sns.violinplot(np.log(data[i]))
            plt.title(f'{i} violint plot ')
            fig2.savefig(f'{fig_savepath}\\{i}_violint_fig.jpg')
            plt.show()

    #一致性

    '''
    数据特征分析
    '''
    #总体规模,二者关系图、对比关系描述
    def relation(self,data,fig_savepath=os.path.abspath('.')):
        col_list=list(data.columns)
        for i in range(len(col_list)):
            for j in range(i+1,len(col_list)):
                f, axes = plt.subplots(2, 1,sharex=True)
                sns.lineplot(x=data[col_list[i]], y=data[col_list[j]], ax=axes[0]).set_title(f"relation between {i} and {j}")
                sns.barplot(x=data[col_list[i]], y=data[col_list[j]], ax=axes[1], palette='rocket').set_title(f"relation between {i} and {j}")
                f.savefig(f"{fig_savepath}\\relation between {i} and {j}.png")
                plt.show()

    #分布分析
    def distribution(self,data,fig_savepath=os.path.abspath('.')):
        for i in data.columns:
            fig=plt.figure()
            plt.subplot(1,3,1)
            plt.hist(data[i], histtype='stepfilled', color='steelblue', edgecolor='none')
            plt.title(f'{i}')
            plt.subplot(1,3,2)
            sns.kdeplot(data[i], shade=True)  # KDE
            plt.title(f'{i}')
            plt.subplot(1,3,3)
            sns.distplot(data[i])  # KDE
            plt.title(f'{i}')
            plt.subplots_adjust(wspace=0.8)
            fig.savefig(f'{fig_savepath}\\distribution_{i}_fig.jpg')
            plt.show()

            skew=data[i].skew() #偏度
            kurt=data[i].kurt() #峰态
            print('偏度', skew)
            print('峰态', kurt)

            sns.set_style('darkgrid')
            fig2=plt.figure()
            sns.distplot(data[i], fit=norm)
            (mu, sigma) = norm.fit(data[i])
            print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
            plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
            plt.ylabel('Frequency')
            fig2.savefig(f'{fig_savepath}\\fenbu_{i}_fig.jpg')
            plt.show()

    #统计分析
    def static(self,data,mode='pearson',fig_savepath=os.path.abspath('.')):
        mean = data.mean()
        max = data.max()
        min = data.min()
        median = data.median()
        zhongshu = data.mode().iloc[0]
        jicha = max - min
        var = data.var()
        std = data.std()
        bianyixishu = std / mean
        list_view = data.describe()
        list_view.loc['median'] = median
        list_view.loc['most'] = zhongshu
        list_view.loc['jicha'] = jicha
        list_view.loc['var'] = var
        list_view.loc['bianyi'] = bianyixishu

        sns.pairplot(data)
        plt.savefig(f'{fig_savepath}//point_matrix.jpg')
        plt.show()

        '''
        连续数据，正态分布，线性关系，用pearson相关系数是最恰当，当然用spearman相关系数也可以，效率没有pearson相关系数高。
        上述任一条件不满足，就用spearman相关系数，不能用pearson相关系数。
        两个定序测量数据（顺序变量）之间也用spearman相关系数，不能用pearson相关系数。
        pearson相关系数的一个明显缺陷是，作为特征排序机制，他只对线性关系敏感。如果关系是非线性的，即便两个变量具有一一对应的关系
        pearson相关性也可能会接近0。
        
        pearson、spearman、kendall，pointbiserialr
        计算积距pearson相关系数，连续性变量才可采用;
        计算Spearman秩相关系数，适合于定序变量或不满足正态分布假设的等间隔数据; 
        计算Kendall秩相关系数，适合于定序变量或不满足正态分布假设的等间隔数据。
        '''
        similar=data.corr(method=mode)

        return list_view,similar

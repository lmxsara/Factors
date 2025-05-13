import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

def get_data(code, startdate, enddate):  # 取数据
    token = 'e44d0c1b455757bd12b08bd16256539e784854bbeb6deae4349f9c03'
    pro = ts.pro_api(token)
    df = pro.daily(ts_code=code, start_date=startdate, end_date=enddate)
    return df

class FactorAnalyzer:
    def __init__(self, df):
        """初始化因子分析器"""
        self.df = df.copy()
        self.factor_names = []
        self.returns_col = 'next_return'

    def cal_liquidity(self): #流动性因子
        self.df['return'] = (self.df['close'] - self.df['open'])/self.df['open']
        # 处理除零风险
        min_return = 0.0001  # 设置最小回报率阈值
        self.df['abs_return'] = np.maximum(np.abs(self.df['return']), min_return)
        self.df['factor_liquidity'] = self.df['vol']/self.df['abs_return']
        factor_name = 'factor_liquidity'
        self.factor_names.append(factor_name)  # 添加到因子列表
        return self.df

    def cal_amplitude(self): #振幅因子
        self.df['factor_amplitude'] = (self.df['high'] - self.df['low'])/self.df['close']
        # 标准化因子
        factor_name = 'factor_amplitude'
        self.factor_names.append(factor_name)  # 添加到因子列表
        return self.df

    def vw_price_strength(self): #相对价格强度因子
        # 计算相对价格强度 (Relative Price Strength)
        self.df['price_strength'] = (self.df['close'] - self.df['low']) / (self.df['high'] - self.df['low'])

        # 处理价格区间为0的情况（避免除零错误）
        self.df.loc[self.df['high'] == self.df['low'], 'price_strength'] = 0.5  # 中间值表示无方向

        # 计算成交量加权的相对价格强度因子
        self.df['factor_vw_price_strength'] = self.df['price_strength'] * self.df['vol']

        # 标准化因子
        factor_name = 'factor_vw_price_strength'
        self.factor_names.append(factor_name)  # 添加到因子列表

        # 计算下一期收益率（假设使用未来一天的收盘价计算）
        self.df['next_close'] = self.df['close'].shift(-1)
        self.df[self.returns_col] = self.df['next_close'] / self.df['close'] - 1
        return self.df

    def calculate_factor_correlation(self):
        # 确保factor_names不为空
        if not self.factor_names:
            print("请先计算因子!")
            return None

        # 计算因子间的相关系数矩阵
        corr_matrix = self.df[self.factor_names].corr()

        # 可视化相关性矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
        plt.title('因子相关性矩阵')
        plt.tight_layout()
        plt.savefig('factor_correlation.png')
        plt.close()
        return corr_matrix

    def calculate_ic(self, window=60):
        ic_results = {}
        for factor in self.factor_names:
            # 计算滚动秩相关系数
            rolling_ic = []
            for i in range(window, len(self.df)):
                subset = self.df.iloc[i - window:i]
                if len(subset) < 10:  # 最小样本要求
                    rolling_ic.append(np.nan)
                    continue
                ic = spearmanr(subset[factor], subset[self.returns_col])[0]
                rolling_ic.append(ic)

            self.df[f'{factor}_rolling_ic'] = pd.Series(rolling_ic, index=self.df.index[window:])

            ic_results[factor] = {
                'mean_ic': np.nanmean(rolling_ic),
                'ic_ir': np.nanmean(rolling_ic) / (np.nanstd(rolling_ic) + 1e-8),
                'positive_ratio': np.mean(np.array(rolling_ic) > 0)
            }

            # 画图
            plt.figure(figsize=(12, 4))
            self.df[f'{factor}_rolling_ic'].plot(title=f'{factor} {window}_IC')
            plt.axhline(0, c='r', ls='--')
            plt.show()

        return ic_results

if __name__ == '__main__':
    #贵州茅台，2015-2025
    #ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
    df = get_data('600519.SH','20150101','20250101')

    # 初始化因子分析器
    analyzer = FactorAnalyzer(df)
    # 计算因子
    analyzer.cal_liquidity()
    analyzer.cal_amplitude()
    analyzer.vw_price_strength()

    # 计算因子相关性
    corr_matrix = analyzer.calculate_factor_correlation()
    print("\n因子相关性矩阵:")
    print(corr_matrix)

    # 计算因子IC
    ic_results = analyzer.calculate_ic()
    print("\nic:",ic_results)

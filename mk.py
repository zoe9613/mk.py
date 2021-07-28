
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mktest(inputdata):
    '''
    曼-肯德尔(Mann-Kendall)检验
    :param inputdata:输入数据，一维序列
    :return: UF,UB
    '''
    inputdata = np.array(inputdata)
    n=inputdata.shape[0]
    Sk             = [0]
    UFk            = [0]
    s              =  0
    Exp_value      = [0]
    Var_value      = [0]
    for i in range(1,n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s+1
            else:
                s = s+0
        Sk.append(s)
        Exp_value.append((i+1)*(i+2)/4 )
        Var_value.append((i+1)*i*(2*(i+1)+5)/72 )
        UFk.append((Sk[i]-Exp_value[i])/np.sqrt(Var_value[i]))
    Sk2             = [0]
    UBk             = [0]
    UBk2            = [0]
    s2              =  0
    Exp_value2      = [0]
    Var_value2      = [0]
    inputdataT = list(reversed(inputdata))
    for i in range(1,n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s2 = s2+1
            else:
                s2 = s2+0
        Sk2.append(s2)
        Exp_value2.append((i+1)*(i+2)/4)
        Var_value2.append((i+1)*i*(2*(i+1)+5)/72)
        UBk.append((Sk2[i]-Exp_value2[i])/np.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])
    UBkT = list(reversed(UBk2))
    return UFk, UBkT


if __name__=="__main__":
    filename = r"E:\project\safe ocean\cd\SC_frequency_monthseasonyear.csv"
    data_csv = pd.read_csv(filename)
    y = np.array(data_csv['frequency'])

    uf, uk = mktest(y)

    fig = plt.figure(figsize=(15, 15), dpi=300)
    f_ax1 = fig.add_subplot(323)
    f_ax1.plot(np.arange(1949, 2021, 1), y, 'k')

    f_ax2 = fig.add_subplot(324)
    f_ax2.plot(np.arange(1949, 2021, 1), uf, 'b', label='UF')
    f_ax2.plot(np.arange(1949, 2021, 1), uk, 'r', label='UK')
    f_ax2.set_xlim(1949, 2021)
    f_ax2.set_ylim(-4, 4)
    # 0.01显著性检验
    f_ax2.axhline(1.96)
    f_ax2.axhline(-1.96)
    plt.show()
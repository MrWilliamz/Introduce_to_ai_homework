import dashed as dashed
import numpy as np
import matplotlib.pyplot as plt

x = np.mat([[1.0,1.2,0.3],[1.0,-2.5,1.1],[1.0,3.4,2.7],[1.0,2.9,6.3]])
y = np.mat([[1.5],[-0.4],[2.1],[-1.6]])

para_x = x.T.dot(x)
#print(para_x)
beta = np.linalg.inv(para_x)*x.T*y

input_X = np.mat([[1,1,1]])

result = beta.T * input_X.T

#print(result)

def caly(xk,wk):   #计算y的值
    y=np.linalg.norm(xk.dot(wk))
    return y

def calMSE(xk,dk):#求MSE的函数
    #num=xk.shape[0]
    delt = dk-d0
    return np.linalg.norm(delt)**2

def BGD(X0,d0,eta,w0,Pmax):
    wk = w0
    MSE=[]

    for i in range(Pmax):
        P = d0.T.dot(X0).T / (X0.shape[0])
        # print(P)
        R = X0.T.dot(X0) / (X0.shape[0])
        dk = X0.dot(wk)
        wk = wk-eta*(-P+R.dot(wk))
        MSE.append(calMSE(X0,dk))
    return wk , MSE

def BGD_1(x0,d0,Pmax):
    X = np.column_stack([np.ones(x0.shape[0]), x0[:, 0], x0[:, 1]])
    X_para = np.array([[1, 1, 1]])
    w = np.mat([[0.0], [0.0], [0.0]])
    wk,MSE = BGD(X, d0, 0.01, w, Pmax)
    y_para = caly(X_para,wk)
    return wk , MSE, y_para


def BGD_2(x0,d0,Pmax):
    X = np.column_stack([np.ones(x0.shape[0]), x0[:, 0], x0[:, 1], x0[:, 0] * x0[:, 1], x0[:, 0] ** 2, x0[:, 1] ** 2])
    X_pre = np.array([[1,1,1,1,1,1]])
    w = np.mat(([0.0],[0.0],[0.0],[0.0],[0.0],[0.0]))
    wk,MSE = BGD(X, d0, 0.001, w, Pmax)
    y_pre = caly(X_pre,wk)
    return wk, MSE,y_pre

def LSE2(x0,d0):
    X = np.column_stack([np.ones(x0.shape[0]), x0[:, 0], x0[:, 1],x0[:, 0] * x0[:, 1], x0[:, 0] ** 2, x0[:, 1] ** 2])
    #print(X)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(d0)
    y_predicted = beta[0] + beta[1] + beta[2]+ beta[4] + beta[5] + beta[3]
    return  np.linalg.norm(y_predicted),beta

d0 = np.mat([[1.5],[-0.4],[2.1],[-1.6]])
x0 = np.array([[1.2,0.3],[-2.5,1.1],[3.4,2.7],[2.9,6.3]])

#Method:BGD wk,MSE,y_predict
wk1_100,MSE1_100,y_BGD1_100 = BGD_1(x0,d0,100)
wk1_1000, MSE1_1000, y_BGD1_1000 = BGD_1(x0, d0, 1000)
wk1_10000, MSE1_10000, y_BGD1_10000 = BGD_1(x0, d0, 10000)
wk2_100, MSE2_100, y_BGD2_100 = BGD_2(x0, d0, 100)
wk2_1000, MSE2_1000, y_BGD2_1000 = BGD_2(x0, d0, 1000)
wk2_10000, MSE2_10000, y_BGD2_10000 = BGD_2(x0, d0, 10000)
y_LSE2 , beta = LSE2(x0, d0)
print(y_LSE2)


def average(list1):
    total = 0
    for i in list1:
        total += i
    return  total/len(list1)

print(average(MSE2_100))
print(y_BGD2_100)

#Show the Curves of MSE
x_array_100 = np.array(range(0,100))
x_array_1000 = np.array(range(0, 1000))
x_array_10000 = np.array(range(0, 10000))
# plt1_100=plt.plot(x_array_100,MSE1_100,'r',label='PMAX = 100')
# plt2_100=plt.plot(x_array_100,MSE2_100,'k',label='MSE1_100')
# plt1_1000 = plt.plot(x_array_1000, MSE1_1000, 'b', label='PMAX = 1000')
# plt2_1000 = plt.plot(x_array_1000, MSE2_1000, 'k', label='MSE1_1000')
#  = plt.plot(x_array_10000, MSE1_10000, 'm', label='PMAX=10000')
# plt2_10000 = plt.plot(x_array_10000, MSE2_10000, 'k', label='MSE1_10000')
plt.xlabel('number of iterations')
plt.ylabel('MSE')
plt.legend()
# plt.show()
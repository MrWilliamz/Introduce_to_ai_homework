import numpy as np



a = np.mat([[1,1,0,1,0,0],[1,2,1,4,1,2],[1,3,2,9,4,6],[1,2,6,4,36,12]])
y = [[1],[0],[2],[1]]
z = []

x = a.T*a
# w = np.linalg.pinv(x)
# print(w)


beta = a.T.dot(np.linalg.inv(a.dot(a.T))).dot(y)
print(beta)
x_input = np.mat([1,1,1,1,1,1])
result = x_input*beta
print(result)

x0=np.array([[1.2,0.3],[-2.5,1.1],[3.4,2.7],[2.9,6.3]])
X = np.column_stack([np.ones(x0.shape[0]), x0[:, 0], x0[:, 1]])
print(X)


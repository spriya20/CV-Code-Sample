import numpy as np
from numpy import linalg as la
# import scipy.linalg as scla
class Camera:
    def __init__(self,K):
        self.K = K
        self.fx = K[0,0]
        self.fy = K[1,1]
        self.cx = K[0,2]
        self.cy = K[1,2]

    def updateK(self,dfx,dfy,dcx,dcy):
        self.fx += dfx
        self.fy += dfy
        self.cx += dcx
        self.cy += dcy
        K=np.zeros((3,3))
        K[0,0]=self.fx
        K[1,1]=self.fy
        K[0,2]=self.cx
        K[1,2]=self.cy
        K[2,2]=1
        self.K = K

def printResult(K,RT):
    for i in range(3):
        print(' '.join(map(str, K[i])))
    print()
    for i in range(3):
        print(' '.join(map(str, RT[i])))
    print()
    print()
    return

def calculateResidual(P,Y,K):
    residual = []
    for i in range(n):
        p = P[i]
        y = Y[i]
        proj = K@y
        proj = proj/proj[2]
        r = proj[:-1] - p
        residual.append(r)
    residual = np.array(residual).flatten()
    return residual

def calculateJacobian(C,Y):
    J = []
    for i in range(n):
        Jrow = getJacobianRows(C,Y[i])
        J.append(Jrow)
    J = np.array(J).reshape(16,-1)
    return J

def getJacobianRows(C,y):
    fx = C.fx
    fy = C.fy
    gx = y[0]
    gy = y[1]
    gz = y[2]
    Jrow = np.array([[gx/gz,0,1,0,fx/gz,0,-(fx*gx)/(gz**2),-(fx*gx*gy)/(gz**2),fx*(1+(gx**2)/(gz**2)),-(fx*gy)/gz],
                    [0,gy/gz,0,1,0,fy/gz,-(fy*gy)/(gz**2),-fy*(1+(gy**2)/(gz**2)),(fy*gx*gy)/(gz**2),(fy*gx)/gz]])
    return Jrow

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def expW(w):
    w_ = skew(w)
    theta = la.norm(w)
    expW = np.eye(3)
    if theta != 0:
        expW += (np.sin(theta)/theta)*w_ + ((1-np.cos(theta))/(theta**2))*(w_**2)
    else:
        expW += w_ + 0.5*(w_**2)
    return expW

def getV(w):
    w_ = skew(w)
    theta = la.norm(w)
    V = np.eye(3) 
    if theta !=0:
        V+= ((1-np.cos(theta))/(theta**2))*w_ + ((theta-np.sin(theta))/(theta**3))*(w_**2)
    else:
        V+= 0.5*w_ + (1/6)*(w_**2)
    return V

def expDelta(delta):
    delMat = np.eye(4)
    w = delta[3:]
    t = delta[:3]
    delMat[:-1,:-1] = expW(w)
    delMat[:-1,-1] = getV(w)@t
    return delMat

def pseudoexp(delta):
    delMat = np.eye(4)
    w = delta[3:]
    t = delta[:3]
    delMat[:-1,:-1] = expW(w)
    delMat[:-1,-1] = t
    return delMat

X = np.array([[0,0,0],
     [0,0,1],
     [0,1,0],
     [0,1,1],
     [1,0,0],
     [1,0,1],
     [1,1,0],
     [1,1,1]])
X = np.insert(X,3,1,axis=1)
n = 8
iters = 10
P = []
for i in range(n):
    row = input().split()
    row = [float(e) for e in row]
    P.append(row)
P = np.array(P)
K = []
for i in range(3):
    row = input().split()
    if row ==[]:
        row = input().split()
    row = [float(e) for e in row]
    K.append(row)
K = np.array(K)
C = Camera(K)
RT = []
for i in range(3):
    row = input().split()
    if row==[]:
        row = input().split()
    row = [float(e) for e in row]
    RT.append(row)
RT = np.array(RT)
for i in range(iters):
    Y = X@RT.T
    J = calculateJacobian(C,Y)
    res = calculateResidual(P,Y,K)
    delta = -la.pinv(J)@res
    C.updateK(*delta[:4])
    K = C.K
    delMat = expDelta(delta[4:])
    RT_ = np.insert(RT,3,0,axis=0)
    RT_[-1,-1] = 1
    RT_ = delMat@RT_
    RT = RT_[:-1,:]
    printResult(K,RT)


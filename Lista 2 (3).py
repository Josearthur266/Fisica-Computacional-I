#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import*
from math import*
import matplotlib.pyplot as plt


# ## Problema 1

# In[2]:


def f(t):
    return e**(-t**2)


# In[3]:


m = 100
xo, xm = 0, 3
h = (xm-xo)/m
x = []
i=0
while(i<=xm):
    x.append(i)
    i+=h
    
p1 = f(xo) + f(xm)
i, temp = 1, 0

while(i<=m/2-1):
    temp = f(x[2*i]) + temp
    i+=1
p2 = 2*temp

i, temp = 1, 0
while(i<=m/2):
    temp = f(x[2*i-1]) + temp
    i+=1
p3 = 4*temp

res = (p1 + p2 + p3)*h/3
res


# In[4]:


x = arange(0,3.1,0.1)
len(x)


# In[5]:


E = [0]
p = 0
for i in range(1,31):
    xo, xm = x[i-1], x[i]
    c = (x[i]+x[i-1])/2
    h = (xm - xo)/2
    p = p + (f(xo) + 4*f(c) + f(xm))*h/3
    E.append(p)
plt.plot(x,E)


# ## Problema 2

# In[6]:


#a)
def f(tet, m, x):
    return (cos(m*tet-x*sin(tet)))/pi

def j(m,x):
    n = 1000
    tet0, tetn = 0, pi
    h = (tetn-tet0)/n
    tet = []
    i = 0
    while(i<=tetn):
        tet.append(i)
        i+=h
        
    p1 = f(tet0, m, x) + f(tetn, m, x)
    i , temp = 1, 0
    
    while(i<=n/2-1):
        temp = f(tet[2*i], m, x) + temp
        i+=1
    p2 = 2*temp
    
    i, temp = 1, 0
    while(i<=n/2):
        temp = f(tet[2*i-1], m, x) + temp
        i+=1
    p3 = 4*temp
    
    return (p1 + p2 + p3)*h/3


# In[7]:


x = arange(0, 20.05, 0.25)
j0 = []
j1 = []
j2 = []
for i in range(0,81):
    temp = j(0,x[i])
    j0.append(temp)
    
    temp = j(1,x[i])
    j1.append(temp)
    
    temp = j(2,x[i])
    j2.append(temp)
j0, j1, j2


# In[8]:


plt.plot(x,j0)
plt.plot(x,j1)
plt.plot(x,j2)
plt.xlabel("X")
plt.ylabel("Jm(x)")


# In[9]:


import numpy as np


# In[10]:


#b)
lan = 500e-9
k = 2*pi/lan
r = []
linha0 = []
x = np.linspace(-1e-6,1e-6,100)
y = np.linspace(-1e-6,1e-6,100)

for i in range(0,100):
    for q in range(0, 100):
        temp = (x[i]**2+y[q]**2)**0.5
        linha0.append(temp)
    r.append(linha0)
    linha0=[]
I = []
linha = []
for i in range(0,100):
    for q in range(0,100):
        temp = (k*r[i][q])
        linha.append((j(1,temp)/temp)**2)
    I.append(linha)
    linha = []


# In[11]:


plt.imshow(I, origin='lower', vmax=0.01)
plt.hot()


# ## Problema 3

# In[12]:


#Definindo as funções para a integral
def inte(x):
    return x**4 - 2*x + 1
def trap(N):
    a, b = 0, 2
    h = (b-a)/N
    res = (inte(a)+inte(b))/2
    for i in range(1,N):
        res = res + inte(a+i*h)
    return res*h
I1, I2 = trap(10), trap(20)
I1, I2


# In[13]:


#Calculando o erro
err = (I2-I1)/3
err, I2-4.4


# Esse erro é calculado apartir da série de Taylor expandida, porém ele só considera os 3 priemeiros termos, dessa forma esse erro não será tão preciso. 

# ## Problema 4

# In[14]:


def fun(x):
    return (sin((100*x)**0.5))**2

def p1(N):
    a,b = 0, 1
    h0 = (b-a)/(N/2)
    p1 = 0.5*(fun(a)+fun(b))
    if(N>2):
        for i in range(1,int(N/2)):
            p1 = p1 + fun(i*h0)
    else:
        p1=p1
    return p1*h0

def p2(N):
    a,b = 0, 1
    h = (b-a)/N
    p2 = 0
    for i in range(1,N,2):
        p2 = p2 + fun(i*h)
        
    return p2*h


# In[15]:


N = 1
err = 1
while(err>1e-6):
    I0 = p1(N)
    I1 = I0/2 + p2(N)
    err = abs((I1-I0)/3)
    print(I1, err, N)
    N*=2


# A quantidade de 4096 fatias para chegar no resultado correto.

# In[ ]:





import numpy as np

a = np.array([[1,2,3,4],[4,5,6, 7],[7,8,9,10],[10,11,12,13]])
f = np.array([[.5,.5,.5], [.5,.5,.5], [.5,.5,.5]])

padding = f.shape[0]-1
a_dim = a.shape[0]

c= np.zeros((a_dim+(padding)*2, a_dim+(padding)*2)) #add the filter -1 *2 <-- That's be


print(c[f.shape[0]-1:f.shape[0]-1+a.shape[0],f.shape[0]-1:f.shape[0]-1+a.shape[0]])


c[padding:padding+a_dim,padding:padding+a_dim] = a
print(c)

import numpy as np

input = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
print(input)

i1 = 0
i2 = 3
j1 = 0
j2 = 3
#print(input[i1:i2,j1:j2])
i1 +=1
i2 +=1
j1 +=0
j2 +=1
#print(input[i1:i2,j1:j2])
#print(input[0:3,1:4])
print(" ")

i1 = 0
i2 = 3
j1 = 0
j2 = 3
for i in range(9):
    if j2 == 6:
        j2=3
        j1=0
        i1+=1
        i2+=1

    print(input[i1:i2,j1:j2])
    print(" ")
    j2+=1
    j1+=1
#print(input[1:3,0:3])

a = np.array([[1,2,3], [4,5,6]])
b=np.reshape(a,(6))
print(b)

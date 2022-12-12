import numpy
import matplotlib.pyplot as plt
from config import *

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

p = numpy.polyfit(x, y, 3)
#mymodel = numpy.poly1d(p)
mymodel = numpy.poly

print("\np:\n", p)
print("\npoly thingy:\n", mymodel)
print("\nend print!\n")

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
fileToSave = f".{IMAGE_FOLDER}/poly_classes.png"
plt.savefig(fileToSave)

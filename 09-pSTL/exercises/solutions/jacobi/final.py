# coding: utf-8
import matplotlib.pyplot as plt
import numpy
data = numpy.loadtxt("final.dat")
plt.imshow(data)
plt.colorbar()
plt.savefig("final.png")

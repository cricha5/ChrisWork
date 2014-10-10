import EuropeanOption

import time
import numpy

import matplotlib.pyplot as plt

import pyopencl as cl
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
#os.environ['PYOPENCL_CTX'] = '0'

platforms = cl.get_platforms()
for platform in platforms:
    print("===============================================================")
    print("Platform name:", platform.name)
    print("Platform profile:", platform.profile)
    print("Platform vendor:", platform.vendor)
    print("Platform version:", platform.version)
    for device in platform.get_devices():
        print("---------------------------------------------------------------")
        print("Device name:", device.name)
        print("Device type:", cl.device_type.to_string(device.type))
        print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
        print("Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("Device compute units:", device.max_compute_units)
        print("Device max work group size:", device.max_work_group_size)
        print("Device max work item sizes:", device.max_work_item_sizes)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

PTYPE = 1
S_init = 100.0
S_max = 300.0
Strike = 100.0
Exp_Time = 1.0
Int_Rate = 0.05
Vol = 0.2

tBegin = time.time()
value = EuropeanOption.valueBlackScholes(S_init,Strike,Exp_Time,Int_Rate,Vol,PTYPE)
tEnd = time.time()

print("Black Scholes value = ",value)
print("Black Scholes compute time = ",tEnd - tBegin)

nAssetSteps = 1*2**9
tBegin = time.time()
value = EuropeanOption.valueFiniteDifferenceGPU(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE)
tEnd = time.time()

print("Finite Difference value and dS = ",value)
print("Finite Difference compute time = ",tEnd - tBegin)

nAssetSteps = 1*2**9
dt = 0.8 / Vol**2 / nAssetSteps**2
print("dt = ", dt)
tBegin = time.time()
value = EuropeanOption.valueFiniteDifferenceNUnderlyingsGPUOptimized(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE,1)
tEnd = time.time()

print("Finite Difference optimized value and dS = ",value)
print("Finite Difference optimized compute time = ",tEnd - tBegin)

nPaths = 2*(2**9)
dtMonte = 1/250
nMonteLoops = 2**8

tBegin = time.time()
value = EuropeanOption.valueMonteCarlo2GPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE)
tEnd = time.time()

value = EuropeanOption.valueMonteCarloGPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE, nMonteLoops)

print("Monte Carlo value and dS = ",value)
print("Monte Carlo compute time = ",tEnd - tBegin)

#Serrors = numpy.zeros(64)
#Saxis = numpy.zeros_like(Serrors)
#for n in range(64):
#    nPaths = (n+1)*2**9
#    Saxis[n] = nPaths
#    value, tError, Serrors[n] = EuropeanOption.valueMonteCarloGPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE, nMonteLoops)

#plt.figure()
#plt.plot(Saxis,Serrors)
#plt.show()






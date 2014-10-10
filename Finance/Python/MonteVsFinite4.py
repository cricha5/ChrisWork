import EuropeanOption

import time
import numpy

import matplotlib.pyplot as plt

import pyopencl as cl
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
os.environ['PYOPENCL_CTX'] = '1'

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
S_max = 200.0
Strike = 100.0

Exp_Time = 1.0
Int_Rate = 0.05
Vol = 0.2


dtMonte = 0.1

setSize = 2**4
Nsteps = 16
timingValuesFD = numpy.zeros((Nsteps,setSize))
timingValuesMC = numpy.zeros((Nsteps,setSize))
timingValuesFDO = numpy.zeros((Nsteps,setSize))
timingMeans = numpy.zeros((3,Nsteps))
timingXAxis = numpy.zeros((Nsteps))


#nAssetSteps = 2**3
#sizeWI = 64
sizeStep = 64
#nPaths = (1**1)*(2**0)*(2**6)

for t in range(Nsteps):
    timingXAxis[t] = (t+1)*sizeStep

for t in range(Nsteps):
    nAssetStepsT = (t+1)*sizeStep
    nPathsT = (t+1)*sizeStep
    for set in range(setSize):
        #print("before fd", t, nAssetStepsT, nPathsT)
        tBegin = time.time()
        value = EuropeanOption.valueFiniteDifferenceGPU(ctx,queue,S_init,S_max,nAssetStepsT,Strike,Exp_Time,Int_Rate,Vol,PTYPE)
        tEnd = time.time()
        
        timingValuesFD[t,set] = tEnd - tBegin
        
        tBegin = time.time()
        value = EuropeanOption.valueFiniteDifferenceGPUOptimized(ctx,queue,S_init,S_max,nAssetStepsT,Strike,Exp_Time,Int_Rate,Vol,PTYPE)
        tEnd = time.time()
        
        timingValuesFDO[t,set] = tEnd - tBegin
        
        #print("after fd", t)
        #print("Finite Difference compute time = ",tEnd - tBegin)
        
        tBegin = time.time()
        value = EuropeanOption.valueMonteCarloGPU(ctx,queue,S_init,nPathsT,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE)
        tEnd = time.time()
        #print("Monte Carlo compute time = ",tEnd - tBegin)
        
        timingValuesMC[t,set] = tEnd - tBegin
    
    timingMeans[0,t] = numpy.mean(timingValuesFD[t])
    timingMeans[1,t] = numpy.mean(timingValuesFDO[t])
    timingMeans[2,t] = numpy.mean(timingValuesMC[t])

print(timingMeans)
print(timingXAxis)
plt.figure()
plt.yscale('log')
plt.plot(timingXAxis,timingMeans[0], '-b', label='FD')
plt.plot(timingXAxis,timingMeans[1], '-r', label='FDO')
plt.plot(timingXAxis,timingMeans[2], '-g', label='MC')
plt.legend(loc='upper right')
plt.show()




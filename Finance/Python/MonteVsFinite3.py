import EuropeanOption2

import time
import numpy
import math

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


dtMonte = 0.01

setSize = 2**6
nAssets = 0*1
timingValuesFD = numpy.zeros((nAssets,setSize))
timingValuesMC = numpy.zeros((nAssets,setSize))
timingMeans = numpy.zeros((2,nAssets))
timingXAxis = numpy.zeros((nAssets))

nAssetSteps = 2**4
nPaths = 2*(2**9)

for t in range(nAssets):
    #nAssetStepsT = nAssetSteps**(t+1)
    nPathsT = nPaths*(t+1)
    for set in range(setSize):
        tBegin = time.time()
        value = EuropeanOption2.valueFiniteDifferenceGPU(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE,(t+1))
        tEnd = time.time()
        
        timingValuesFD[t,set] = tEnd - tBegin

        tBegin = time.time()
        value = EuropeanOption2.valueMonteCarloGPU(ctx,queue,S_init,nPathsT,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE)
        tEnd = time.time()
        
        timingValuesMC[t,set] = tEnd - tBegin
    
    timingMeans[0,t] = numpy.mean(timingValuesFD[t])
    timingMeans[1,t] = numpy.mean(timingValuesMC[t])
    timingXAxis[t] = t+1


plt.figure()
plt.yscale('log')
plt.ylabel('Time (s)')
plt.xlabel('Number of Dimensions')
plt.xticks(numpy.arange(0, 4, 1))
plt.plot(timingXAxis,timingMeans[0], '-b', label='Finite Difference')
plt.plot(timingXAxis,timingMeans[1], '-g', label='Monte Carlo')
plt.legend(loc='lower right')
#plt.show()

nMonteLoops = 2**7
nRange = 1*2**5
Serrors = numpy.zeros(nRange)
Sfit1 = numpy.zeros(nRange)
Sfit2 = numpy.zeros(nRange)
Saxis = numpy.zeros_like(Serrors)
for n in range(nRange):
    nPaths = (n+1)*2**9
    Saxis[n] = nPaths
    value, tError, Serrors[n] = EuropeanOption2.valueMonteCarloGPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE, nMonteLoops)
    #Sfit1[n] = 1/ (nPaths * dtMonte)
    Sfit2[n] = 1/ (math.sqrt(nPaths * 0.5 * dtMonte))

plt.figure()
plt.ylabel('Standard Deviation of Option Value')
plt.xlabel('Number of Paths')
plt.plot(Saxis,Serrors,'ro')
#plt.plot(Saxis,Sfit1)
plt.plot(Saxis,Sfit2)
plt.show()

nRange = 0*9
Serrors = numpy.zeros(nRange)
Saxis = numpy.zeros_like(Serrors)
for n in range(nRange):
    nAssetSteps = 2**(n+1)
    Saxis[n] = nAssetSteps
    #value, tError, Serrors[n] = EuropeanOption2.valueFiniteDifferenceGPU(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE)
    Serrors[n] = S_max/nAssetSteps

plt.figure()
plt.ylabel('Discretization Error of Option Value')
plt.xlabel('Grid Size')
plt.plot(Saxis,Serrors)
plt.show()


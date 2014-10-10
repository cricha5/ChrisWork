import numpy
import math
import time
import pyopencl as cl
import pyopencl.array as clarray
import matplotlib.pyplot as plt
import random

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
os.environ['PYOPENCL_CTX'] = '1'

finiteDifferenceNextStep_source = """
    __kernel void finiteDifferenceNextStep(__global float *fd1,__global float *fd2, const uint timeStep, const uint nAssetSteps, const float dS, const float dt, const float Int_Rate, const float Vol)
    {
    
    size_t global_id = get_global_id(0);
    
    float initSVal = global_id * dS;
    
    float Delta;
    float Gamma;
    float Theta;
    
    size_t index_previousStep = global_id-1;
    size_t index_nextStep = global_id+1;
    size_t index_current = global_id;
    
    if(global_id != 0 && global_id != (nAssetSteps - 1)){
    
        Delta = (fd1[index_nextStep] - fd1[index_previousStep]) / (2.f * dS);
        Gamma = (fd1[index_nextStep] - (2.f * fd1[index_current]) + fd1[index_previousStep]) / (dS * dS);
        Theta = 0.5f * Vol * Vol * initSVal * initSVal * Gamma + Int_Rate * initSVal * Delta - Int_Rate * fd1[index_current];
    
        fd2[index_current] = fd1[index_current] + dt * Theta;
    
    }else if(global_id == 0){
    
        fd2[index_current] = fd1[index_current] * (1-Int_Rate * dt);
    
    }else{
    
        fd2[index_current] = 2 * fd2[index_previousStep] - fd2[index_previousStep-1];
    
    }
    
    }
    """


monteCarloNextStep_source = """
    __kernel void monteCarloNextStep(__global float *vec_V, const uint timeStep, const uint nAssetSteps, const float dS, const float dt, const float Int_Rate, const float Vol)
    {
    
    
    
    }
    """
PTYPE = 1
S_init = 100
Strike = 100
nPaths = 10000
nAssetSteps = 2**8
Exp_Time = 1
#dt = 0.05
Int_Rate = 0.05
Vol = 0.2

dS = 2.0 * Strike / nAssetSteps

dt = 0.5 / Vol**2 / nAssetSteps**2

nTimeSteps = math.ceil(Exp_Time / dt)+1

dt = Exp_Time / nTimeSteps

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

mf = cl.mem_flags

fdResult = numpy.zeros(nAssetSteps,numpy.float32)
fdBuffer = numpy.zeros_like(fdResult)

for i in range(nAssetSteps):
    fdResult[i] = max(PTYPE * (i*dS - Strike),0.0)

t_start_gpu = time.time()

fdResult_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fdResult)
fdBuffer_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fdBuffer)

prg = cl.Program(ctx, finiteDifferenceNextStep_source).build()

for t in range(0,nTimeSteps):
    prg.finiteDifferenceNextStep(queue, fdResult.shape, None, fdResult_dev, fdBuffer_dev, numpy.uint32(t), numpy.uint32(nAssetSteps), numpy.float32(dS), numpy.float32(dt), numpy.float32(Int_Rate), numpy.float32(Vol))
  
    
    fdResult_dev, fdBuffer_dev = fdBuffer_dev, fdResult_dev

cl.enqueue_copy(queue, fdResult, fdResult_dev).wait()

t_end_gpu = time.time()

#MONTE CARLO

from scipy.stats import norm
from pyopencl.clrandom import RanluxGenerator
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

nPaths = 16*(2**9)
nLoops = 2**4
dtMonte = 0.01
nTimeStepsMonte = math.ceil(1 / dtMonte) + 1

nextStepPathKernel = ElementwiseKernel(ctx,"float *latestStep, float *ran, float Strike, float Int_Rate, float Exp_Time, float dt, float Vol","float rval = exp((Int_Rate - 0.5f * Vol*Vol)*dt + Vol * sqrt(dt) * ran[i]); latestStep[i] *= rval;","nextStepPathKernel")

excersisePriceKernel = ElementwiseKernel(ctx,"float *latestStep, float Strike, float Int_Rate, float Exp_Time","float rval = (latestStep[i]-Strike); latestStep[i] = exp(-Int_Rate*Exp_Time)  * max(rval,0.0f);","excersisePriceKernel")

#sumKernel = ReductionKernel(ctx, numpy.int32, neutral="0", reduce_expr="a+b", map_expr="x[i]", #*y[i]", arguments="__global float *x")#, __global in *y")

finiteDifferenceNextStep_source = """
    __kernel void reduce(
        __global float* buffer,
        __local float* scratch,
        __const int length,
        __global float* result) 
    {
    
        int global_index = get_global_id(0);
        int local_index = get_local_id(0);
        
        // Load data into local memory
        if (global_index < length) {
            scratch[local_index] = buffer[global_index];
        } else {
            // Infinity is the identity element for the min operation
            scratch[local_index] = INFINITY;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int offset = 1; offset < get_local_size(0); offset <<= 1) {
            int mask = (offset << 1) - 1;
            if ((local_index & mask) == 0) {
                float other = scratch[local_index + offset];
                float mine = scratch[local_index];
                scratch[local_index] = (mine < other) ? mine : other;
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (local_index == 0) {
            result[get_group_id(0)] = scratch[0];
        }
    }
"""

gen = RanluxGenerator(queue, nPaths, luxury=4, seed=time.time())
#gen = RanluxGenerator(queue, nPaths, luxury=4)

ran = cl.array.zeros(queue, nPaths, numpy.float32)
latestStep = cl.array.empty_like(ran)

averages = numpy.zeros(nLoops)
#averages = cl.array.zeros(queue, nLoops, numpy.float32)

tStartMonte = time.time()
theSum = 0

for loop in range(0,nLoops):
    
    latestStep.fill(S_init)
    
    for t in range(0,nTimeStepsMonte):
        gen.fill_normal(ran)
        gen.synchronize(queue)
        nextStepPathKernel(latestStep, ran, Strike, Int_Rate, Exp_Time, dtMonte, Vol)


    excersisePriceKernel(latestStep, Strike, Int_Rate, Exp_Time)

    #theSum = cl.array.sum(latestStep).get()

    latestStep_numpy = latestStep.get()
    theSum = numpy.sum(latestStep_numpy)

    #theSum = sumKernel(latestStep).get()

    if(theSum == 0):
        print("theSum was zero")
    averages[loop] = theSum / nPaths


monteAverage = numpy.mean(averages)
monteStdDeviation = numpy.std(averages)

tEndMonte = time.time()
print(averages)
#plt.figure()
#plt.hist(averages,40)
#plt.show()

#RESULTS

print("Finite difference = ",fdResult[100/ dS])
print("Finite difference time = ",t_end_gpu-t_start_gpu)
print("Finite difference dS = ",dS)
print("Monte Carlo = ", monteAverage)
print("Monte Carlo time = ",tEndMonte-tStartMonte)
print("Monte Carlo dS = ",monteStdDeviation)


#plt.figure()
#for x in range(0,6):
#    gen.fill_normal(ran)
#    print(numpy.std(ran.get()))
#plt.hist(ran.get(),40)
#plt.show()

#ranNumpy = numpy.random.normal(0,1,nPaths**2)
#print(ranNumpy.shape)
#plt.figure()
#for x in range(0,6):
#    ranNumpy = numpy.random.normal(0,1,nPaths)
#    print(numpy.std(ranNumpy))
#plt.hist(ranNumpy,40)
#plt.show()

print(2**12,2**7,2**9)
#from pyopencl.characterize import get_simd_group_size
#result = get_simd_group_size(device, out_type_size)
#print(result)

import EuropeanOption

PTYPE = 1
S_init = 100
S_max = 200
Strike = 100
nAssetSteps = 2**9
Exp_Time = 1
Int_Rate = 0.05
Vol = 0.2

print("Black Scholes value = ",EuropeanOption.valueBlackScholes(S_init,Strike,Exp_Time,Int_Rate,Vol,PTYPE))

print("Finite Difference value and dS = ",EuropeanOption.valueFiniteDifferenceGPU(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE))

nPaths = 2**9
dtMonte = 0.01
nMonteLoops = 10

print("Monte Carlo value = ",EuropeanOption.valueMonteCarloGPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE, nMonteLoops))



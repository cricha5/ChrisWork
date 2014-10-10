import numpy
import math
import time
import pyopencl as cl
import pyopencl.array as clarray
import matplotlib.pyplot as plt
import random

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

def evolve(vec_V,nTimeSteps, nAssetSteps, dS, dt,Int_Rate, Vol):
    for t_index in range(0,(nTimeSteps-1)):
        for x in range(1,(nAssetSteps-1)):

            vec_V[t_index+1,x] = vec_V[t_index,x] - dt * ((-0.5 * (Vol**2) * ((x*dS)**2) * ((vec_V[t_index,x + 1] - 2 * vec_V[t_index,x] + vec_V[t_index,x - 1]) / dS / dS)) - (Int_Rate * (x*dS) * ((vec_V[t_index,x + 1] - vec_V[t_index,x - 1]) / 2 / dS)) + (Int_Rate * vec_V[t_index,x]))
    
        #boundries
        vec_V[t_index+1,0] = vec_V[t_index,0] * (1-Int_Rate * dt)
        #vec_V[t_index,0] = 0
        vec_V[t_index+1,-1] = 2 * vec_V[t_index+1,-2] - vec_V[t_index+1,-3]

evolve_source = """
    __kernel void evolve(__global float *vec_V, const uint nTimeSteps, const uint nAssetSteps, const float dS, const float dt, const float Int_Rate, const float Vol)
    {

        size_t global_id = get_global_id(0);
        
        float initSVal = global_id * dS;
        uint t;
        size_t index_next;
        size_t index_current;
        
        float Delta;
        float Gamma;
        float Theta;
        
        
        for(t=0; t < nTimeSteps-1; t++){
            
            index_next = ((t+1)*nAssetSteps)+global_id;
            index_current = (t*nAssetSteps)+global_id;
            
            if(global_id != 0 && global_id != (nAssetSteps - 1)){
            
                Delta = (vec_V[index_current + 1] - vec_V[index_current - 1]) / (2.f * dS);
                Gamma = (vec_V[index_current + 1] - (2.f * vec_V[index_current]) + vec_V[index_current - 1]) / (dS * dS);
                Theta = 0.5f * Vol * Vol * initSVal * initSVal * Gamma + Int_Rate * initSVal * Delta - Int_Rate * vec_V[index_current];
                
                vec_V[index_next] = vec_V[index_current] + dt * Theta;

            }else if(global_id == 0){
            
                vec_V[index_next] = vec_V[index_current] * (1-Int_Rate * dt);
                
            }else{
            
                vec_V[index_next] = 2 * vec_V[index_next-1] - vec_V[index_next-2];
            
            }
            
            barrier(CLK_GLOBAL_MEM_FENCE);
            
        }
    }
    """

nextStep_source = """
    __kernel void nextStep(__global float *vec_V, const uint timeStep, const uint nAssetSteps, const float dS, const float dt, const float Int_Rate, const float Vol)
    {
    
    size_t global_id = get_global_id(0);
    
    float initSVal = global_id * dS;
    uint t = timeStep;
    size_t index_next;
    size_t index_current;
    
    float Delta;
    float Gamma;
    float Theta;
    
    index_next = ((t+1)*nAssetSteps)+global_id;
    index_current = (t*nAssetSteps)+global_id;
    
    if(global_id != 0 && global_id != (nAssetSteps - 1)){
    
    Delta = (vec_V[index_current + 1] - vec_V[index_current - 1]) / (2.f * dS);
    Gamma = (vec_V[index_current + 1] - (2.f * vec_V[index_current]) + vec_V[index_current - 1]) / (dS * dS);
    Theta = 0.5f * Vol * Vol * initSVal * initSVal * Gamma + Int_Rate * initSVal * Delta - Int_Rate * vec_V[index_current];
    
    vec_V[index_next] = vec_V[index_current] + dt * Theta;
    
    }else if(global_id == 0){
    
    vec_V[index_next] = vec_V[index_current] * (1-Int_Rate * dt);
    
    }else{
    
    vec_V[index_next] = 2 * vec_V[index_next-1] - vec_V[index_next-2];
    
    }

    }
    """


RATE = 0.05
PTYPE = 1.0
STRIKE = 100.0
EXPIRATION = 1.0
NASSETSTEPS = 2**8
VOLATILITY = 0.2

dS = 2.0 * STRIKE / NASSETSTEPS

dt = 0.9 / VOLATILITY**2 / NASSETSTEPS**2

nTimeSteps = math.ceil(EXPIRATION / dt)+1

dt = EXPIRATION / nTimeSteps

platforms = cl.get_platforms()
for platform in platforms:
    for device in platform.get_devices():
        print(device)
        print("local memory size",device.local_mem_size)
        print("global memory size",device.global_mem_size)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

vec_V = numpy.zeros((nTimeSteps,NASSETSTEPS),numpy.float32)
result = numpy.empty_like(vec_V)

for i in range(NASSETSTEPS):
    vec_V[0,i] = max(PTYPE * (i*dS - STRIKE),0.0)

t_start_gpu = time.time()

vec_V_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vec_V)

prg = cl.Program(ctx, evolve_source).build()

prg.evolve(queue, vec_V[0].shape, None, vec_V_dev, numpy.uint32(nTimeSteps), numpy.uint32(NASSETSTEPS), numpy.float32(dS), numpy.float32(dt), numpy.float32(RATE), numpy.float32(VOLATILITY))

cl.enqueue_copy(queue, result, vec_V_dev).wait()

t_end_gpu = time.time()

#print(result[-1])

#plt.figure()
#plt.plot(result[-1])


#Loop it!
t_start_gpu2 = time.time()

prg = cl.Program(ctx, nextStep_source).build()

for t in range(0,nTimeSteps-1):
    prg.nextStep(queue, vec_V[0].shape, None, vec_V_dev, numpy.uint32(t), numpy.uint32(NASSETSTEPS), numpy.float32(dS), numpy.float32(dt), numpy.float32(RATE), numpy.float32(VOLATILITY))

cl.enqueue_copy(queue, result, vec_V_dev).wait()

t_end_gpu2 = time.time()

#print(result[-1])

t_start_cpu = time.time()
#evolve(vec_V, nTimeSteps, NASSETSTEPS, dS, dt,RATE, VOLATILITY)
t_end_cpu = time.time()

#plt.plot(vec_V[-1])

print("Total gpu time = ", t_end_gpu - t_start_gpu)
print("Total gpu2 time = ", t_end_gpu2- t_start_gpu2)
print("Total cpu time = ", t_end_cpu - t_start_cpu)

#plt.figure()

#plt.show()

def stockPathsRandom(S_init, nPaths, dt, Exp_Time, Int_Rate, Vol):
    nTimeSteps = math.ceil(Exp_Time / dt) + 1
    
    stockPaths = numpy.empty((nTimeSteps,nPaths))
    stockPaths[0] = S_init
    previousStockPath = stockPaths[0]
    
    for t in range(0,nTimeSteps-1):
        stockPaths[t+1] = stockPaths[t] * numpy.exp((Int_Rate - 0.5 * Vol**2)*dt + Vol * math.sqrt(dt) * numpy.random.normal(loc=0.0, scale=1.0, size=nPaths))
    
    #print(stockPaths)

    return stockPaths


S_init = 100
Strike = 100
nPaths = 10000
Exp_Time = 1
dt = 0.05
Int_Rate = 0.05
Vol = 0.2

nTimeSteps = math.ceil(Exp_Time / dt) + 1

stockPaths = stockPathsRandom(S_init, nPaths, dt, Exp_Time, Int_Rate, Vol)
timeAxis = numpy.empty_like(stockPaths)

for t in range(0,nTimeSteps):
    timeAxis[t] = t * dt

#plt.figure()
#for y in range(0,nPaths-1):
#    plt.plot(timeAxis[:,y], stockPaths[:,y])
#plt.axis([0, Exp_Time, 0, 160])
#plt.show()

optionValueAtExp = numpy.empty_like(stockPaths[-1])
for y in range(0,nPaths-1):
    optionValueAtExp[y] = max(math.exp(-Int_Rate*Exp_Time)*(-(Strike-stockPaths[-1,y])),0)


avg = numpy.sum(optionValueAtExp)/nPaths
print("average = ", avg)

from scipy.stats import norm

time_sqrt = math.sqrt(Exp_Time)
d1 = (math.log(S_init/Strike)+(Int_Rate+Vol*Vol/2.)*Exp_Time)/(Vol*time_sqrt)
d2 = d1-(Vol*time_sqrt)
if 1==1:
    c = S_init * norm.cdf(d1) - Strike * math.exp(-Int_Rate*Exp_Time) * norm.cdf(d2)
else:
    c =  Strike * math.exp(-Int_Rate*Exp_Time) * norm.cdf(-d2) - S_init * norm.cdf(-d1)

print("black scholes =", c)


#print("",result[0])
#print("",result[-1])
print("Finite difference = ",result[-1,100/ (2.0 * STRIKE / NASSETSTEPS)])

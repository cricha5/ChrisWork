from scipy.stats import norm
import math

def valueEuropeanOption(S_init,Strike,Exp_Time,Int_Rate,Vol,PTYPE):
    time_sqrt = math.sqrt(Exp_Time)
    d1 = (math.log(S_init/Strike)+(Int_Rate+Vol*Vol/2.)*Exp_Time)/(Vol*time_sqrt)
    d2 = d1-(Vol*time_sqrt)
    if 1==1:
        c = S_init * norm.cdf(d1) - Strike * math.exp(-Int_Rate*Exp_Time) * norm.cdf(d2)
    else:
        c =  Strike * math.exp(-Int_Rate*Exp_Time) * norm.cdf(-d2) - S_init * norm.cdf(-d1)

    return c
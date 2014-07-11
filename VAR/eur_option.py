'''
Created on 14/11/2014

@author: javgar119
'''
from QuantLib import *

# dates
calendar = TARGET()
todaysDate = Date(14, 11, 2014)
Settings.instance().evaluationDate = todaysDate
settlementDate = Date(14, 11, 2014)
maturity = Date(17, 2, 2015)
dayCounter = Actual365Fixed()

# option parameters
option_type = Option.Call
underlying = 39.0
strike = 41.5
dividendYield = 0.0
riskFreeRate = 0.01
volatility = 0.17

# basic option
payoff = PlainVanillaPayoff(option_type, strike)
exercise = EuropeanExercise(maturity)
europeanOption = VanillaOption(payoff, exercise)

# handle setups
underlyingH = QuoteHandle(SimpleQuote(underlying))
flatTermStructure = YieldTermStructureHandle(FlatForward(settlementDate, riskFreeRate, dayCounter))
dividendYield = YieldTermStructureHandle(FlatForward(settlementDate, dividendYield, Actual365Fixed()))
flatVolTS = BlackVolTermStructureHandle(BlackConstantVol(settlementDate, calendar, volatility, dayCounter))

# done
bsmProcess = BlackScholesMertonProcess(underlyingH,
                                       dividendYield,
                                       flatTermStructure,
                                       flatVolTS)

# method: analytic

europeanOption.setPricingEngine(AnalyticEuropeanEngine(bsmProcess))

value = europeanOption.NPV()

print("European option value ", value)

















import numpy as np

# === DEFINE: EMA, MACD, CSD, SRInc === #

def EMA(data, period, smoothing="normal"):
    '''
    data -> numpy array of prices (USUALLY CLOSE)
    period -> time period used to calculate the EMA (e.g. 50/200-day)

    return -> EMA calculated over all the input data. The size will be smaller;
              it will be the length of the data MINUS period PLUS 1
    '''

    length = data.shape[0] - period + 1
    averages = np.ndarray((length))
    
    alpha = 2 / (period + 1)
    if smoothing == "normal":
        alpha = 2 / (period + 1)
    elif smoothing == "wilder":
        alpha = 1 / period

    averages[0] = np.average(data[0:10])

    for i in range(1, length):
        averages[i] = data[i + period - 1] * alpha + averages[i - 1] * (1 - alpha)

    return averages

def MACD(EMA12, EMA26):
    return EMA12 - EMA26

def RSI(data, period):
    '''
    SEE: https://en.wikipedia.org/wiki/Relative_strength_index 

    data -> numpy array of prices (USUALLY CLOSE)
    period -> period that RSI is calculated over

    return -> RSI calculated over input data. Size reduced to length of the data MINUS period
    '''

    # DELTA: Close (now) - Close (previous)
    # does NOT contain the delta for the LATEST period
    deltas = data[1:] - data[:-1]

    ups = np.where(deltas > 0, deltas, 0)
    downs = np.where(deltas < 0, -deltas, 0)


    RS = EMA(ups, period, smoothing="wilder") / EMA(downs, period, smoothing="wilder")

    # print(deltas)
    # print(ups)
    # print(downs)
    # print(EMA(downs, period, smoothing="wilder"))
    # print(RS)
    # print()
    RSI = 100 - 100 / (1 + RS)

    return RSI

def CSD(df):
    '''
    CSD: Candlestick Delta

    For each period in the dataframe (open, high, low, close), calculates the delta, and the high/low depending
    on the direction of the delta (+ -> low, - -> high).

    return -> The ratio between the length of the extrema bar in question, and the length of the opposite bar. 
              Signed; positive with positive delta, and vice versa
              (also returns JUST the deltas)
    '''

    opens = df.loc[:, "Open"].to_numpy()
    closes = df.loc[:, "Close"].to_numpy()
    highs = df.loc[:, "High"].to_numpy()
    lows = df.loc[:, "Low"].to_numpy()
    
    # max and min endpoint values (e.g. open/close)
    max_endpts = np.where(opens > closes, opens, closes)
    min_endpts = np.where(opens < closes, opens, closes)

    # change from open to close
    deltas = closes - opens

    # the difference between the extreme and the open/close price to the OTHER extreme, depending on delta direction.
    # EXAMPLE: O: 20, H: 30, L: 15, C: 18
    #          delta: 18 - 20 = -2
    #          numerator: 30 - 20 = 10
    #          denominator: 20 - 15 = 5
    #          ratio: 10 / 5 = 2
    extrema_lengths = np.where(deltas > 0, min_endpts - lows, highs - max_endpts)
    opposite_lengths = np.where(deltas > 0, highs - min_endpts, max_endpts - lows)
    ratios = np.where(deltas > 0, extrema_lengths / opposite_lengths, -extrema_lengths / opposite_lengths)

    return deltas, ratios

def SRD(df, period=7, threshold=0.1):
    '''
    SRD: Short-Run Delta

    For each time in the given dataframe, determines if in the following period a percent increase/decrease greater than 
    the threshold happens.

    df -> stock history
    period -> time period over which to look for increases
    threshold -> the threshold for increase/decrease ratio. 

    return -> truth array for each time in the history.
              +1: increase
               0: no change greater than threshold
              -1: decrease
    '''

    closes = df.loc[:, "Close"].to_numpy()

    delta_truth = np.zeros((closes.shape))

    for i in range(1, period):
        starts = closes[:-i]
        ends = closes[i:]
        deltas = (ends - starts) / starts

        deltas = np.where(deltas >= threshold, 1, 0) + np.where(deltas <= -threshold, -1, 0)
        fills = np.zeros((i,))
        deltas = np.concatenate((deltas, fills))

        delta_truth = delta_truth + deltas

    return delta_truth


'''It is important to remember that since these
   are not independent events, P(A) * P(B) =/= P(A ∩ B)
   
   Therefore, P(A ∩ B) needs to be calculated by finding the actual
   intersection of sets and finding the probabilities from there.'''

def RWEXP(a, b, flatten=False):
    '''
    RWEXP: Row-Wise Expansion
    For arrays a (x, N) and b (y, N), computes the Hadamard product
    (elementwise product) of each row-row combination between arrays a and b.

    a, b -> arrays of shape (x, N) and (y, N) respectively
    flatten -> take a fucking guess

    return -> row-wise expansion/outer product
    '''
    
    output = np.ndarray(np.concatenate((a.shape[:1], b.shape)))
    for i in range(a.shape[0]):
        output[i] = a[i] * b

    if flatten:
        return output.reshape(a.shape[0] * b.shape[0], b.shape[1])
    else:
        return output



def intersect_probs(indicators, possibilities):
    '''
    indicators -> a LIST of np arrays, where each array is one indicator with N events.
                  These indicators should ALREADY have been one-hotted.
                  Indicators to include: Price deltas
                                         CSD
                                         MACD
                                         RSI

    possibilities -> number of distinct events possible for each indicator (i.e. N for each indicator)
    '''

    # intersects for all indicators, EXCLUDING deltas
    intersects = indicators[-1]
    for i in range(len(indicators) - 2, 0, -1):
        intersects = RWEXP(indicators[i], intersects)

    # intersects INCLUDING indicators
    delta_ints = RWEXP(indicators[0], intersects)

    P_deltas = delta_ints.sum(axis=-1) / delta_ints.shape[-1]
    P_indicators = intersects.sum(axis=-1) / intersects.shape[-1]

    P_indicators = np.repeat(P_indicators[np.newaxis, :], possibilities[0], axis=0)

    # SOMETIMES WHEN INTERSECTS DON'T HAPPEN, P(x) = 0, SO THERE IS DIV0 ERR
    P_inc_with_givens = P_deltas / P_indicators

    '''Return with sample size for all intersects EXCLUDING deltas'''
    # return P_inc_with_givens, intersects.sum(axis=1)

    '''Return INCLUDING deltas in the intersects'''
    return P_inc_with_givens, delta_ints.sum(axis=-1)
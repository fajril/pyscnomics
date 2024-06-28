import numpy as np
from . import RPDModel


def oil_ltp_predict(volume: float) -> np.ndarray:
    """Oil forecast based on SKK Migas LTP Model
    ==============================================
    This function automatically set the RDPModel parameters with
    SKK Migas LTP forecast for oil. There might be discrepancy
    from the rounding errors with SKK Migas official LTP.

    Parameter
    ---
    volume : float
        Volume of resources (MSTB)

    Return
    ---
    rate : ndarray
        Yearly production rate (MSTB/yr)
    """
    if volume <= 4000:
        year_rampup = 0
        drate = 0.200633347
        q_plateau_ratio = 0.182937586835236
        q_min_ratio = 0.0014828
    elif volume <= 10_000:
        year_rampup = 1
        drate = 0.160502926
        q_plateau_ratio = 0.147021007393939
        q_min_ratio = 0.0036657
    elif volume <= 20_000:
        year_rampup = 1
        drate = 0.160549732
        q_plateau_ratio = 0.128715547194678
        q_min_ratio = 0.0037641
    elif volume <= 50_000:
        year_rampup = 2
        drate = 0.138588938
        q_plateau_ratio = 0.111027381539511
        q_min_ratio = 0.0060461
    elif volume <= 100_000:
        year_rampup = 2
        drate = 0.130770622
        q_plateau_ratio = 0.0972034569873284
        q_min_ratio = 0.0071092
    elif volume <= 200_000:
        year_rampup = 3
        drate = 0.131308644
        q_plateau_ratio = 0.0861313010167252
        q_min_ratio = 0.0081037
    elif volume <= 500_000:
        year_rampup = 3
        drate = 0.109315569
        q_plateau_ratio = 0.0742950873371555
        q_min_ratio = 0.0115846
    else:
        year_rampup = 3
        drate = 0.103678476
        q_plateau_ratio = 0.0650446693978552
        q_min_ratio = 0.0137343
    rpd = RPDModel(year_rampup, drate, q_plateau_ratio, q_min_ratio)
    return rpd.predict(volume)


def gas_ltp_predict(volume: float) -> np.ndarray:
    """Gas forecast based on SKK Migas LTP Model
    ==============================================
    This method automatically set the class parameters with
    SKK Migas LTP forecast for gas. There might be discrepancy
    from the rounding errors.

    Parameter
    ---
    volume : float
        Volume of resources (BSCF)

    Return
    ---
    rate : ndarray
        Yearly production rate (BSCF/yr)
    """
    if volume <= 100:
        year_rampup = 2
        drate = 0.510825623765991
        q_plateau_ratio = 0.123459112068285
        q_min_ratio = 0.0000125385020719761
    elif volume <= 200:
        year_rampup = 2
        drate = 0.510825623765991
        q_plateau_ratio = 0.0990140494482555
        q_min_ratio = 0.0000279329515875812
    elif volume <= 500:
        year_rampup = 2
        drate = 0.510825623765991
        q_plateau_ratio = 0.0900958146440382
        q_min_ratio = 0.0000423616992187692
    elif volume <= 1000:
        year_rampup = 2
        drate = 0.510825623765991
        q_plateau_ratio = 0.0763472955954202
        q_min_ratio = 0.0000997148666720888
    elif volume <= 2000:
        year_rampup = 2
        drate = 0.510825623765991
        q_plateau_ratio = 0.0662490413481761
        q_min_ratio = 0.000240349571639406
    else:
        year_rampup = 2
        drate = 0.510825623765991
        q_plateau_ratio = 0.0585312743031216
        q_min_ratio = 0.000589860388919471
    rpd = RPDModel(year_rampup, drate, q_plateau_ratio, q_min_ratio)
    return rpd.predict(volume)

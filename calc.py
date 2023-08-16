import numpy as np

def lldistkm(latlon1, latlon2):
    radius = 6371
    lat1 = latlon1[:, 0] * np.pi / 180
    lat2 = latlon2[:, 0] * np.pi / 180
    lon1 = latlon1[:, 1] * np.pi / 180
    lon2 = latlon2[:, 1] * np.pi / 180
    deltaLat = lat2 - lat1
    deltaLon = lon2 - lon1
    a = np.sin((deltaLat) / 2) ** 2 + np.multiply(np.multiply(np.cos(lat1), np.cos(lat2)), np.sin(deltaLon / 2) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d1km = np.multiply(radius, c)

    x = np.multiply(deltaLon, np.cos((lat1 + lat2) / 2))
    y = deltaLat
    d2km = np.multiply(radius, np.sqrt(np.multiply(x, x) + np.multiply(y, y)))

    return d1km, d2km

def calc_neival(c_trans, p_sl, y_node, q_node, lccf, rtpref, dtonei, cutflag, totcpcty, totstock, edgechange):
    pay_noevent = np.zeros((c_trans.shape[1], 1))  # len in numpy defaults to 1st dimension and shape provides a tuple
    pay_event = np.zeros((c_trans.shape[1], 1))
    xpay_noevent = np.zeros((c_trans.shape[1], 1))
    ypay_noevent = np.zeros((c_trans.shape[1], 1))
    xpay_event = np.zeros((c_trans.shape[1], 1))
    ypay_event = np.zeros((c_trans.shape[1], 1))
    value_noevent = np.zeros((c_trans.shape[1], 1))
    value_event = np.zeros((c_trans.shape[1], 1))
    ival_noevent = np.zeros((c_trans.shape[1], 1))
    ival_event = np.zeros((c_trans.shape[1], 1))
    dwght_noevent = np.zeros((c_trans.shape[1], 1))
    dwght_event = np.zeros((c_trans.shape[1], 1))
    salwght_noevent = np.zeros((c_trans.shape[1], 1))
    salwght_event = np.zeros((c_trans.shape[1], 1))
    valuex = np.zeros((c_trans.shape[1], 1))
    valuey = np.zeros((c_trans.shape[1], 1))
    iset = np.arange(0, c_trans.shape[1]).reshape(1, c_trans.shape[1])

    for i in np.arange(0, c_trans.shape[1]):
        pay_noevent[i, 0] = y_node[i, 0] * q_node[0, i] - c_trans[0, i] * q_node[0, i]  # payoff with no S&L event
        pay_event[i, 0] = y_node[i, 0] * q_node[0, i] - c_trans[0, i] * q_node[0, i] - y_node[i, 0] * \
                          q_node[0, i]  # payoff with S&L event
        xpay_noevent[i, 0] = pay_noevent[i, 0]  # payoff for route A with no S&L event
        xpay_event[i, 0] = pay_event[i, 0]  # payoff for route A with S&L event

    for i in np.arange(0, c_trans.shape[1]):
        inset = np.where(dtonei == dtonei[i, 0])[0]
        mask = np.not_equal(inset, i)
        ypay_noevent[i, 0] = np.mean([pay_noevent[j, 0] for j in range(len(mask)) if mask[j]])
        ypay_event[i, 0] = np.mean([pay_event[j, 0] for j in range(len(mask)) if mask[j]])
        value_noevent[i, 0] = np.abs(ypay_noevent[i, 0] - xpay_noevent[i, 0]) / (np.abs(ypay_noevent[i, 0]) + np.abs(xpay_noevent[i, 0]) + 1)
        value_event[i, 0] = np.abs(ypay_event[i, 0] - xpay_event[i, 0]) / (np.abs(ypay_event[i, 0]) + np.abs(xpay_event[i, 0]) + 1)
        ipntlval = np.flip(np.argsort(np.array([value_noevent[i, 0], value_event[i, 0]])))
        ival_noevent[i, 0] = ipntlval[0]
        ival_event[i, 0] = ipntlval[1]
        dwght_noevent[i, 0] = (lccf ** ival_noevent[i, 0]) / ((lccf ** ival_noevent[i, 0]) * (1 - p_sl[0, i]) + (lccf ** ival_event[i, 0]) * p_sl[0, i])
        dwght_event[i, 0] = (lccf ** ival_event[i, 0]) / ((lccf ** ival_noevent[i, 0]) * (1 - p_sl[0, i]) + (lccf ** ival_event[i, 0]) * p_sl[0, i])
        salwght_noevent[i, 0] = (1 - p_sl[0, i]) * dwght_noevent[i, 0]
        salwght_event[i, 0] = p_sl[0, i] * dwght_event[i, 0]
        valuey[i, 0] = salwght_noevent[i, 0] * ypay_noevent[i, 0] + salwght_event[i, 0] * ypay_event[i, 0]
        valuex[i, 0] = salwght_noevent[i, 0] * xpay_noevent[i, 0] + salwght_event[i, 0] * xpay_event[i, 0]

    # Selection based on maximize profits while less than average S&L risk
    route = np.stack([np.multiply(np.transpose(rtpref), valuex)[:, 0].tolist(), np.transpose(p_sl)[:, 0].tolist(),
                      np.transpose(q_node)[:, 0].tolist(), np.transpose(iset)[:, 0].tolist(), dtonei[:, 0].tolist(),
                      np.transpose(totcpcty)[:, 0].tolist()], axis=1)
    rankroute = route[route[:, 0].argsort()[::-1]]

    # dtos = np.unique(dtonei[dtonei.any(axis=1)])
    dtos = np.unique(dtonei)
    icut = []

    '''
    07/15/2023 Add if condition for rankroute being empty arr
    '''
    if len(rankroute) == 0:
      neipick = []
      neivalue = []
      return neipick, neivalue, valuex

    if len(dtos) > 1:
        for j in np.arange(0, len(dtos)):
            idto = np.where(rankroute[:, 4] == dtos[j])[0]
            if np.where(valuex[np.where(dtonei == dtos[j])[0]] > 0)[0].size == 0:  # CHECK
                subicut = np.where(np.cumsum(rankroute[idto, 5]) >= totstock)[0][0:int(edgechange[j][0])][0]
            elif np.where(rankroute[idto, 1] > 0)[0].size == 0:
                subicut = np.where(rankroute[idto, 0] >= 0)[0][0:int(edgechange[j][0])] # Hashir: edgechange[j][0]: float type -- > convert to int type
            elif np.where(np.cumsum(rankroute[idto, 5]) >= totstock)[0].size == 0:
                subicut = np.where(rankroute[idto, 0] >= 0)[0][0:int(edgechange[j][0])] # Hashir: edgechange[j][0]: float type -- > convert to int type
            else:
                subicut = np.where(rankroute[idto, 0] >= 0)[0][0:int(edgechange[j][0])] # Hashir: edgechange[j][0]: float type -- > convert to int type
            if cutflag[int(dtos[j] - 1), 0] == 1:
                subicut = []
            # icut = np.concatenate((icut, idto[subicut]), axis=0)
            icut.append(idto[subicut])
    else:
        if len(np.where(valuex > 0)[0]) == 0:
            icut = np.transpose(np.arange(0, np.where(np.cumsum(rankroute[:, 5]) >= totstock)[0][0]))
        elif np.where(rankroute[:, 0] > 0)[0].size == 0:
            volcut = np.transpose(np.arange(0, np.where(np.cumsum(rankroute[:, 5]) >= totstock)[0][0]))
            valcut = np.where(rankroute[:, 0] >= 0)
            icut = np.isin(valcut, volcut)
        elif np.where(np.cumsum(rankroute[:, 5]) >= totstock)[0].size == 0:
            icut = np.where(rankroute[:, 0] >= 0)
        else:
            icut = np.where(rankroute[:, 0] >= 0)

    neipick = rankroute[icut, 3]
    neivalue = rankroute[icut, 0]

    return neipick, neivalue, valuex

def calc_intrisk(sloccur, t_eff, alpharisk, betarisk, timeweight):
    slevnt = np.sum(np.multiply(sloccur, np.tile(np.transpose(np.power(timeweight, t_eff)), (1, len(sloccur[1, :])))),
                    0, keepdims=True)  # make sure to load t_eff as 2D array in order for this to work properly
    tmevnt = np.sum(np.power(timeweight, t_eff))
    sl_risk = (slevnt + alpharisk) / (tmevnt + alpharisk + betarisk)
    return sl_risk, slevnt, tmevnt

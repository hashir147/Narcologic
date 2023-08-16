import numpy as np
import pandas as pd
import scipy

from calc import lldistkm, calc_neival, calc_intrisk
from data import data_sourcing, MTMCI_func, data_processing
from initialize import load_expmntl_parms, intrd_tables_batch
from optimize import optimize_interdiction_batch, optimizeroute_multidto

np.seterr(divide = 'ignore')

ERUNS = 11
TSTART = 0
TMAX = 181 # 15 years at monthly time steps

erun = 3
mrun = 2

np.random.seed(mrun)

# load experimental parameters file
sl_max, sl_min, baserisk, riskmltplr, startstock, sl_learn, rt_learn, losslim, prodgrow, targetseize, \
    intcpctymodel, profitmodel, endstock, growthmdl, timewght, locthink, expandmax, empSLflag, optSLflag, suitflag, \
    rtcap, basecap, p_sucintcpt = load_expmntl_parms(ERUNS)

dcoast = scipy.io.loadmat('data/coast_dist.mat')['dcoast']
LANDSUIT = scipy.io.loadmat("data/landsuit_file_default.mat")['LANDSUIT']

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@ Agent Attributes @@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Network Agent #
ndto = 2  # initial number of DTOs
dtocutflag = np.zeros((ndto, 1))
DTOBDGT = np.zeros((ndto, TMAX))
losstol = losslim[0, erun]  # tolerance threshold for loss due to S&L, triggers route fragmentation
stock_0 = startstock[0, erun]  # initial cocaine stock at producer node
stock_max = endstock[0, erun]
startvalue = 4500  # producer price, $385/kg: Zoe's numbers 4,500 in Panama
deltavalue = 4.46  # added value for distance traveled $8/kilo/km: Zoe's numbers $4.46
nodeloss = 0  # amount of cocaine that is normally lost (i.e., non-interdiction) at each node
ctrans_inland = 371  # transportation costs (kg/km) over-ground (3.5), includes
ctrans_coast = 160  # transportation costs (kg/km) via plane or boat (1.5)
ctrans_air = 3486
delta_rt = rt_learn[0, erun]  # reinforcement learning rate for network agent

# (i.e., weight on new information for successful routes)
# perceived risk model
alpharisk = 2
betarisk = 0.5
timewght_0 = timewght[0, erun]

slprob_0 = 1 / (np.sum(np.power(timewght_0, np.array((np.arange(0, 13))))) + betarisk)
bribepct = 0.3
rentcap = 1 - bribepct
edgechange = expandmax[0, erun] * np.ones((ndto, 1))

savedState = np.random.get_state()  # CHECK


###################################################################
#   Build trafficking network - NodeTable and EdgeTable   #
###################################################################

EdgeTable = pd.read_csv('data/EdgeTable.csv')
EdgeTable['EndNodes'] = EdgeTable[['EndNodes_1', 'EndNodes_2']].values.tolist()
EdgeTable = EdgeTable.drop(columns=['EndNodes_1', 'EndNodes_2'])
NodeTable = pd.read_csv('data/NodeTable.csv')
NodeTable['Row'] = NodeTable['Row'] - 1
NodeTable['Col'] = NodeTable['Col'] - 1
EdgeTable['Capacity'] = rtcap[0, erun] * np.ones(EdgeTable.shape[0])
nnodes = NodeTable.shape[0]
mexnode = nnodes - 1
endnodeset = mexnode
coastdist = [dcoast[row][col] for row, col in zip(NodeTable['Row'], NodeTable['Col'])]
NodeTable['CoastDist'] = coastdist
NodeTable['CoastDist'][0] = 0
NodeTable['CoastDist'][nnodes - 1] = 0

# Assign nodes to initials DTOs
for nn in range(1, nnodes - 2):
    try:
        westdir = NodeTable['Col'][nn] - np.where(np.isnan(dcoast[NodeTable['Row'][nn],
        np.arange(0, NodeTable['Col'][nn] - 1)]) == 1)[0][-1]
    except IndexError:
        westdir = 0
    try:
        eastdir = np.where(np.isnan(dcoast[NodeTable['Row'][nn], np.arange(NodeTable['Col'][nn],
                                                                           LANDSUIT.shape[1] - 1)]) == 1)[0][0]
    except IndexError:
        eastdir = 0

    if westdir < 2.5 * eastdir:
        NodeTable['DTO'][nn] = 1
    else:
        NodeTable['DTO'][nn] = 2

ADJ = np.zeros((nnodes, nnodes))  # adjacency matrix for trafficking network
DIST = np.zeros((nnodes, nnodes))  # geographic distance associated with edges
ADDVAL = np.zeros((nnodes, nnodes))  # added value per edge in trafficking network
WGHT = np.ones((nnodes, nnodes))  # dynamic weighting of edges
SLRISK = slprob_0 * np.ones((nnodes, nnodes))  # dynamic perceived risk of seisure and loss per edge by node agent
CPCTY = np.zeros((nnodes, nnodes))  # maximum flow possible between nodes
CTRANS = np.zeros((nnodes, nnodes, TMAX))  # ransportation costs between nodes
RMTFAC = np.zeros((nnodes, nnodes))  # landscape factor (remoteness) influencing S&L risk
COASTFAC = np.zeros((nnodes, nnodes))  # landscape factor (distance to coast) influencing S&L risk
LATFAC = np.zeros((nnodes, nnodes))  # decreased likelihood of S&L moving north to reflect greater DTO investment
BRDRFAC = np.zeros((nnodes, nnodes))  # increased probability of S&L in department bordering an
# international border

SUITFAC = np.zeros((nnodes, nnodes))
STOCK = np.zeros((nnodes, TMAX))  # dynamic cocaine stock at each node
PRICE = np.zeros((nnodes, TMAX))  # $/kilo at each node
RISKPREM = np.ones((nnodes, nnodes, TMAX))
OUTFLOW = np.zeros((nnodes, TMAX))  # dynamic stock of cocaine leaving from each node
TOTCPTL = np.zeros((nnodes, TMAX))  # total value of cocaine at each node
ICPTL = np.zeros((nnodes, TMAX))  # dynamic illicit capital accumulated at each node
BRIBE = np.zeros((nnodes, TMAX))  # annual bribe payments made at each node to maintain control
MARGIN = np.zeros((nnodes, TMAX))  # gross profit per node after purchasing, trafficking, and selling
RENTCAP = np.zeros((nnodes, TMAX))  # portion of MARGIN retained at node as profit
LEAK = np.zeros((nnodes, TMAX))  # dynamic amount of cocaine leaked at each node
activeroute = np.empty((nnodes, TMAX), dtype=object)  # track active routes
avgslrisk = np.empty((nnodes, TMAX), dtype=object)  # average S&L risk at each node given active routes
totslrisk = np.zeros((1, TMAX))  # network-wide average S&L risk
slcpcty = np.zeros((1, TMAX))

np.random.set_state(savedState)

for k in range(0, nnodes):
    # Create adjacency matrix
    ADJ[k, EdgeTable['EndNodes'].str[1][np.where(EdgeTable['EndNodes'].str[0] == k)[0]]] = 1

# Node Attributes
remotefac = (1 - NodeTable['PopSuit'].to_numpy()).reshape(-1, 1)
remotefac[0, 0] = 0
brdrfac = NodeTable['DistBorderSuit'].to_numpy().reshape(-1, 1)
brdrfac[0, 0] = 0
suitfac = NodeTable['LandSuit'].to_numpy().reshape(-1, 1)
suitfac[0, 0] = 0
coastfac = (NodeTable['CoastDist'].to_numpy() / np.amax(NodeTable['CoastDist'])).reshape(-1, 1)
coastfac[0, 0] = 0
nwvec = (np.sqrt(np.multiply(0.9, NodeTable['Lat'].to_numpy() ** 2) +
                 np.multiply(0.1, NodeTable['Lon'].to_numpy() ** 2))).reshape(-1, 1)
latfac = 1 - nwvec / np.amax(nwvec)
latfac[0, 0] = 0

# Create adjacency matrix
iendnode = NodeTable.loc[NodeTable['DeptCode'] == 2, 'ID'].iloc[0]
ADJ[EdgeTable['EndNodes'].str[0][np.where(EdgeTable['EndNodes'].str[1] == iendnode)[0]], iendnode] = 1

'''
Put distance between Node 0 and Node 162 into DIST
hard-coded
'''
node0_location = NodeTable.loc[[0], ['Lat', 'Lon']].to_numpy()
node162_location = NodeTable.loc[[162], ['Lat', 'Lon']].to_numpy()
d1km_0_162, d2km_0_162 = lldistkm(node0_location, node162_location)
DIST[0, 162] = d1km_0_162

# ---------------end----------------------

for j in range(0, nnodes):

    # Create weight and capacity matrices
    WGHT[0, np.where(ADJ[j, :] == 1)[0]] = EdgeTable['Weight'][np.where(ADJ[j, :] == 1)[0]]
    CPCTY[j, np.where(ADJ[j, :] == 1)[0]] = EdgeTable['Capacity'][np.where(ADJ[j, :] == 1)[0]]
    # Create distance (in km) matrix
    rows = [int(x) for x in range(len(ADJ[j, :])) if ADJ[j, x] == 1]
    print("j = ", j, "rows = ", rows)
    latlon2 = NodeTable.loc[rows, ['Lat', 'Lon']].to_numpy()

    latlon1 = np.tile(NodeTable.loc[j, ['Lat', 'Lon']].to_numpy(), (len(latlon2[:, 1]), 1))
    d1km, d2km = lldistkm(latlon1, latlon2)
    DIST[j, np.where(ADJ[j, :] == 1)[0]] = d1km
    # print("distance for DIST: ", d1km)

    # Create added value matrix (USD) and price per node
    ADDVAL[j, np.where(ADJ[j, :] == 1)[0]] = np.multiply(deltavalue, DIST[j, np.where(ADJ[j, :] == 1)[0]])
    if j == 0:
        PRICE[j, TSTART] = startvalue
    elif j == endnodeset:
        continue
    elif 156 <= j <= 159:
        isender = EdgeTable['EndNodes'].str[0][np.where(EdgeTable['EndNodes'].str[1] == j)[0]]
        inextleg = EdgeTable['EndNodes'].str[1][np.where(EdgeTable['EndNodes'].str[0] == j)[0]]
        PRICE[j, TSTART] = PRICE[isender, TSTART] + ADDVAL[isender, j] + PRICE[isender, TSTART] + np.mean(
            ADDVAL[j, inextleg])
        # even prices for long haul routes
        if j == 159:
            PRICE[np.array([156, 159]), TSTART] = np.amin(PRICE[np.array([156, 159]), TSTART])
            PRICE[np.array([157, 158]), TSTART] = np.amin(PRICE[np.array([157, 158]), TSTART])
    else:
        isender = EdgeTable['EndNodes'].str[0][EdgeTable['EndNodes'].str[1] == j]
        PRICE[j, TSTART] = np.mean(PRICE[isender, TSTART] + ADDVAL[isender, j])

    """
    loop not needed as endnodeset is an integer
    for en in range(0, len(endnodeset)+1):
        PRICE[endnodeset[en], TSTART] = np.amax(PRICE[np.where(ADJ[:, endnodeset[en]] == 1)[0], TSTART])
    """
    PRICE[endnodeset, TSTART] = np.amax(PRICE[np.where(ADJ[:, endnodeset] == 1)[0], TSTART])
    RMTFAC[j, np.where(ADJ[j, :] == 1)[0]] = remotefac[np.where(ADJ[j, :] == 1)[0]].flatten()
    COASTFAC[j, np.where(ADJ[j, :] == 1)[0]] = coastfac[np.where(ADJ[j, :] == 1)[0]].flatten()
    LATFAC[j, np.where(ADJ[j, :] == 1)[0]] = latfac[np.where(ADJ[j, :] == 1)[0]].flatten()
    BRDRFAC[j, np.where(ADJ[j, :] == 1)[0]] = brdrfac[np.where(ADJ[j, :] == 1)[0]].flatten()
    SUITFAC[j, np.where(ADJ[j, :] == 1)[0]] = suitfac[np.where(ADJ[j, :] == 1)[0]].flatten()

    # Transportation costs
    ireceiver = (EdgeTable['EndNodes'].str[1][np.where(EdgeTable['EndNodes'].str[0] == j)[0]]).to_numpy()
    ireceiver = ireceiver.reshape(len(ireceiver), 1)
    idist_ground = np.logical_and(DIST[j, ireceiver] > 0, DIST[j, ireceiver] <= 500).reshape(-1, 1)
    idist_air = (DIST[j, ireceiver] > 500).reshape(-1, 1)
    """ CHECK FOR DIVIDE BY ZERO for CTRANS calculations """
    CTRANS[j, ireceiver[idist_ground], TSTART] = np.multiply(ctrans_inland,
                                                             DIST[j, ireceiver[idist_ground]]) / DIST[0, mexnode]
    CTRANS[j, ireceiver[idist_air], TSTART] = np.multiply(ctrans_air,
                                                          DIST[j, ireceiver[idist_air]]) / DIST[0, mexnode]

    print("CTRANS[j, ireceiver[idist_ground], TSTART] = ", CTRANS[j, ireceiver[idist_ground], TSTART])
    print("CTRANS[j, ireceiver[idist_air], TSTART] = ", CTRANS[j, ireceiver[idist_air], TSTART])
    print("DIST[0, mexnode] = ", DIST[0, mexnode])


    if NodeTable.loc[j, 'CoastDist'] < 20 or 156 <= j <= 158:
        ireceiver = (EdgeTable['EndNodes'].str[1][EdgeTable['EndNodes'].str[0] == j]).to_numpy()
        ireceiver = ireceiver.reshape(len(ireceiver), 1)
        idist_coast = (NodeTable.loc[ireceiver[:, 0], 'CoastDist'] < 20).to_numpy().reshape(-1, 1)
        CTRANS[j, ireceiver[idist_coast], TSTART] = np.multiply(ctrans_coast,
                                                                DIST[j, ireceiver[idist_coast]]) / DIST[0, mexnode]
        if 156 <= j <= 158:
            CTRANS[j, ireceiver[idist_coast], TSTART] = 0
            CTRANS[0, j, TSTART] = CTRANS[0, j, TSTART] + np.mean(
                np.multiply(ctrans_coast, DIST[j, ireceiver[idist_coast]]) / DIST[1, mexnode])


# Initialize Interdiction agent
# Create S&L probability layer
routepref = np.zeros((nnodes, nnodes, TMAX))  # weighting by network agent of successful routes
slevent = np.zeros((nnodes, nnodes, TMAX))  # occurrence of S&L event
intrdctobs = np.zeros((nnodes, nnodes, TMAX))
slnodes = [[] for _ in range(TMAX)]
slsuccess = np.zeros((nnodes, nnodes, TMAX))  # volume of cocaine seized in S&L events
slvalue = np.zeros((nnodes, nnodes, TMAX))  # value of cocaine seized in S&L events
slcount_edges = np.zeros((1, TMAX))
slcount_vol = np.zeros((1, TMAX))
INTRDPROB = np.zeros((nnodes, TMAX))
SLPROB = np.zeros((nnodes, nnodes, TMAX))  # dynamic probability of S&L event per edge

facmat_list = [LATFAC, COASTFAC, RMTFAC, DIST / np.amax(np.amax(DIST)), BRDRFAC, SUITFAC]
facmat = np.stack(facmat_list, axis=2)
SLPROB[:, :, TSTART] = np.mean(facmat[:, :, range(0, 5)], 2)
SLPROB[:, :, TSTART + 1] = SLPROB[:, :, TSTART]
INTRDPROB[:, TSTART + 1] = slprob_0 * np.ones(nnodes)  # dynamic probability of interdiction at nodes

# Initialize Node agents
STOCK[:, TSTART] = NodeTable['Stock']
TOTCPTL[:, TSTART] = NodeTable['Capital']
PRICE[:, TSTART + 1] = PRICE[:, TSTART]
slcpcty_0 = sl_min[0, erun]
slcpcty[0, TSTART + 1] = slcpcty_0

# subjective risk perception with time distortion
twght = timewght_0 * np.ones((nnodes, 1))

# Set-up trafficking netowrk benefit-cost logic  ############
ltcoeff = locthink[0, erun] * np.ones((nnodes, 1))
margval = np.zeros((nnodes, nnodes, TMAX))
for q in range(0, nnodes):
    if len(np.where(ADJ[q, :] == 1)[0]) > 0:
        margval[q, range(q, nnodes), TSTART] = PRICE[range(q, nnodes), TSTART] - PRICE[q, TSTART]

for nd in range(0, ndto):
    idto = np.where(NodeTable['DTO'] == nd + 1)[0]
    margvalset = [idto[x] for x in range(len(idto)) if idto[x] != endnodeset]
    routepref[0, idto, TSTART + 1] = margval[0, idto, 0] / np.amax(margval[0, margvalset])

routepref[:, endnodeset, TSTART + 1] = 1
totslrisk[0, TSTART + 1] = 1

CTRANS[:, :, TSTART + 1] = CTRANS[:, :, TSTART]

# Set-up figure for trafficking movie
MOV = np.zeros((nnodes, nnodes, TMAX))

# Output tables for flows(t) and interdiction prob(t-1)
t = TSTART + 1

FLOW = scipy.io.loadmat(r'data/init_flow_ext.mat')['FLOW']

init = np.where(FLOW[:, :, 0] > 0)
rinit = init[0]
cinit = init[1]

for w in range(0, len(rinit)):
    MOV[rinit[w], cinit[w], 1] = FLOW[rinit[w], cinit[w], 1]

Tflow, Tintrd = intrd_tables_batch(FLOW, slsuccess, SLPROB, NodeTable, EdgeTable, t, erun, mrun)

mr = range(0, 31)
times = range(1, 180)
for m in mr:
    for time in times:
        print("m = ", m, ", time = ", time)
        MTMCI_func(data_sourcing(), time, m)

        intrdct_events, intrdct_nodes = optimize_interdiction_batch(ADJ)
        slevent[:, :, time] = intrdct_events
        slnodes[time].append(intrdct_nodes)
        MOV[:, 0, time] = NodeTable['Stock']

        # Iterate through trafficking nodes
        for n in range(0, nnodes):
            print("current n = ",n)
            if len(np.where(ADJ[n, :] == 1)[0]) == 0 or n == endnodeset:
                print("here")
                continue

            # Route cocaine shipments #
            STOCK[n, time] = STOCK[n, time - 1] + STOCK[n, time]
            rtdto = NodeTable.loc[np.where(ADJ[n, :] == 1)[0], 'DTO']
            if len(np.where(rtdto == 0)[0]) > 0:
                rtdto[(np.where[rtdto] == 0)[0]] = NodeTable.loc[n, 'DTO']
            CPCTY[n, np.where(ADJ[n, :] == 1)[0]] = basecap[0, erun] * rtcap[
                rtdto - 1, int(np.floor(time / 12)) + 1]
            TOTCPTL[n, time] = TOTCPTL[n, time - 1] + TOTCPTL[n, time]
            if STOCK[n, time] > 0:
                inei = []
                if n > 0:
                    LEAK[n, time] = nodeloss * STOCK[n, time]
                    STOCK[n, time] = STOCK[n, time] - LEAK[n, time]
                elif n == 0:
                    inei = np.intersect1d(np.where(ADJ[n, :] == 1)[0], np.where(routepref[n, :, time] > 0)[0])
                    for nd in range(0, len(np.unique(NodeTable.loc[1:nnodes, 'DTO']))):
                        if len(np.where(NodeTable.loc[inei, 'DTO'] == nd)) == 0:
                            idtombr = np.where(NodeTable['DTO'] == nd)[0]
                            subinei = np.intersect1d(np.where(ADJ[n, idtombr] == 1)[0],
                                                     np.where(routepref[n, idtombr, time] > 0)[0])
                            if len(subinei) == 0:
                                subinei = np.intersect1d(np.where(ADJ[n, idtombr] == 1)[0],
                                                         np.where(routepref[n, idtombr, time] ==
                                                                  np.max(routepref[n, idtombr, time]))[0])
                            np.append(inei, subinei)
                else:
                    inei = np.intersect1d(np.where(ADJ[n, :] == 1)[0], np.where(routepref[n, :, time] > 0)[0])
                    inei = inei[np.isin(inei, np.append(np.where(NodeTable['DTO'] == NodeTable.loc[n, 'DTO'])[0],
                                                        endnodeset))]
                    if len(np.where(inei != 0)[0]) == 0:
                        inei = np.intersect1d(np.where(ADJ[n, :] == 1)[0],
                                              np.where(routepref[n, :, time] == np.max(routepref[n, :, time]))[0])
                        inei = inei[np.isin(inei, np.append(np.where(NodeTable['DTO'] == NodeTable.loc[n, 'DTO'])[0]
                                                            , endnodeset))]
                '''
                check variables
                '''
                print("STOCK[n, time]: ", STOCK[n, time])
                print("inei: ", inei, time, n)
                # print('c_trans: ', c_trans)


                # Procedure for selecting routes based on expected profit #
                c_trans = CTRANS[n, inei, time].reshape(1, -1)
                p_sl = SLRISK[n, inei].reshape(1, -1)
                y_node = (PRICE[inei, time] - PRICE[n, time]).reshape(-1, 1)
                q_node = np.minimum(STOCK[n, time] / len(inei), CPCTY[n, inei]).reshape(1, -1)
                lccf = ltcoeff[n, 0]
                totstock = STOCK[n, time]
                totcpcty = CPCTY[n, inei].reshape(1, -1)
                rtpref = routepref[n, inei, time].reshape(1, -1)
                dtonei = NodeTable.loc[inei, 'DTO'].to_numpy().reshape(-1, 1)
                """ check NodeTable DTO should have 0 and 1 instead of 1 and 2 - if so remove -1 from line below """
                cutflag = dtocutflag[np.unique(dtonei[np.where(dtonei != 0)]) - 1].reshape(-1, 1)

                # !! If conditions are needed to handle empty values
                neipick, neivalue, valuex = calc_neival(c_trans, p_sl, y_node, q_node, lccf, rtpref, dtonei,
                                                        cutflag, totcpcty, totstock, edgechange)

                neipick = np.int_(neipick)
                # With top-down route optimization
                inei = inei[neipick]                                                                                        #### !!!! Empty neipick!!!
                # weight according to salience value function
                if len(np.where(valuex <= 0)) > 0:
                    WGHT[n, inei] = (1 - SLRISK[n, inei]) / np.sum(1 - SLRISK[n, inei])
                else:
                    WGHT[n, inei] = np.transpose(
                        np.maximum(valuex[neipick], 0) / np.sum(np.amax(valuex[neipick], 0)))

                activeroute[n, time] = inei
                FLOW[n, inei, time] = np.minimum(np.multiply(WGHT[n, inei] / np.sum(WGHT[n, inei]), STOCK[n, time]),
                                                 CPCTY[n, inei])

                OUTFLOW[n, time] = np.sum(FLOW[n, inei, time])
                STOCK[n, time] = STOCK[n, time] - OUTFLOW[n, time]
                nodecosts = np.sum(np.multiply(FLOW[n, inei, time], CTRANS[n, inei, time]))

                # Check for S#L event
                if np.any(np.isin(np.where(slevent[n, :, time] != 0)[0], inei)):
                    isl = np.where(slevent[n, inei, time] == 1)
                    intrdctobs[n, inei[isl], time] = 1
                    intcpt = np.minimum(p_sucintcpt[erun] * NodeTable.loc[inei[isl], 'pintcpt'], 1)

                    # interception probability
                    p_int = np.random.rand(len(intcpt), 1)
                    for p in range(len(intcpt)):
                        if p_int[p] <= intcpt[p]:
                            slsuccess[n, inei[isl[p]], time] = FLOW[n, inei[isl[p]], time]
                            slvalue[n, inei[isl[p]], time] = np.multiply(FLOW[n, inei[isl[p]], time],
                                                                         np.transpose(PRICE[inei[isl[p]], time]))
                            FLOW[n, inei[isl[p]], time] = 0
                        else:
                            slsuccess[n, inei[isl[p]], time] = 0
                            slvalue[n, inei[isl[p]], time] = 0

                    STOCK[inei, time] = STOCK[inei, time] + np.transpose(FLOW[n, inei, time])
                    noderevenue = np.sum(np.multiply(FLOW[n, inei, time], np.transpose(PRICE[inei, time])))
                    TOTCPTL[inei, time] = TOTCPTL[inei, time] - (np.multiply(np.transpose(FLOW[n, inei, time]),
                                                                             PRICE[inei, time]))
                    ICPTL[n, time] = rentcap * np.sum(np.multiply(FLOW[n, inei], ADDVAL[n, inei].reshape(-1, 1))) # need to reshape!!!

                    MARGIN[n, time] = noderevenue - nodecosts + min(TOTCPTL[n, time], 0)
                    if n > 1:
                        BRIBE[n, time] = max(bribepct * MARGIN[n, time], 0)
                        if MARGIN[n, time] > 0:
                            RENTCAP[n, time] = MARGIN[n, time] - BRIBE[n, time]
                        else:
                            RENTCAP[n, time] = MARGIN[n, time]
                        TOTCPTL[n, time] = max(TOTCPTL[n, time], 0) + RENTCAP[n, time]
                    else:
                        RENTCAP[n, time] = MARGIN[n, time]
                        TOTCPTL[n, time] = TOTCPTL[n, time] + RENTCAP[n, time]
                else:
                    # STOCK[inei, time] = STOCK[inei, time] + np.transpose(FLOW[n, inei, time])
                    STOCK[inei, time] = STOCK[inei, time] + FLOW[n, inei, time] # Hashir: STOCK[inei, time]: (2, 9); FLOW[n, inei, time]: (2, 9)
                    nodecosts = np.sum(np.multiply(FLOW[n, inei, time], CTRANS[n, inei, time]))
                    # noderevenue = np.sum(np.multiply(FLOW[n, inei, time], np.transpose(PRICE[inei, time])))
                    noderevenue = np.sum(np.multiply(FLOW[n, inei, time], PRICE[inei, time])) # Hashir: FLOW[n, inei, time]: (2, 9), PRICE[inei, time]: (2, 9)
                    # TOTCPTL[inei, time] = TOTCPTL[inei, time] - np.multiply(np.transpose(FLOW[n, inei, time]), PRICE[inei, time])
                    TOTCPTL[inei, time] = TOTCPTL[inei, time] - np.multiply(FLOW[n, inei, time], PRICE[inei, time]) # Hashir: removed np.transpose(..)
                    ICPTL[n, time] = rentcap * np.sum(np.multiply(FLOW[n, inei], ADDVAL[n, inei, np.newaxis])) # need to reshape/transpose!!!
                    MARGIN[n, time] = noderevenue - nodecosts + min(TOTCPTL[n, time], 0)
                    if n > 1:
                        BRIBE[n, time] = max(bribepct * MARGIN[n, time], 0)
                        if MARGIN[n, time] > 0:
                            RENTCAP[n, time] = MARGIN[n, time] - BRIBE[n, time]
                        else:
                            RENTCAP[n, time] = MARGIN[n, time]
                        TOTCPTL[n, time] = max(TOTCPTL[n, time], 0) + RENTCAP[n, time]
                    else:
                        RENTCAP[n, time] = MARGIN[n, time]
                        TOTCPTL[n, time] = TOTCPTL[n, time] + RENTCAP[n, time]

                # Update perceived risk in response to S&L and Interdiction events
                timeweight = twght[n]

                # identify neighbors in network (without network toolbox)
                fwdnei = inei
                t_eff = np.arange(0, 12)
                if t == TSTART + 1:
                    # Risk perception only updated when successful interdiction takes place
                    sloccur = np.array([[np.zeros((12, len(fwdnei)))], [(slsuccess[n, fwdnei, TSTART + 1] > 0).astype(np.uint)]])
                elif t > TSTART + 1 and len(fwdnei) == 1:
                    sloccur = np.array([[np.zeros((13 - len(np.arange(np.amax(TSTART + 1, time - 12), time + 1)),
                                                   1))],
                                        [np.squeeze(slsuccess[n, fwdnei, np.arange(np.amax(TSTART + 1, time - 12),
                                                                                   time + 1)] > 0)]])
                else:
                    sloccur = np.array([[np.zeros((13 - len(np.arange(np.amax(TSTART + 1, time - 12), time + 1)),
                                                   len(fwdnei)))],
                                        [np.transpose(np.squeeze(slsuccess[n, fwdnei,
                                        np.arange(np.amax(TSTART + 1, time - 12), time + 1)] > 0))]])

                sl_risk, slevnt, tmevnt = calc_intrisk(sloccur, t_eff, alpharisk, betarisk, timeweight)         # SLRISK[n, fwdnei] = sl_risk # sl_risk:shape(1, 12) --> put into SLRISH[n, fwdnei] = (2,)
                SLRISK[n, fwdnei] = np.sum(np.sum(sl_risk[0]), axis = 0)
                # SLRISK[n, fwdnei] = np.mean(np.mean(sl_risk[0]), axis = 0)

                if len(np.where(sl_risk != 0)) > 0:
                    avgslrisk[n, time] = SLRISK[n, activeroute[n, time]]

                NodeTable['Stock'] = STOCK[:, time]
                NodeTable['Capital'] = TOTCPTL[:, time]
                RISKPREM[:, :, time] = np.amax(np.multiply((1 - delta_rt), RISKPREM[:, :, time - 1]) + np.multiply(
                    delta_rt, ((SLRISK / baserisk[0][erun]) ** riskmltplr[0][erun])), 1) # both baserick & riskmltplr are in [1, 11] shape

                # Make trafficking move
                MOV[:, n, time] = STOCK[:, time]

        # Risk premium on cost of doing business (transport costs)
        CTRANS[:, :, time + 1] = np.multiply(CTRANS[:, :, time], RISKPREM[:, :, time])
        totslrisk[time + 1] = np.mean(np.concatenate(2, avgslrisk[:, time]))

        # Calculate updated marginal profit
        for q in range(0, nnodes):
            if len(np.where(ADJ[q, :] == 1)) == 0:
                margval[q, q + 1: nnodes, time] = PRICE[q + 1: nnodes, time] - PRICE[q, time]

        # Route Optimization ###########
        for dt in range(0, ndto):
            idto = np.where(NodeTable['DTO'] == dt)[0]
            DTOBDGT[dt, time] = np.sum(np.multiply(STOCK[endnodeset, time], PRICE[endnodeset, time]))  # total DTO
            # funds for expansion/viability
            dtorefvec = np.array([[1], [idto], [mexnode]])
            subroutepref = routepref[dtorefvec, dtorefvec, time]
            subflow = FLOW[dtorefvec, dtorefvec, time]
            dtoslsuc = slsuccess[dtorefvec, dtorefvec, time]
            allflows = subflow + dtoslsuc
            # locate active edges
            indices = np.where(allflows > 0)
            irow = indices[0]
            dtoEdgeTable = EdgeTable.loc[:, 'sendedge']
            dtoEdgeTable = dtoEdgeTable[np.isin(dtoEdgeTable['EndNodes'].str[1], dtorefvec), :]
            dtoSLRISK = SLRISK[dtorefvec, dtorefvec]
            dtoADDVAL = margval[dtorefvec, dtorefvec, time]
            dtoCTRANS = CTRANS[dtorefvec, dtorefvec, time]

            # calculate losses from S&L events
            # volume-based - does not matter where in supply chain
            ipossl = np.where(dtoslsuc > 0)[0]
            nrow = ipossl[0]
            ncol = ipossl[1]
            flowvalues = np.multiply(allflows[allflows > 0], (
                    (PRICE[dtorefvec[ncol], time] - PRICE[dtorefvec[irow], time]) - dtoCTRANS[allflows > 0]))
            supplyfit = np.sum(np.multiply(dtoslsuc[ipossl], (
                    (PRICE[dtorefvec[ncol], time] - PRICE[dtorefvec[nrow], time]) - dtoCTRANS[ipossl])))
            losstolval = losstol * np.maximum(flowvalues)
            if len(np.where(supplyfit != 0)[0]) == 0 and len(np.where(losstolval != 0)[0]) == 0:
                supplyfit = 0.1

            # Route capacity constrains flow volumes, need to expand routes
            idtonet = np.delete(dtorefvec, np.in1d(dtorefvec, endnodeset))

            if np.sum(STOCK[idtonet, time]) >= np.amax(dtoEdgeTable['Capacity']):
                supplyfit = max(supplyfit, losstolval * np.sum(STOCK[idtonet, time]) / rtcap[erun])

            # call top-down route optimization
            expmax = expandmax[erun]

            newroutepref, newedgechange = optimizeroute_multidto(dtorefvec, subflow, supplyfit, expmax,
                                                                 subroutepref, dtoEdgeTable, dtoSLRISK, dtoADDVAL,
                                                                 dtoCTRANS, losstolval, dtoslsuc)

            edgechange[dt] = newedgechange

            # Bottom-up route optimization
            routepref[dtorefvec, dtorefvec, time + 1] = newroutepref

        PRICE[:, time + 1] = PRICE[:, time]

        if growthmdl[erun] == 1:
            STOCK[0, time] = stock_0 + (prodgrow[erun] * np.ceil((time - TSTART) / 12))  # additional production to
            # enter network next time step
        elif growthmdl[erun] == 2:
            STOCK[0, time] = (stock_max * stock_0 * np.exp(prodgrow[erun] * int(np.floor(time / 12)))) / (
                    stock_max + stock_0 * (np.exp(prodgrow[erun] * int(np.floor(time / 12))) - 1))

        STOCK[endnodeset, time + 1] = 0
        NodeTable.loc[1, 'Stock'] = STOCK[1, time + 1]
        NodeTable[endnodeset, 'Stock'] = 0
        slcount_edges[time] = len(np.where(slsuccess[:, :, time] > 0)[0])
        h_slsuccess = slsuccess[:, :, time]
        slcount_vol[time] = np.sum(h_slsuccess[h_slsuccess > 0])

        # Output tables for flows(t) and interdiction prob(t-1)
        Tflow, Tintrd = intrd_tables_batch(FLOW, slsuccess, SLPROB, NodeTable, EdgeTable, t, erun, m)
        data_processing(Tflow, time, m)


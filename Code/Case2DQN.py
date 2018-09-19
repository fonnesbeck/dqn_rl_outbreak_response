#!/usr/bin/env python
#Date: 2/8/18
#Author: Sandya Lakkur 
#scp /Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/PickNextBextFarm/ACCREStuff/3_16_18/AtariParams_updatetarget_8kEp_2500deque2_DC5_15x15_3_3_18.py lakkurss@login.accre.vanderbilt.edu:~/simulations/SpatiallyExplicitDecisions/scripts
#scp /Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/PickNextBextFarm/ACCREStuff/3_16_18/AtariParams_updatetarget_8kEp_2500deque2_DC5_15x15_3_3_18.slurm lakkurss@login.accre.vanderbilt.edu:~/simulations/SpatiallyExplicitDecisions/slurm_scripts

from scipy import spatial
from scipy.stats import rankdata
import numpy as np
import itertools
import heapq
import random
import copy
from copy import deepcopy
from random import gauss
from math import pi
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
import tensorflow as tf
import pickle
import time 
#import matplotlib
#matplotlib.use('TkAgg') #include this line so matplotlib.pyplot will run in a virtual enviornment
#import matplotlib.pyplot as plt

results_path = "/home/lakkurss/simulations/SpatiallyExplicitDecisions/results/"

#I WANT TO TRY REDUCING THE EPISODES OF TRAINING AND THE SIZE OF THE DEQUE, BASED ON THE RESULTS FROM THE SAME CODE IN THE 3/13/18 FOLDER.

    #THIS DID WELL, ABOUT 69 FARMS WERE CULLED. BUT THE REWARDS WERE UP TO 10K, I THINK IT COULD GO HIGHER.
    
EPISODES = 8000

def CoordsAndGridSquares(N,numsquares):
    ######################
    #Inputs:
    #     N = number of farms
    #     numsquares = number of grid squares in 1-d
    #Outputs:
    #     setlist = coordinates of grid square point is in 
    #     xcoord = x-coordinate of farm
    #     ycoord = y-coordinate of farm
    #     gridsquare = grid square (scalar) the farm is in
    ######################
    
    #Generate set of N tuples without replacement
    ret = set()
    while len(ret) < N:
        ret.add((random.randint(1, numsquares), random.randint(1, numsquares)))
    
    #Convert set to list
    setlist = list(ret)
    
    #Create grid indicies in matrix form
    listofsquares = np.arange(1,(numsquares**2)+1)
    matrixofsquares = listofsquares.reshape(numsquares,numsquares)
    
    #Initialize lists
    xcoord = []
    ycoord = []
    gridsquare = []
    for i in range(N):
        #Generate x and y coords
        xcoord.append(np.random.uniform(low=setlist[i][0]-1, high=setlist[i][0]))
        ycoord.append(np.random.uniform(low=setlist[i][1]-1, high=setlist[i][1]))
        #Determine grid square the coordinate belongs to
        gridsquare.append(matrixofsquares[numsquares - setlist[i][1],setlist[i][0]-1])
    return(np.c_[setlist,xcoord,ycoord,gridsquare])
	
def GenerateAnimals(N):
    #np.random.seed(42017)
    Cows = np.ceil(np.random.uniform(24,500,N))
    return(Cows)
    
######################
Size = 15
N = 120
dailyCapacity = 5
outbreak_thres = 0
ALPHA = 0.5
epsilon_frames = 1000000
epsilon_vector = list(np.arange(0.01,1, (1-0.01)/epsilon_frames))
#epsilon = 1
#epsilon_decay = 0.999
epsilon_min = 0.01
gamma = 1
coords = np.load('/home/lakkurss/simulations/SpatiallyExplicitDecisions/data/coords15x15_3_3_18.npy')
x = coords[:,2]
y = coords[:,3]
gridsquare = coords[:,4]
Cows = np.load('/home/lakkurss/simulations/SpatiallyExplicitDecisions/data/Cows15x15_3_3_18.npy')

###################### 

# USE THIS FUNCTION TO CALCULATE WHICH GRID SQUARE A FARM IS IN
def WhichGrid(x,y,XRange,YRange,XNum,YNum):
    #Essentially: floor(Unif[0,1)griddim)griddim+floor(Unif[0,1)griddim)+1
    #Returns a number from 1 to griddim^2
    return(np.floor(x*(XNum/XRange))*YNum+np.floor(y*(YNum/YRange))+1)
	
# USE THIS FUNCTION TO FIND THE KERNEL VALUE FOR DISTANCE
def Kernel(dist_squared):
    dist_squared = np.asarray(dist_squared)
    is_scalar = False if dist_squared.ndim > 0 else True
    dist_squared.shape = (1,)*(1-dist_squared.ndim) + dist_squared.shape
    K = 1 / (pi * (1 + dist_squared**2))
    K[(dist_squared < 0.0138)] = 0.3093
    K[(dist_squared > 60*60)] = 0
    return(K if not is_scalar else K[0])

# USE THIS FUNCTION TO GENERATE THE DISEASE PARAMETERS OF THE OUTBREAK
def GenOutbreakParams(Size,N,x,y,Cows):
    #This is an attempt of converting the Matlab Program 7.6 Code into Python
    #Cows are 10.5 times more susceptible to disease than sheep
    Suscept = Cows
    Transmiss = 49e-6*Cows 

    #Set up the grid
    grid = WhichGrid(x,y,Size,Size,10.0,10.0)
    ranks = rankdata(grid,method="dense") #need to do this because the grid numbers are not necessarily consecutive    
    tmp = sorted(grid) #Sort grid values
    #i = np.argsort(grid) #get indexed values after sort
    i = [i[0] for i in sorted(enumerate(grid), key=lambda x:x[1])]
    x = x[i]
    y = y[i]
    grid = grid[i]
    ranks = ranks[i]
    Transmiss = Transmiss[i]
    Suscept = Suscept[i]
    Cows = Cows[i]
    Xgrid = []
    Ygrid = []
    Num = []
    first_in_grid = []
    last_in_grid = []
    Max_Sus_grid = []
    m2 = np.array(np.where(grid==1))

    unique_grid = np.unique(grid)
    grid_ints = unique_grid.astype(np.int64)

    for i in range(len(grid_ints)):
        #turn the grid square number into an x-coordinate and y-coordinate (should not exceed XNum)
        Xgrid.append(np.floor((grid_ints[i]-1)/10))
        Ygrid.append((grid_ints[i]-1)%10)
        m = np.array(np.where(grid==grid_ints[i]))
        Num.append(m.shape[1])
        
        if Num[i-1] > 0:
            first_in_grid.append(m.min()) #Add the "+1" here so the indicies match those in the Keeling code
            last_in_grid.append(m.max())
            Max_Sus_grid.append(Suscept[m].max())
        else:
            first_in_grid.append(0)
            last_in_grid.append(-1)
            Max_Sus_grid.append(0)

    #Work out grid to maximum grid transmission probabilities
    from numpy import ndarray
    MaxRate = ndarray((len(grid_ints),len(grid_ints)))
    
    #Determine maximum number of animals to be infected in each grid square

    for i in range (1,len(grid_ints)):
        for j in range(1,len(grid_ints)):
            if ((i-1)==(j-1)) | (Num[i-1]==0) | (Num[j-1] == 0):
                MaxRate[i-1,j-1] = np.inf
            else:
                Dist2 = (Size*max([0,(abs(Xgrid[i-1]-Xgrid[j-1])-1)])/10)**2+(Size*max([0,(abs(Ygrid[i-1]-Ygrid[j-1])-1)])/10)**2
                MaxRate[i-1,j-1] = Max_Sus_grid[j-1]*Kernel(Dist2)
    return([Suscept,Transmiss,ranks,MaxRate,Num,last_in_grid,first_in_grid,grid])

# USE THIS FUNCTION TO EVOLVE THE OUTBREAK
def Iterate(asarray,OutbreakParams, x, y):
    Status = asarray
    Suscept = OutbreakParams[0]
    Transmiss = OutbreakParams[1]
    ranks = OutbreakParams[2]
    grid = OutbreakParams[7]
    first_in_grid = OutbreakParams[6]
    last_in_grid = OutbreakParams[5]
    Num = OutbreakParams[4]
    MaxRate = OutbreakParams[3]
    Event = 0*Status
    INF = np.where(Status>5)[0]
    NI = INF.size # Note reported farms still infectious
    IGrids = ranks[INF]-1
        
    for ii in range(NI):
        INFi = INF[ii]
        trans = np.multiply(-Transmiss[INFi],Num) #transmissibility of infected farm to all other grid squares 
        maxr = MaxRate[int(IGrids[ii])-1,:] #max number of animals to be infected in infected grid square
        # Elementwise multiplication
        rate = np.multiply(trans, maxr) #max number of animals to be infected in each grid square based on infected grid square
        MaxProb = 1 - np.exp(rate) #Max probability that infected farm infected noninfected farm
        rng = np.random.rand(len(MaxProb))
        m = np.where((MaxProb - rng)>0)[0]  #these grid squares need further consideration
        for n in range(len(m)):
            s = 1
            M = m[n]
            PAB = 1 - np.exp(-Transmiss[INFi]*MaxRate[int(IGrids[ii]),M]) #Max probability that infected farm infects noninfected farms under consideration
            
            ##FOR NOW YOU CAN CONVERT ANY PAB = 0 TO SOMETHING A LITTLE GREATER THAN ZERO
            ##I THINK THIS QUANITTY IS ZERO BECAUSE THE GRID AND THEREFORE THE DISTANCES ARE SO SMALL
            ##NEED TO CHECK IF THIS IS STILL THE CASE WHEN YOU MAKE A LARGER GRID
            #if PAB < 0.00000000000000001:
            #    PAB = 0.0001
            
            if (PAB == 1):
                # Calculate the infection probability for each farm in the susceptible grid
                leng = last_in_grid[M]-first_in_grid[M]+1
                R = np.random.rand(leng)
                for j in range(leng):
                    ind1 = first_in_grid[M]+j-1
                    Q = 1 - np.exp(-Transmiss[INFi]*Suscept[ind1]*Kernel((x[INFi]-x[ind1])**2+(y[INFi]-y[ind1])**2))
                    if ((R[j] < Q) & (Status[ind1] == 0)):
                        Event[ind1] = 1
            else:
                R = np.random.rand(Num[M])
                # Loop through all susceptible farms in the grids where an infection event occurred.  
                for j in range(Num[M]):
                    P = 1 - s*(1 - PAB)**(Num[M] - j)
                    
                    if (R[j] < (PAB / P)):
                        s = 0
                        ind1=first_in_grid[M]+j-1
                        Q=1-np.exp(-Transmiss[INFi]*Suscept[ind1]*Kernel((x[INFi]-x[ind1])**2+(y[INFi]-y[ind1])**2))
                        if ((R[j]< Q/P) & (Status[ind1] == 0)):
                            Event[ind1] = 1
    # Evolve the infection process of those farms which have been exposed and already infectious ones.  
    Status[Status > 0] += 1
    Status = Status + Event
    #Status[Status>=1] = 1 #For now we are not worried about exposed farms, just make status 1,0,-1
    
    #m=np.where(Status==13); # Initiate Ring Culling Around Reported Farm
    #for i in range(len(m)):
    #    Status[m[i]]=-1;
    return(Status)
    
# USE THIS FUNCTION TO EVOLVE THE OUTBREAK, NO ACTIONS NECESSARY
def justEvolve(asarray,OutbreakParams, x, y):
    Status = asarray
    Suscept = OutbreakParams[0]
    Transmiss = OutbreakParams[1]
    ranks = OutbreakParams[2]
    grid = OutbreakParams[7]
    first_in_grid = OutbreakParams[6]
    last_in_grid = OutbreakParams[5]
    Num = OutbreakParams[4]
    MaxRate = OutbreakParams[3]
    Event = 0*Status
    INF = np.where(Status>5)[0]
    NI = INF.size # Note reported farms still infectious
    IGrids = ranks[INF]-1
        
    for ii in range(NI):
        INFi = INF[ii]
        trans = np.multiply(-Transmiss[INFi],Num) #transmissibility of infected farm to all other grid squares 
        maxr = MaxRate[int(IGrids[ii])-1,:] #max number of animals to be infected in infected grid square
        # Elementwise multiplication
        rate = np.multiply(trans, maxr) #max number of animals to be infected in each grid square based on infected grid square
        MaxProb = 1 - np.exp(rate) #Max probability that infected farm infected noninfected farm
        rng = np.random.rand(len(MaxProb))
        m = np.where((MaxProb - rng)>0)[0]  #these grid squares need further consideration
        for n in range(len(m)):
            s = 1
            M = m[n]
            PAB = 1 - np.exp(-Transmiss[INFi]*MaxRate[int(IGrids[ii]),M]) #Max probability that infected farm infects noninfected farms under consideration
            
            ##FOR NOW YOU CAN CONVERT ANY PAB = 0 TO SOMETHING A LITTLE GREATER THAN ZERO
            ##I THINK THIS QUANITTY IS ZERO BECAUSE THE GRID AND THEREFORE THE DISTANCES ARE SO SMALL
            ##NEED TO CHECK IF THIS IS STILL THE CASE WHEN YOU MAKE A LARGER GRID
            #if PAB < 0.00000000000000001:
            #    PAB = 0.0001
            
            if (PAB == 1):
                # Calculate the infection probability for each farm in the susceptible grid
                leng = last_in_grid[M]-first_in_grid[M]+1
                R = np.random.rand(leng)
                for j in range(leng):
                    ind1 = first_in_grid[M]+j-1
                    Q = 1 - np.exp(-Transmiss[INFi]*Suscept[ind1]*Kernel((x[INFi]-x[ind1])**2+(y[INFi]-y[ind1])**2))
                    if ((R[j] < Q) & (Status[ind1] == 0)):
                        Event[ind1] = 1
            else:
                R = np.random.rand(Num[M])
                # Loop through all susceptible farms in the grids where an infection event occurred.  
                for j in range(Num[M]):
                    P = 1 - s*(1 - PAB)**(Num[M] - j)
                    
                    if (R[j] < (PAB / P)):
                        s = 0
                        ind1=first_in_grid[M]+j-1
                        Q=1-np.exp(-Transmiss[INFi]*Suscept[ind1]*Kernel((x[INFi]-x[ind1])**2+(y[INFi]-y[ind1])**2))
                        if ((R[j]< Q/P) & (Status[ind1] == 0)):
                            Event[ind1] = 1
    # Evolve the infection process of those farms which have been exposed and already infectious ones.  
    Status[Status > 0] += 1
    Status = Status + Event
    #Status[Status>=1] = 1 #For now we are not worried about exposed farms, just make status 1,0,-1
    
    #m=np.where(Status==13); # Initiate Ring Culling Around Reported Farm
    #for i in range(len(m)):
    #    Status[m[i]]=-1;
    return(Status)
    
def chooseAction(currentState,epsilon,original_farm_inds):
    #print([len(original_farm_inds),len(currentState[original_farm_inds])])
    farm_mat = np.c_[list(range(N)),original_farm_inds,currentState[original_farm_inds]]
    sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
    ##Choose your action
    if np.random.rand() <= epsilon:
        #print("in if")
        #Only cull farms that are farms or not culled
        currentAction = np.random.choice(sub_farm_mat[:,0])
        return(currentAction)
    else:
        #print("in else")
        mat = currentState.reshape(Size,Size,1)
        #print([mat,np.array([mat]).shape])
        prediction = model3.predict(np.array([mat])) #put the mat in a list so the dimesion becomes (1,Size,Size,1) a 4-d tensor, consistent with Input shape
        #print("Model.predict in chooseAction function: %s" %prediction)
        #print([sub_farm_mat[:,0],prediction[0][sub_farm_mat[:,0]]])
        #print([sub_farm_mat[:,0],type(prediction),type(sub_farm_mat[:,0]),prediction.shape])
        currentAction_ind = np.argmax(prediction[0][sub_farm_mat[:,0]])
        currentAction = sub_farm_mat[:,0][currentAction_ind]
        return(currentAction)


# pick samples randomly from replay memory (with batch_size)
def train_replay(batch_size, discount_factor,state_size):
    if len(memory) < train_start:
        #print("Length of memory is: %d" %len(memory))
        return
    #if len(self.memory) > self.memmaxlen:
    #    self.memory = self.memory
    batch_size = min(batch_size, len(memory))
    mini_batch = random.sample(list(memory), batch_size)
    #inds = random.sample(list(range(len(memory))),batch_size)
    #print(inds)
    #mini_batch = list(memory)[inds]
    
    update_input, update_target,action, reward, done, inds_to_cull = [], [], [], [], [], []
    
    for i in range(batch_size):
        update_input.append(mini_batch[i][0])
        action.append(mini_batch[i][1])
        reward.append(mini_batch[i][2])
        #update_target[i] = mini_batch[i][3]
        done.append(mini_batch[i][3])
    
    
        if len(mini_batch[i]) == 6:
            update_target.append(mini_batch[i][4])
            inds_to_cull.append(mini_batch[i][5])
    
    update_input = np.stack(update_input,axis=0)
    update_target = np.stack(update_target,axis=0)      
    target = model3.predict(update_input)
    #for i in range(batch_size):
    #    print([mini_batch[i][0],target[i,:]])
    target_val = target_model3.predict(update_target) 
    
    
    else_counter2 = 0
    for i in range(batch_size):
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done[i] == 1:
            target[i][action[i]] = reward[i]
        else:
            target[i][action[i]] = reward[i] + discount_factor * np.amax(target_val[else_counter2][inds_to_cull[else_counter2]])
            else_counter2 += 1
    model3.fit(update_input, target, batch_size=batch_size, epochs=1, verbose=0)

# after some time interval update the target model to be same with model
def update_target_model(model3weights):
    target_model3.set_weights(model3weights)
      
def AnimateOutbreak(coords, currentStatus, tag, culled_farms,cull_num,string):
    fig, ax = plt.subplots()
    x = coords[:,2]
    y = coords[:,3]
    inf = currentStatus
    colors=["#00FF00","#FF0066","#0000FF","000000"]
    color_list = []
    for i in range(len(Cows)):
        if inf[i] == 0:
            color_list.append(colors[0])
        elif inf[i] > 5:
            color_list.append(colors[1])
        elif 0 < inf[i] <= 5:
            color_list.append(colors[2])
        else:
            color_list.append(colors[3])
    
    labels = np.arange(0,len(Cows))
    for i in range(len(Cows)):
        ax.scatter(x[i],y[i],color = color_list[i], s = 50)
        ax.annotate(Cows[i].astype(int), (x[i],y[i]))
    
    if tag == "Initial State":
        #Add the gridlines just to make sure you are not putting multiple farms in a grid square
        for i in range(1,10): 
            ax.plot(np.repeat(i,11),list(range(11)),color="black")
        for i in range(1,10):
            ax.plot(list(range(11)), np.repeat(i,11),color = "black")
        plt.title("Initial State")
        plt.show()
    
    else:
        for i in range(len(Cows)):
            ax.scatter(x[i],y[i],color = color_list[i], s = 50)
            ax.annotate(Cows[i].astype(int), (x[i],y[i]))
        
        #ax.scatter(coords[culled_farms,2],coords[culled_farms,3],marker = 'x',color = 'black', linewidths = 2)
            
        plt.title("Cull %d, %s!" %(cull_num, string))
        plt.show()
    
#####################################################   
#Load Outbreak Parameters
with open('/home/lakkurss/simulations/SpatiallyExplicitDecisions/data/OutbreakParams15x15_3_3_18.txt', "rb") as fp: # Unpickling
    OutbreakParams = pickle.load(fp)

memory = deque(maxlen=2500)
train_start = 100
batch_size = 32
discount_factor = 0.99
state_size = Size**2
sum_stop = 80 #number of consecutive episodes where the agent acts in the best way, in order to stop training
sum_stop_array = np.zeros((sum_stop,), dtype=np.int) #have a filler array with"sum_stop" number of zeros

#Start the clock
starttime = time.process_time()

inp = Input(batch_shape=(None,Size, Size, 1)) #Specify "None" as the first argument so it won't matter how many samples are in the batch
conv_1 = Convolution2D(32, (3, 3), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(64, (3, 3), padding='same', activation='relu')(conv_1)
conv_3 = Convolution2D(64, (3, 3), padding='same', activation='relu')(conv_2)
flat = Flatten()(conv_3)
hidden = Dense(155, activation='relu')(flat)
out = Dense(N, activation='linear')(hidden)
model3 = Model(inputs=inp, outputs=out)
model3.compile(loss='mse',optimizer=RMSprop(lr=0.0001))

target_model3 = Model(inputs=inp, outputs=out)
target_model3.compile(loss='mse',optimizer=RMSprop(lr=0.0001))

time_list = np.empty([EPISODES,1])
reward_list = []

#Read in original_pixelarray and original_currentStatus
original_pixelarray = np.load('/home/lakkurss/simulations/SpatiallyExplicitDecisions/data/original_pixelarray15x15_3_3_18.npy')
original_currentStatus = np.load('/home/lakkurss/simulations/SpatiallyExplicitDecisions/data/original_currentStatus15x15_3_3_18.npy')
farm_inds = np.where(original_pixelarray != 2)[0]

#Choose the first epsilon from the list
epsilon = epsilon_vector.pop()

for e in range(EPISODES): 
    #INITIALIZE SOME DIAGNOSTIC TOOLS
    time_count = 0 
    done = False
    num_culls = 0
    reward_counter = 0
    
    print("#### Episode: %d ####" %e)
    #Re-set the "board" to your original pixelarray and status
    pixelarray = original_pixelarray
    currentStatus = original_currentStatus
    #print("Initial number of infected: %d" %(np.sum(pixelarray == 1)))	
    #print("Initial status is: %s" %currentStatus)
    
    #Begin iterations
    while np.sum(pixelarray == 1) > outbreak_thres:
        #print("## NumCulled: %d, Timestep: %d, Episode: %d ##" %(num_culls,time_count, e))
        #print("Current status is: %s" %currentStatus)
        #print("Current #Infected: %d" %np.sum(pixelarray == 1))
        
        print("Epsilon is: %0.4f" %epsilon)
        
        #Choose action
        placeholder = chooseAction(pixelarray,epsilon,farm_inds) 
        num_culls += 1
        #print("Cull farm: %d" %placeholder)
        
        #Initialize the new state
        newStatus = currentStatus.copy()
        newPixelArray = pixelarray.copy()
    
        #If you haven't reached daily capacity, cull the farm
        if num_culls < dailyCapacity:
            #print("Under daily capacity")
            #Cull the chosen farm
            newPixelArray[farm_inds[placeholder]] = 2
            newStatus[placeholder] = -1
        
            #Generate rewards
            reward = 0
            #Penalize for culling an already culled farm
            #if pixelarray[farm_inds[placeholder]] == 2:
            #    print("Reward before")
            #    print("The reward is: %d" %reward)
            #    reward = reward -1000000
            #    print("The reward is: %d" %reward)
            #    print("You culled a culled farm!")
            
            #Penalize for outbreak still continuing
            if np.sum(newPixelArray == 1) > outbreak_thres:
                #print("Outbreak still continuing")
                #print("Reward before")
                #print("The reward is: %d" %reward)
                reward = reward -100
                #print("The reward is: %d" %reward)
        
                #Save sample <s,a,r,done> to replay memory
                done = 0
                farm_mat = np.c_[list(range(N)),farm_inds,newPixelArray[farm_inds]]
                sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                not_culled_farm_inds = sub_farm_mat[:,0]
                #print("Farms not culled: %s" %not_culled_farm_inds)
                mat = np.array(pixelarray.reshape(Size,Size,1))
                newMat = np.array(newPixelArray.reshape(Size,Size,1))
                memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                #print("Start the replay")
                #train_replay(batch_size,discount_factor,state_size)
                
                #Make currentState the newState
                #print("Update State")
                currentStatus = newStatus
                pixelarray = newPixelArray
                
                inf_farms = np.where(newPixelArray == 1)[0]
                not_inf_farms = np.where(newPixelArray == 0)[0]
                inds = np.r_[inf_farms,not_inf_farms]
                
                #Update epsilon
                if len(epsilon_vector) > 1:
                    epsilon = epsilon_vector.pop()
                else:
                    epsilon = epsilon_min
        
            #If outbreak is done then reward based on number of uninfected farms still standing
            if np.sum(newPixelArray == 1) == outbreak_thres:
                print("I think outbreak is finished")
                
                #Evolve the outbreak for 6 timesteps to check if done, don't give a reward or fit model while you wait
                new2 = justEvolve(newStatus, OutbreakParams, x, y)
                new_inf_inds = np.where(new2 > 5)[0]
                new2PixelArray = newPixelArray.copy()
                new2PixelArray[farm_inds[new_inf_inds]] = 1
                if np.sum(new2 > 5) == 0:
                    print("No infecteds for 2 days")
                    new3 = justEvolve(new2, OutbreakParams, x, y)
                    new_inf_inds = np.where(new3 > 5)[0]
                    new3PixelArray = new2PixelArray.copy()
                    new3PixelArray[farm_inds[new_inf_inds]] = 1
                    if np.sum(new3 > 5) == 0:
                        print("No infecteds for 3 days")
                        new4 = justEvolve(new3, OutbreakParams, x, y)
                        new_inf_inds = np.where(new4 > 5)[0]
                        new4PixelArray = new3PixelArray.copy()
                        new4PixelArray[farm_inds[new_inf_inds]] = 1
                        if np.sum(new4 > 5) == 0:
                            print("No infecteds for 4 days")
                            new5 = justEvolve(new4, OutbreakParams, x, y)
                            new_inf_inds = np.where(new5 > 5)[0]
                            new5PixelArray = new4PixelArray.copy()
                            new5PixelArray[farm_inds[new_inf_inds]] = 1
                            if np.sum(new5 > 5) == 0:
                                print("No infecteds for 5 days")
                                new6 = justEvolve(new5, OutbreakParams, x, y)
                                new_inf_inds = np.where(new6 > 5)[0]
                                new6PixelArray = new5PixelArray.copy()
                                new6PixelArray[farm_inds[new_inf_inds]] = 1
                                if np.sum(new6 > 5) == 0:
                                    print("No infecteds for 6 days")
                                    warning = 6
                                    #CONSTRCUT THE TERMINAL REWARD BASED ON NUMBER OF NONINFECTED COWS STILL STANDING 
                                    #num_alive = np.sum(newPixelArray == 0)
                                    farms_alive= np.where(np.logical_and(new6 >= 0,new6 <= 5))[0]
                                    #print([farms_alive,Cows[farms_alive]])
                                    cows_alive = np.sum(Cows[farms_alive])
                                    #print("The reward before is: %d" %reward)
                                    reward = cows_alive
                                    #print("The reward after is: %d" %reward)
                                    #Save sample <s,a,r,done> to replay memory
                                    done = 1
                                    mat = np.array(pixelarray.reshape(Size,Size,1))
                                    memory.append((mat, placeholder, reward, done))
                                    #print("Start the replay")
                                    #train_replay(batch_size,discount_factor,state_size)
                
                                    #Make currentState the newState
                                    #print("Update State")
                                    currentStatus = new6
                                    pixelarray = new6PixelArray
                                    #print(pixelarray)
                                    
                                    #Make the target model the current model
                                    print("updating the target weights")
                                    model3weights = model3.get_weights()
                                    update_target_model(model3weights)
                
                                    time_list[e] = time_count
                                else:
                                    warning = 0
                                    print("Infecteds on 6th day")
                                    reward = reward -500
                                    #print("The reward is: %d" %reward)
                                    farm_mat = np.c_[list(range(N)),farm_inds,new6PixelArray[farm_inds]]
                                    sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                                    not_culled_farm_inds = sub_farm_mat[:,0]
                                    #print("Farms not culled: %s" %not_culled_farm_inds)
                                    mat = np.array(pixelarray.reshape(Size,Size,1))
                                    newMat = np.array(new6PixelArray.reshape(Size,Size,1))
                                    memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                                    currentStatus = new6
                                    pixelarray = new6PixelArray
                                    if len(epsilon_vector) > 1:
                                        epsilon = epsilon_vector.pop()
                                    else:
                                        epsilon = epsilon_min
                                        
                            else:
                                warning = 0
                                print("Infecteds on 5th day")
                                reward = reward -500
                                #print("The reward is: %d" %reward)
                                farm_mat = np.c_[list(range(N)),farm_inds,new5PixelArray[farm_inds]]
                                sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                                not_culled_farm_inds = sub_farm_mat[:,0]
                                #print("Farms not culled: %s" %not_culled_farm_inds)
                                mat = np.array(pixelarray.reshape(Size,Size,1))
                                newMat = np.array(new5PixelArray.reshape(Size,Size,1))
                                memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                                currentStatus = new5
                                pixelarray = new5PixelArray
                                if len(epsilon_vector) > 1:
                                    epsilon = epsilon_vector.pop()
                                else:
                                    epsilon = epsilon_min
                        else:
                            warning = 0
                            print("Infecteds on 4th day")
                            reward = reward -500
                            #print("The reward is: %d" %reward)
                            farm_mat = np.c_[list(range(N)),farm_inds,new4PixelArray[farm_inds]]
                            sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                            not_culled_farm_inds = sub_farm_mat[:,0]
                            #print("Farms not culled: %s" %not_culled_farm_inds)
                            mat = np.array(pixelarray.reshape(Size,Size,1))
                            newMat = np.array(new4PixelArray.reshape(Size,Size,1))
                            memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                            currentStatus = new4
                            pixelarray = new4PixelArray
                            if len(epsilon_vector) > 1:
                                epsilon = epsilon_vector.pop()
                            else:
                                epsilon = epsilon_min
                    else:
                        warning = 0
                        print("Infecteds on 3rd day")
                        reward = reward -500
                        #print("The reward is: %d" %reward)
                        farm_mat = np.c_[list(range(N)),farm_inds,new3PixelArray[farm_inds]]
                        sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                        not_culled_farm_inds = sub_farm_mat[:,0]
                        #print("Farms not culled: %s" %not_culled_farm_inds)
                        mat = np.array(pixelarray.reshape(Size,Size,1))
                        newMat = np.array(new3PixelArray.reshape(Size,Size,1))
                        memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                        currentStatus = new3
                        pixelarray = new3PixelArray
                        if len(epsilon_vector) > 1:
                            epsilon = epsilon_vector.pop()
                        else:
                            epsilon = epsilon_min
                else:
                    warning = 0
                    print("Infecteds on 2nd day")
                    reward = reward -500
                    #print("The reward is: %d" %reward)
                    farm_mat = np.c_[list(range(N)),farm_inds,new2PixelArray[farm_inds]]
                    sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                    not_culled_farm_inds = sub_farm_mat[:,0]
                    #print("Farms not culled: %s" %not_culled_farm_inds)
                    mat = np.array(pixelarray.reshape(Size,Size,1))
                    newMat = np.array(new2PixelArray.reshape(Size,Size,1))
                    memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                    currentStatus = new2
                    pixelarray = new2PixelArray
                    if len(epsilon_vector) > 1:
                        epsilon = epsilon_vector.pop()
                    else:
                        epsilon = epsilon_min
            
        #If you have reached daily capacity, evolve outbreak with culls
        else:
            num_culls = 0
            time_count = time_count + 1 
            #Cull the chosen farm
            newStatus[placeholder] = -1
            newPixelArray[farm_inds[placeholder]] = 2
            #Evolve the outbreak
            print("You are evolving")
            evolvedStatus = Iterate(newStatus,OutbreakParams, x, y)
            new_inf_inds = np.where(evolvedStatus > 5)[0]
            newPixelArray[farm_inds[new_inf_inds]] = 1
        
            #Generate rewards
            reward = 0
            #Penalize for culling an already culled farm
            #if pixelarray[farm_inds[placeholder]] == 2:
            #    print("Reward before")
            #    print("The reward is: %d" %reward)
            #    reward = reward -1000000
            #    print("The reward is: %d" %reward)
            #    print("Reward after,You culled a culled farm!")
            
            #Penalize for outbreak still continuing
            if np.sum(newPixelArray == 1) > outbreak_thres:
                #print("Outbreak still continuing")
                #print("The reward before is: %d" %reward)
                reward = reward -100
                #print("The reward after is: %d" %reward)
        
                #Save sample <s,a,r,done> to replay memory
                done = 0
                farm_mat = np.c_[list(range(N)),farm_inds,newPixelArray[farm_inds]]
                sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                not_culled_farm_inds = sub_farm_mat[:,0]
                #print("Farms not culled: %s" %not_culled_farm_inds)
                mat = np.array(pixelarray.reshape(Size,Size,1))
                newMat = np.array(newPixelArray.reshape(Size,Size,1))
                memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                #print("Start the replay")
                #train_replay(batch_size,discount_factor,state_size)
                #print("Finished the replay")
                
                #Make currentState the newState
                #print("Update State")
                currentStatus = evolvedStatus
                pixelarray = newPixelArray
                
                inf_farms = np.where(newPixelArray == 1)[0]
                not_inf_farms = np.where(newPixelArray == 0)[0]
                inds = np.r_[inf_farms,not_inf_farms]
                
                #Update epsilon
                if len(epsilon_vector) > 1:
                    epsilon = epsilon_vector.pop()
                else:
                    epsilon = epsilon_min
                
        
            #If outbreak is done then reward based on number of uninfected farms still standing 
            if np.sum(newPixelArray == 1) == outbreak_thres:
                print("I think outbreak is done")
                
                #Evolve the outbreak for 6 timesteps to check if done, don't give a reward or fit model while you wait
                new2 = justEvolve(newStatus, OutbreakParams, x, y)
                new_inf_inds = np.where(new2 > 5)[0]
                new2PixelArray = newPixelArray.copy()
                new2PixelArray[farm_inds[new_inf_inds]] = 1
                if np.sum(new2 > 5) == 0:
                    print("No infecteds for 2 days")
                    new3 = justEvolve(new2, OutbreakParams, x, y)
                    new_inf_inds = np.where(new3 > 5)[0]
                    new3PixelArray = new2PixelArray.copy()
                    new3PixelArray[farm_inds[new_inf_inds]] = 1
                    if np.sum(new3 > 5) == 0:
                        print("No infecteds for 3 days")
                        new4 = justEvolve(new3, OutbreakParams, x, y)
                        new_inf_inds = np.where(new4 > 5)[0]
                        new4PixelArray = new3PixelArray.copy()
                        new4PixelArray[farm_inds[new_inf_inds]] = 1
                        if np.sum(new4 > 5) == 0:
                            print("No infecteds for 4 days")
                            new5 = justEvolve(new4, OutbreakParams, x, y)
                            new_inf_inds = np.where(new5 > 5)[0]
                            new5PixelArray = new4PixelArray.copy()
                            new5PixelArray[farm_inds[new_inf_inds]] = 1
                            if np.sum(new5 > 5) == 0:
                                print("No infecteds for 5 days")
                                new6 = justEvolve(new5, OutbreakParams, x, y)
                                new_inf_inds = np.where(new6 > 5)[0]
                                new6PixelArray = new5PixelArray.copy()
                                new6PixelArray[farm_inds[new_inf_inds]] = 1
                                if np.sum(new6 > 5) == 0:
                                    print("No infecteds for 6 days")
                                    warning = 6
                                    #CONSTRCUT THE TERMINAL REWARD BASED ON THE NUMBER OF NONINFECTED COWS STILL STANDING
                                    #num_alive = np.sum(newPixelArray == 0)
                                    farms_alive= np.where(np.logical_and(new6 >= 0,new6 <= 5))[0]
                                    #print([farms_alive,Cows[farms_alive]])
                                    cows_alive = np.sum(Cows[farms_alive])
                                    #print("The reward before is: %d" %reward)
                                    reward = cows_alive
                                    #print("The reward after is: %d" %reward)
                                    #Save sample <s,a,r,done> to replay memory
                                    done = 1
                                    mat = np.array(pixelarray.reshape(Size,Size,1))
                                    memory.append((mat, placeholder, reward, done))
                                    #print("Start the replay")
                                    #train_replay(batch_size,discount_factor,state_size)
                
                                    #Make currentState the newState
                                    #print("Update State")
                                    currentStatus = new6
                                    pixelarray = new6PixelArray
                                    #print(pixelarray)
                                    
                                    #Make the target model the current model
                                    print("updating the target weights")
                                    model3weights = model3.get_weights()
                                    update_target_model(model3weights)
                
                                    time_list[e] = time_count
                                else:
                                    warning = 0
                                    print("Infecteds on 6th day")
                                    reward = reward -500
                                    #print("The reward is: %d" %reward)
                                    farm_mat = np.c_[list(range(N)),farm_inds,new6PixelArray[farm_inds]]
                                    sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                                    not_culled_farm_inds = sub_farm_mat[:,0]
                                    #print("Farms not culled: %s" %not_culled_farm_inds)
                                    mat = np.array(pixelarray.reshape(Size,Size,1))
                                    newMat = np.array(new6PixelArray.reshape(Size,Size,1))
                                    memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                                    currentStatus = new6
                                    pixelarray = new6PixelArray
                                    if len(epsilon_vector) > 1:
                                        epsilon = epsilon_vector.pop()
                                    else:
                                        epsilon = epsilon_min
                            else:
                                warning = 0
                                print("Infecteds on 5th day")
                                reward = reward -500
                                #print("The reward is: %d" %reward)
                                farm_mat = np.c_[list(range(N)),farm_inds,new5PixelArray[farm_inds]]
                                sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                                not_culled_farm_inds = sub_farm_mat[:,0]
                                #print("Farms not culled: %s" %not_culled_farm_inds)
                                mat = np.array(pixelarray.reshape(Size,Size,1))
                                newMat = np.array(new5PixelArray.reshape(Size,Size,1))
                                memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                                currentStatus = new5
                                pixelarray = new5PixelArray
                                if len(epsilon_vector) > 1:
                                    epsilon = epsilon_vector.pop()
                                else:
                                    epsilon = epsilon_min
                        
                        else:
                            warning = 0
                            print("Infecteds on 4th day")
                            reward = reward -500
                            #print("The reward is: %d" %reward)
                            farm_mat = np.c_[list(range(N)),farm_inds,new4PixelArray[farm_inds]]
                            sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                            not_culled_farm_inds = sub_farm_mat[:,0]
                            #print("Farms not culled: %s" %not_culled_farm_inds)
                            mat = np.array(pixelarray.reshape(Size,Size,1))
                            newMat = np.array(new4PixelArray.reshape(Size,Size,1))
                            memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                            currentStatus = new4
                            pixelarray = new4PixelArray
                            if len(epsilon_vector) > 1:
                                epsilon = epsilon_vector.pop()
                            else:
                                epsilon = epsilon_min
                    else:
                        warning = 0
                        print("Infecteds on 3rd day")
                        reward = reward -500
                        #print("The reward is: %d" %reward)
                        farm_mat = np.c_[list(range(N)),farm_inds,new3PixelArray[farm_inds]]
                        sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                        not_culled_farm_inds = sub_farm_mat[:,0]
                        #print("Farms not culled: %s" %not_culled_farm_inds)
                        mat = np.array(pixelarray.reshape(Size,Size,1))
                        newMat = np.array(new3PixelArray.reshape(Size,Size,1))
                        memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                        currentStatus = new3
                        pixelarray = new3PixelArray
                        if len(epsilon_vector) > 1:
                            epsilon = epsilon_vector.pop()
                        else:
                            epsilon = epsilon_min
                else:
                    warning = 0
                    print("Infecteds on 2nd day")
                    reward = reward -500
                    #print("The reward is: %d" %reward)
                    farm_mat = np.c_[list(range(N)),farm_inds,new2PixelArray[farm_inds]]
                    sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
                    not_culled_farm_inds = sub_farm_mat[:,0]
                    #print("Farms not culled: %s" %not_culled_farm_inds)
                    mat = np.array(pixelarray.reshape(Size,Size,1))
                    newMat = np.array(new2PixelArray.reshape(Size,Size,1))
                    memory.append((mat, placeholder, reward, done, newMat,not_culled_farm_inds))
                    currentStatus = new2
                    pixelarray = new2PixelArray
                    if len(epsilon_vector) > 1:
                        epsilon = epsilon_vector.pop()
                    else:
                        epsilon = epsilon_min
                    
        
        reward_counter += reward
        Sus = np.where(currentStatus == 0)[0]
        Exp = np.where(np.logical_and(0 < currentStatus, currentStatus <= 5))[0]
        Inf = np.where(currentStatus > 5)[0]
        Culled = np.where(currentStatus == -1)[0]
        #print("Sus: %d, Exp: %d, Inf: %d, Culled: %d" %(len(Sus),len(Exp),len(Inf),len(Culled)))
        
        #print("The new status is: %s" %currentStatus)
        #print("The number of infected farms left is: %d" %np.sum(newPixelArray == 1))
        #print("Start the replay")
        train_replay(batch_size,discount_factor,state_size)
        #print("Finished the replay")
        
        
        #Don't do the mini batch, just fit based on the new observation
        #print("Start the fit")
        #target = model.predict(np.array([pixelarray]))
        #target_val = model.predict(np.array([newPixelArray]))
        #if done == 1:
        #    target[0][placeholder] = reward
        #else:
        #    target[0][placeholder] = reward + discount_factor*np.amax(target_val)
        
        #print("This is right before the model.fit step %s" %target)
        #model.fit(np.array([pixelarray]),target, epochs = 1000, verbose=0)
        #train_replay(batch_size,discount_factor,state_size)
        #print("Finished the fit")
        
        #print("The updated model.fit is: %s" %model.predict(np.array([pixelarray])))
    print("The reward at the end of the episode is: %d" %reward_counter)
    reward_list.append(reward_counter)
endtime = time.process_time()
print(endtime-starttime)

model3.save(results_path+'my_model_AtariParams_updatetarget_8kEp_2500deque2_DC5_15x15_3_3_18.h5')


with open(results_path+"time_listAtariParams_updatetarget_8kEp_2500deque2_DC5_15x15_3_3_18.txt", "wb") as fp:   
    #Pickling
    pickle.dump(time_list, fp)

with open(results_path+"reward_listAtariParams_updatetarget_8kEp_2500deque2_DC5_15x15_3_3_18.txt", "wb") as fp:   
    #Pickling
    pickle.dump(reward_list, fp)
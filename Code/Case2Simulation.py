import pandas as pd
import numpy as np
from keras.models import load_model
from scipy import spatial
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import pickle
from math import pi
import copy
import random
import matplotlib as mpl
import matplotlib
matplotlib.use('TkAgg') #include this line so matplotlib.pyplot will run in a virtual enviornment
import matplotlib.pyplot as plt

#Load saved model
model = load_model('/Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/PickNextBextFarm/ACCREStuff/3_16_18/my_model_AtariParams_updatetarget_8kEp_2500deque2_DC5_15x15_3_3_18.h5')

#Load coordinate location, farm sizes, Outbreak Parameters, original pixel array, and original status
coords = np.load('/Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/coords15x15_3_3_18.npy') 
x = coords[:,2]
y = coords[:,3]
Cows = np.load('/Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/Cows15x15_3_3_18.npy')
with open("/Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/OutbreakParams15x15_3_3_18.txt", "rb") as fp: # Unpickling
    OutbreakParams = pickle.load(fp)
original_pixelarray = np.load('/Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/original_pixelarray15x15_3_3_18.npy') 
original_currentStatus = np.load('/Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/original_currentStatus15x15_3_3_18.npy') 
    
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
    
def AnimateOutbreak(coords, currentStatus, tag, culled_farms,cull_num,string):
    fig, ax = plt.subplots()
    x = coords[:,2]
    y = coords[:,3]
    inf = currentStatus
    print(inf)
    colors=["#00FF00","#FF0066","#0000FF","000000"]
    color_list = []
    for i in range(len(Cows)):
        if inf[i] == 0:
            color_list.append(colors[0])
        elif inf[i] > 5:
            color_list.append(colors[1])
        elif 0 < inf[i] <= 5:
            color_list.append(colors[2])
        elif inf[i] == -1:
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
    
############
### MAIN ###
############

#Initialize some parameters
Size = 15
N = 120
dailyCapacity = 5
outbreak_thres = 0

cowsSaved_list = []
total_farms_culled_list = []
time_count_list = []
init_infect_list = []
culled_farms = []
cullsonly_list = []
outbreakagain_list = []

master_exp_list = []

susceptible_cull_list = []
exposed_cull_list = []
infected_cull_list = []

all_dist = []
all_cattle = []

##THE FOLLOWING CODE CONSTRUCTS A NUMPY ARRAY OF THE AVERAGE DISTANCE TO THE INITIAL INFECTED FARMS AND THE NUMBER OF COWS AT THE FARM 
init_relevant_info = np.c_[original_currentStatus, x,y, Cows] ##Make a numpy array of relevant info 
not_infected = init_relevant_info[init_relevant_info[:,0] == 0] ##Subset non-infected farms
not_infected_farm_coords = list(zip(not_infected[:,1],not_infected[:,2])) ##Zip the non-infected coordinates
list_of_not_infected_coords = [list(elem) for elem in not_infected_farm_coords] ##Convert the non-infected coordinates to list
infected = init_relevant_info[init_relevant_info[:,0] == 6] ##Subset infected farms
infected_farm_coords = list(zip(infected[:,1],infected[:,2])) ##Zip the infected coordinates
list_of_infected_coords = [list(elem) for elem in infected_farm_coords]  ##Convert the infected coordinates to a list
distance_matrix = spatial.distance_matrix(list_of_not_infected_coords,list_of_infected_coords) ##Calculate distance from non-infected to each init infected
avg_init_distance_to_infected = np.apply_along_axis(np.mean,1,distance_matrix) ##Calculate average distance
Cows_and_avg_init_distance = np.c_[avg_init_distance_to_infected,not_infected[:,3]] ##Combine #Cows, and avg. init distance in a numpy array
plt.scatter(Cows_and_avg_init_distance[:,0], Cows_and_avg_init_distance[:,1])
plt.xlabel("Average initial distance to infected farms")
plt.ylabel("#Cows at susceptible farm")
plt.show()
cull_counter = np.repeat(0,120)

for i in range(2000):
    print("You are on episode: %d" %i)
    #Initialize some parameters
    Size = 15
    N = 120
    dailyCapacity = 5
    outbreak_thres = 0
    gridsquare = coords[:,4]

    #Begin Management
    num_culls = 0
    time_count = 0
    total_culls = 0
    culls_only = 0
    outbreakagain = 0
    exp_list = [] #THIS IS TO STORE THE AMOUNT OF EXPOSED FARMS AFTER THE FIRST EVOLUTION
    firstevolution = 0 #THIS IS TO INDICATE WHEN TO SUBSET THE AMOUNT OF EXPOSED AFTER THE FIRST EVOLUTION
    susceptible_cull_counter = 0
    exposed_cull_counter = 0
    infected_cull_counter = 0

    #Read in the initial status and pixel array
    pixelarray = original_pixelarray
    currentStatus = original_currentStatus
    farm_inds = np.where(pixelarray != 2)[0]
    init_infect_ind = np.where(currentStatus == 6)
    init_infect = np.sum(Cows[init_infect_ind])
    
    #print(pixelarray.reshape(Size,Size))
    #AnimateOutbreak(coords, currentStatus, "Initial State", culled_farms, 0, 0)
    
    cattle_list = []
    dist_list = []
    
    while np.sum(pixelarray == 1) > outbreak_thres:
        print("Timestep is: %d" %time_count)
        #Choose an action
        mat = pixelarray.reshape(Size,Size,1)
        prediction = model.predict(np.array([mat]))
        farm_mat = np.c_[list(range(N)),gridsquare,pixelarray[farm_inds]]
        sub_farm_mat = farm_mat[farm_mat[:,2] != 2]
        currentAction_ind = np.argmax(prediction[0][sub_farm_mat[:,0].astype(int)])
        currentAction = sub_farm_mat[:,0][currentAction_ind].astype(int)
        print("You culled farm: %d" %currentAction)
        culled_farms.append(currentAction)
        cull_counter[currentAction] = cull_counter[currentAction] + 1
        num_culls += 1 
        culls_only += 1
        if currentStatus[currentAction] == 0:
            string = "You culled a susceptible farm"
            print(string)
            
            #Calculate the distance from the culled susceptible farm to nearest infected farm
            ##Start building the dataset using covariates you already have
            data = np.c_[list(range(N)),currentStatus,x,y,Cows]
            
            ##Subset the infected farms
            inf_farms = data[data[:,1]> 5]
            
            ##Subset the culled farm
            culled_farm = data[data[:,0] == currentAction]
            
            #Combine culled and infected farms
            relevant_farms = np.r_[culled_farm,inf_farms]
            
            #Calculate distance to nearest infected farm
            farm_coords = relevant_farms[:,[2,3]]
            dist_to_nearest_infected = sorted(squareform(pdist(farm_coords))[0])[1]
            dist_list.append(dist_to_nearest_infected)
            
            #Determine number of cattle at the culled, susceptible farm
            #cattle_at_nearest_infected = relevant_farms[np.argsort(squareform(pdist(farm_coords))[0])[1],4]
            #print(Cows[currentAction])
            cattle_list.append(Cows[currentAction])
            
            susceptible_cull_counter += 1
            
        elif 0 < currentStatus[currentAction] <= 5:
            string = "You culled an exposed farm"
            print(string)
            exposed_cull_counter += 1
        elif currentStatus[currentAction] > 5:
            string = "You culled an infected farm"
            print(string)
            infected_cull_counter += 1
        else:
            string = "You culled a culled farm!"
            print(string)
        
        #Initialize the new state
        newStatus = currentStatus.copy()
        newPixelArray = pixelarray.copy()
    
        #If you haven't reached daily capacity, cull the farm
        if num_culls < dailyCapacity:
            print("Under daily capacity")
            #Cull the chosen farm
            newPixelArray[farm_inds[currentAction]] = 2
            newStatus[currentAction] = -1  
            #AnimateOutbreak(coords,newStatus,"", culled_farms, total_culls + 1, string) 
        
            #Make currentState the newState
            print("Update State")
            currentStatus = newStatus
            pixelarray = newPixelArray
            total_culls += 1
        
            Sus = np.where(newStatus == 0)[0]
            Exp = np.where(np.logical_and(0 < newStatus, newStatus <= 5))[0]
            Inf = np.where(newStatus > 5)[0]
            Culled = np.where(newStatus == -1)[0]
            print("Sus: %d, Exp: %d, Inf: %d, Culled: %d" %(len(Sus),len(Exp),len(Inf),len(Culled)))
            #print(currentStatus)
        
            if np.sum(pixelarray == 1) == outbreak_thres:
                print("I think the outbreak is finished.")
                
                #Evolve the outbreak for 6 timesteps to check if done, don't give a reward or fit model while you wait
                new2 = justEvolve(newStatus, OutbreakParams, x, y)
                new_inf_inds = np.where(new2 > 5)[0]
                new2PixelArray = newPixelArray.copy()
                new2PixelArray[farm_inds[new_inf_inds]] = 1 
                if np.sum(new2 > 5) == 0:
                    string = "No infecteds for 2 days"
                    print(string)
                    #AnimateOutbreak(coords,new2,"", culled_farms, total_culls + 1, string)
                    new3 = justEvolve(new2, OutbreakParams, x, y)
                    new_inf_inds = np.where(new3 > 5)[0]
                    new3PixelArray = new2PixelArray.copy()
                    new3PixelArray[farm_inds[new_inf_inds]] = 1
                    if np.sum(new3 > 5) == 0:
                        string = "No infecteds for 3 days"
                        print(string)
                        #AnimateOutbreak(coords,new3,"", culled_farms, total_culls + 1, string)
                        new4 = justEvolve(new3, OutbreakParams, x, y)
                        new_inf_inds = np.where(new4 > 5)[0]
                        new4PixelArray = new3PixelArray.copy()
                        new4PixelArray[farm_inds[new_inf_inds]] = 1
                        if np.sum(new4 > 5) == 0:
                            string = "No infecteds for 4 days"
                            print(string)
                            #AnimateOutbreak(coords,new4,"", culled_farms, total_culls + 1, string)
                            new5 = justEvolve(new4, OutbreakParams, x, y)
                            new_inf_inds = np.where(new5 > 5)[0]
                            new5PixelArray = new4PixelArray.copy()
                            new5PixelArray[farm_inds[new_inf_inds]] = 1
                            if np.sum(new5 > 5) == 0:
                                string = "No infecteds for 5 days"
                                print(string)
                                #AnimateOutbreak(coords,new5,"", culled_farms, total_culls + 1, string)
                                new6 = justEvolve(new5, OutbreakParams, x, y)
                                new_inf_inds = np.where(new6 > 5)[0]
                                new6PixelArray = new5PixelArray.copy()
                                new6PixelArray[farm_inds[new_inf_inds]] = 1 
                                if np.sum(new6 > 5) == 0:
                                    string = "Outbreak done!"
                                    print(string)
                                    #AnimateOutbreak(coords,new6,"", culled_farms, total_culls + 1, string)
                                else:
                                    string = "Infecteds on 6th day"
                                    print(string)
                                    #AnimateOutbreak(coords,new6,"", culled_farms, total_culls + 1, string)
                                    currentStatus = new6
                                    pixelarray = new6PixelArray
                                    total_culls += 1
                                    outbreakagain += 1
                            else:
                                string = "Infecteds on 5th day"
                                print(string)
                                #AnimateOutbreak(coords,new5,"", culled_farms, total_culls + 1, string)
                                currentStatus = new5
                                pixelarray = new5PixelArray
                                total_culls += 1
                                outbreakagain += 1
                        else:
                            string = "Infecteds on 4th day"
                            print(string)
                            #AnimateOutbreak(coords,new4,"", culled_farms, total_culls + 1, string)
                            currentStatus = new4
                            pixelarray = new4PixelArray
                            total_culls += 1
                            outbreakagain += 1
                    else:
                        string = "Infecteds on 3rd day"
                        print(string)
                        #AnimateOutbreak(coords,new3,"", culled_farms, total_culls + 1, string)
                        currentStatus = new3
                        pixelarray = new3PixelArray
                        total_culls += 1
                        outbreakagain += 1
                else:
                    string = "Infecteds on 2nd day"
                    print(string)
                    #AnimateOutbreak(coords,new2,"", culled_farms, total_culls + 1, string)
                    currentStatus = new2
                    pixelarray = new2PixelArray
                    total_culls += 1
                    outbreakagain += 1
                
        #If you have reached daily capacity evolve and cull
        else:
            time_count += 1 
            #Cull the chosen farm
            newStatus[currentAction] = -1
            newPixelArray[farm_inds[currentAction]] = 2
            #AnimateOutbreak(coords,newStatus,"", culled_farms, total_culls + 1, string)
            
            #Evolve the outbreak
            print("You are evolving")
            evolvedStatus = Iterate(newStatus,OutbreakParams,coords[:,2],coords[:,3])
            #print(evolvedStatus)
            new_inf_inds = np.where(evolvedStatus > 5)[0]
            newPixelArray[farm_inds[new_inf_inds]] = 1
            num_culls = 0
        
            Sus = np.where(evolvedStatus == 0)[0]
            Exp = np.where(np.logical_and(0 < evolvedStatus, evolvedStatus <= 5))[0]
            Inf = np.where(evolvedStatus > 5)[0]
            Culled = np.where(evolvedStatus == -1)[0]
            print("Sus: %d, Exp: %d, Inf: %d, Culled: %d" %(len(Sus),len(Exp),len(Inf),len(Culled)))
            
            firstevolution += 1
            if firstevolution == 1:
                exp_list.append(len(Exp))
        
            #Make currentState the newState
            print("Update State")
            currentStatus = evolvedStatus
            pixelarray = newPixelArray
            total_culls += 1
        
            if np.sum(pixelarray == 1) == outbreak_thres:
                print("I think the outbreak is finished.")
                
                #Evolve the outbreak for 6 timesteps to check if done, don't give a reward or fit model while you wait
                new2 = justEvolve(newStatus, OutbreakParams, x, y)
                new_inf_inds = np.where(new2 > 5)[0]
                new2PixelArray = newPixelArray.copy()
                new2PixelArray[farm_inds[new_inf_inds]] = 1
                if np.sum(new2 > 5) == 0:
                    string = "No infecteds for 2 days"
                    print(string)
                    #AnimateOutbreak(coords,new2,"", culled_farms, total_culls + 1, string)
                    new3 = justEvolve(new2, OutbreakParams, x, y)
                    new_inf_inds = np.where(new3 > 5)[0]
                    new3PixelArray = new2PixelArray.copy()
                    new3PixelArray[farm_inds[new_inf_inds]] = 1
                    if np.sum(new3 > 5) == 0:
                        string = "No infecteds for 3 days"
                        print(string)
                        #AnimateOutbreak(coords,new3,"", culled_farms, total_culls + 1, string)
                        new4 = justEvolve(new3, OutbreakParams, x, y)
                        new_inf_inds = np.where(new4 > 5)[0]
                        new4PixelArray = new3PixelArray.copy()
                        new4PixelArray[farm_inds[new_inf_inds]] = 1
                        if np.sum(new4 > 5) == 0:
                            string = "No infecteds for 4 days"
                            print(string)
                            #AnimateOutbreak(coords,new4,"", culled_farms, total_culls + 1, string)
                            new5 = justEvolve(new4, OutbreakParams, x, y)
                            new_inf_inds = np.where(new5 > 5)[0]
                            new5PixelArray = new4PixelArray.copy()
                            new5PixelArray[farm_inds[new_inf_inds]] = 1
                            if np.sum(new5 > 5) == 0:
                                string = "No infecteds for 5 days"
                                print(string)
                                #AnimateOutbreak(coords,new5,"", culled_farms, total_culls + 1, string)
                                new6 = justEvolve(new5, OutbreakParams, x, y)
                                new_inf_inds = np.where(new6 > 5)[0]
                                new6PixelArray = new5PixelArray.copy()
                                new6PixelArray[farm_inds[new_inf_inds]] = 1
                                if np.sum(new6 > 5) == 0:
                                    string = "Outbreak done!"
                                    print(string)
                                    #AnimateOutbreak(coords,new6,"", culled_farms, total_culls + 1, string)
                                else:
                                    string = "Infecteds on 6th day"
                                    print(string)
                                    #AnimateOutbreak(coords,new6,"", culled_farms, total_culls + 1, string)
                                    currentStatus = new6
                                    pixelarray = new6PixelArray
                                    total_culls += 1
                                    outbreakagain += 1
                            else:
                                string = "Infecteds on 5th day"
                                print(string)
                                #AnimateOutbreak(coords,new5,"", culled_farms, total_culls + 1, string)
                                currentStatus = new5
                                pixelarray = new5PixelArray
                                total_culls += 1
                                outbreakagain += 1
                        else:
                            string = "Infecteds on 4th day"
                            print(string)
                            #AnimateOutbreak(coords,new4,"", culled_farms, total_culls + 1, string)
                            currentStatus = new4
                            pixelarray = new4PixelArray
                            total_culls += 1
                            outbreakagain += 1
                    else:
                        string = "Infecteds on 3rd day"
                        print(string)
                        #AnimateOutbreak(coords,new3,"", culled_farms, total_culls + 1, string)
                        currentStatus = new3
                        pixelarray = new3PixelArray
                        total_culls += 1
                        outbreakagain += 1
                else:
                    string = "Infecteds on 2nd day"
                    print(string)
                    #AnimateOutbreak(coords,new2,"", culled_farms, total_culls + 1, string)
                    currentStatus = new2
                    pixelarray = new2PixelArray
                    total_culls += 1
                    outbreakagain += 1
                
                        
        total_cows_culled_ind = np.where(currentStatus == -1)
        total_cows_culled = np.sum(Cows[total_cows_culled_ind])
           
        print("Remember, there were initially this number of infected cows: %d" %init_infect)
        print("You culled a total of: %d, and you took this many days: %d" %(total_cows_culled, time_count + 1))
    
    cowsSaved = np.sum(Cows[np.where(currentStatus == 0)[0]])
    
    #Append results to a list 
    cowsSaved_list.append(cowsSaved)
    total_farms_culled_list.append(total_culls)
    time_count_list.append(time_count+1)
    init_infect_list.append(init_infect)
    cullsonly_list.append(culls_only)
    outbreakagain_list.append(outbreakagain)
    
    master_exp_list.append(exp_list)
    
    susceptible_cull_list.append(susceptible_cull_counter)
    exposed_cull_list.append(exposed_cull_counter)
    infected_cull_list.append(infected_cull_counter)

fulldat = np.c_[original_currentStatus,x,y,Cows,cull_counter]
new_not_infected = fulldat[fulldat[:,0] == 0]
new_not_infected = np.c_[new_not_infected, Cows_and_avg_init_distance[:,0]] ##Includes original status, x, y, #Cows, #culls in 2000 sims, avg distance to init infected
np.save("/Users/sandyalakkur/Documents/GradSchool/Dissertation/Paper2/Analysis/DenseBiggerUnclusteredLowerCullCapacity/RLStuff/PickNextBextFarm/ACCREStuff/3_16_18/PublicationStuff/new_not_infected.npy",new_not_infected)
#cmap = mpl.cm.Blues(np.linspace(0,1,20))
#cmap = mpl.colors.ListedColormap(cmap[10:,:-1])
plt.scatter(Cows_and_avg_init_distance[:,0], Cows_and_avg_init_distance[:,1], c = new_not_infected[:,4], cmap = plt.cm.bwr)
cbar = plt.colorbar()
cbar.set_label("#culls in 2000 simulations", rotation=270, labelpad=13)
plt.xlabel("Average distance to initial infected farms")
plt.ylabel("#Cows at susceptible farm")
plt.show() 
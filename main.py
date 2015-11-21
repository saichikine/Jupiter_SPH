
# coding: utf-8

# # Collision of Two Jovian Planets
# 
# ## Roadmap
# * Initialization <b>DONE!</b>
#     * Initialize position
#         * Distribute particles radially using [equation 4.18]~~
#     * Initialize velocity
#         * Uniform velocity for every particle in planet
#     * Initialize density $\rho_i$
#         * Using $\rho = \rho_c * sin(\zeta) / \zeta $
#     * Calculate mass of each particle
#         * Use the smoothing function to relate mass to density
#         * Keep the mass of each particle fixed for the remainder of calculations
#     * Choose time scale (not done)
#         * Time must satisfy the CFL condition [equation 4.12] 
#         
# * Calculate pressure $P_i$, given density $\rho_i$ <b> Done!</b>
#     * More generaly, the 'entire'?? planet can be modeled with $P = K \rho ^\gamma$ 
#     * [Equations 4.14 - 4.17] go into more detail
#     
# * Calculate $\nabla P_i$ <b> Possibly done! </b>
#     * Calculate the gradient of the smoothing kernel
#     * Then use [equation 4.9]
# 
# * Calculate gravitational forces $\nabla \Phi_i$
#     * First pass can use brute force O(N^2) method [equation 3.1]
#     * Second pass would use trees or FFT
# 
# * Calculate change in velocity $v_i$ [equation 4.2], given pressure, density, and gravitational forces
#     * This involves solving an ODE
#         * First pass: RK4
#         * If bad: RK1
#         * And then: Leapfrog
#         
# * Calculate change in position $r_i$ [equation], given velocity
#     * This involves solving an ODE
#         * First pass: RK4
#         * If bad: RK1
#         * And then: Leapfrog
#         
# ## Weird Things / Concerning Things / Bugs
# * When I initalized 100 particles, I ended up with 94 after all of that
#     * We lose some of them when rounding, but we shouldn't lose 6
# * The max number of particles is not located at the outermost edge of Jupiter
#     * This distribution comes from [equation 4.18]. The error might come from the fact that I didn't normalize alpha = 1, rho_c = 1
# * Time step is 1e-21 for gravitational forces
# * The smoothing function is returning negative values, but they're in the correct order of magnitude
# 
# ## Notes
# * Changed number of partitions to 10
# * Density of Jupiter is ~1.33g/cm^3
#         
# ### Initialize Libraries

# In[1]:

from __future__ import division

get_ipython().magic(u'matplotlib inline')


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time


# ### Initialize Model Constants

# In[2]:

RJupiter = 6.99e9    # cm

gamma = 2
G = 6.67e-8    # dyne*cm^2/g^2

rhoC = 5    # g/cm^3, central density
PC = 6.5e13    # dyne/cm^2
TC = 22000    # K

K = 2.6e12    # dyne*cm^4/g^2

alpha = np.sqrt(K/(2*np.pi*G))


# ### Initialize Positions of Planets

# In[3]:

N1 = 1000    # Particles in Planet 1
N2 = N1     # Particles in Planet 2

# Equations 4.17
# Use partition to give initial positions of particles
partitionNum = 5     
rSpace = np.linspace(0, RJupiter, partitionNum)
zetaSpace = rSpace/alpha

# Establish number of particles in each region of delta(zeta)
NDistribution = []    
for i in range(1,len(zetaSpace)):
    zeta2 = zetaSpace[i]
    zeta1 = zetaSpace[i-1]
    NDistribution.append((np.sin(zeta2) - zeta2*np.cos(zeta2) - np.sin(zeta1) + zeta1*np.cos(zeta1))                         *N1/np.pi)
    
NDistribution = np.array(NDistribution)
NDistribution = np.round(NDistribution)

# Create radial distribution
radiusDistribution = []
i = 0
for N in NDistribution:
    radiusDistribution.append(np.random.uniform(rSpace[i], rSpace[i+1], size=N))
    i += 1

# Flatten radial array
radiusDistribution = [item for sublist in radiusDistribution for item in sublist]
radiusDistribution = np.array(radiusDistribution)

# Create angle distribution
thetaDistribution = np.random.uniform(0, 2*np.pi, size=len(radiusDistribution))


# ### Plot Initial Spatial Distribution

# In[4]:

plt.figure(figsize=(10,10))
plt.scatter(radiusDistribution*np.cos(thetaDistribution), radiusDistribution*np.sin(thetaDistribution))
plt.title("Initial Position Distribution of Particles")
plt.show()


# ### Model Initial Densities
# * Currently modeling density as 4 distinct regions
#     * Might want to follow up with the smoothing that comes with the density_j summation
# * Modeling density as a continuous function of r now, using [equation 4.16]

# In[5]:

rhoDistribution = rhoC * np.sin(radiusDistribution/alpha) / (radiusDistribution/alpha)

print radiusDistribution.shape, rhoDistribution.shape


# ### Plot Initial Densities

# In[6]:

plt.figure(figsize=(10,8))
plt.scatter(radiusDistribution*np.cos(thetaDistribution), radiusDistribution*np.sin(thetaDistribution), c=rhoDistribution)
plt.title("Initial Density Distribution of Particles")
plt.colorbar()
plt.show()


# ### Model Initial Pressure Distribution
# * Currently modeling as simply polytropic
#     * Might want to find P by solving the ODE [equation 4.14]

# In[7]:

pressureDistribution = K * rhoDistribution ** gamma

plt.figure(figsize=(10,8))
plt.scatter(radiusDistribution*np.cos(thetaDistribution),             radiusDistribution*np.sin(thetaDistribution), c=pressureDistribution)
plt.title("Initial Density Distribution of Particles")
plt.colorbar()
plt.show()


# ### Helper Function (Convert Polar Coordinates to Cartesian)

# In[8]:

def polar2cart(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)]).T

xyDistribution = polar2cart(radiusDistribution, thetaDistribution)


# ### Define Smoothing Function

# In[9]:

# MJupiter = 1.89e30    # grams

def W(xyDistribution, h= 3e9, verbose=False):
    '''
    h = ~3e9 looks like a good range
    Finding W for all pairs (i,j), instead of just j. 
    I'm flattening W at the end of the function. I might be meant to flatten W
        by flattening the |r-rj| by taking their sum
    '''
    
    distVect = np.zeros((len(xyDistribution), len(xyDistribution)))
                  
    # Find distance between each j and the other points
    for i in range(len(xyDistribution)):
        particle_i = xyDistribution[i]
        dist = np.sqrt((particle_i[0] - xyDistribution[:,0])**2 + (particle_i[1] - xyDistribution[:,1])**2)
        # Store distances in a matrix
        distVect[:,i] = dist
    
    # r < 1
    distVect1 = distVect < h
    distVect1 = distVect1.astype(int)
    if verbose:
        print "Percent of molecules within 1 smoothing length:",             np.count_nonzero(distVect1) / (len(xyDistribution)**2)
            
    # Matrix containing radii (where only nonzero values are radii < h)
    R1 = distVect1 * distVect   
    W1 = distVect1 / (np.pi * h**3) * (1 - 3/2*(R1/h)**2 + 3/4*(R1/h)**3)
    
    if np.min(W1) < 0:
        print "Warning! Negative smoothing kernel detected!"

    # r > 2h
    distVect2 = distVect > 2 * h
    distVect2 = distVect2.astype(int)
    if verbose:
        print "Percent of molecules which do not affect the particle:",             np.count_nonzero(distVect2) / (len(xyDistribution)**2)
    
    # h < r < 2h
    distVect12 = np.logical_not(np.logical_or(distVect1, distVect2))
    if verbose:
        print "Percent of molecules within 2 smoothing lengths:",                 np.count_nonzero(distVect12) / (len(xyDistribution)**2)
            
    R12 = distVect12 * distVect 
    W12 = distVect12 / (4 * np.pi * h**3) * (2 - (R12/h))**3
    
    if np.min(W12) < 0:
        print "Warning! Negative smoothing kernel detected!"

    W = W1 + W12
    W = np.sum(W, axis=1)
    
    print np.count_nonzero(W)#, np.count_nonzero(distVect)
    return W


# ### Define Gradient of Smoothing Function
# 
# #### Not finished!
# 
# \begin{equation}
# \nabla W(r,h) = \frac{1}{\pi h^3} \left( -3(r/h) + \frac{9}{4}(r/h)^2 \right)
# \end{equation}
# 
# \begin{equation}
# \nabla W(r,h) = -\frac{3}{4 \pi h^3} (r/h)^2
# \end{equation}
# 

# In[10]:

def gradW(xyDistribution, h= 3e9, verbose=False):
    '''
    h = ~3e9 looks like a good range
    Finding W for all pairs (i,j), instead of just j. 
    I'm flattening W at the end of the function. I might be meant to flatten W
        by flattening the |r-rj| by taking their sum
    Should I be inputting abs(X) or just X? What is X relative to?
    Might need to do final summation wrt axis 2, not axis1
    '''
    distVect = np.zeros((len(xyDistribution), len(xyDistribution)))
    thetaVect = np.zeros_like(dist_Vect)
                  
    # Find distance between each j and the other points
    for i in range(len(xyDistribution)):
        particle_i = xyDistribution[i]
        # Sloppy right now. Can save save computing power later.
        dist = np.sqrt((particle_i[0] - xyDistribution[:,0])**2 + (particle_i[1] - xyDistribution[:,1])**2)
        theta = np.arctan((particle_i[1] - xyDistribution[:,1])/particle_i[0] - xyDistribution[:,0])
        # Store radial distances in a matrix
        distVect[:,i] = dist
#        thetaVect[]
    
    print distVect[:4,:4]
    
    # r < 1
    # Vector containing 1's and 0's where the below inequality is satisfied
    distVect1 = distVect < h
    distVect1 = distVect1.astype(int)
    if verbose:
        print "Percent of molecules within 1 smoothing length:",             np.count_nonzero(distVect1) / (len(xyDistribution)**2)
            
    # Matrix containing radii (where only nonzero values are radii < h)
    R1 = distVect1 * distVect
    

    # X and Y components of the smoothing kernel
    W1 = np.array(
         [distVect1 * 3/ (2*np.pi * h**4) * ( -2*X1 + 3*X1*np.sqrt(R1)/h), \
          distVect1 * 3/ (2*np.pi * h**4) * ( -2*Y1 + 3*Y1*np.sqrt(R1)/h)]
                  )

    # r > 2h
    distVect2 = distVect > 2 * h
    distVect2 = distVect2.astype(int)
    if verbose:
        print "Percent of molecules which do not affect the particle:",             np.count_nonzero(distVect2) / (len(xyDistribution)**2)
    
    # h < r < 2h
    distVect12 = np.logical_not(np.logical_or(distVect1, distVect2))
    if verbose:
        print "Percent of molecules within 2 smoothing lengths:",                 np.count_nonzero(distVect12) / (len(xyDistribution)**2)
            
    R12 = distVect12 * distVect 
    X12 = distVect12 * XVect
    Y12 = distVect12 * YVect
    W12 = np.array(
         [distVect1 / (4*np.pi * h**6) * ( -2*X1 / np.sqrt(R1)), \
          distVect1 / (4*np.pi * h**6) * ( -2*Y1 / np.sqrt(R1))]
                    )
    W12 = np.nan_to_num(W12)
    
    W = W1 + W12
    W = np.nan_to_num(W)
    W = np.sum(W, axis=1)

    print np.min(W12)
    
    return W

#z = gradW(xyDistribution)


# In[ ]:




# ### Calculate Initial Mass Distribution
# * Calculate using the smoothing equation [equation 4.4]
# * Assume mass is fixed per particle afterwards
# * The calculated mass distribution is on the same order of magnitude as Jupiter's true mass, which implies that the smoothing function doesn't have any bugs

# In[11]:

mDistribution = rhoDistribution/W(xyDistribution)
trueJupiterMass = 1.89e30    #g

massPercentError  = (trueJupiterMass- np.sum(mDistribution))/trueJupiterMass
print "The percent error of the calculated mass of Jupiter with respect to its true mass is",             "%.4f" % massPercentError, "%"

plt.figure(figsize=(10,8))
plt.scatter(radiusDistribution*np.cos(thetaDistribution),             radiusDistribution*np.sin(thetaDistribution), c=mDistribution)
plt.title("Initial Mass Distribution of Particles")
plt.colorbar()
plt.show()

plt.figure()
plt.hist(mDistribution)
plt.title("Mass Distribution")
plt.show()


# ### Model Initial Velocities

# In[12]:

velocityDistribution = np.zeros_like(xyDistribution)


# ### Model Initial Gravitational Force
# 
# #### Need to Run Entire Code to get reliable visuals! 
# 
# * Looking into https://www.wakari.io/sharing/bundle/yves/Continuum_N_Body_Simulation_Numba to do calcs
# * Here has better approaches http://www.scholarpedia.org/article/N-body_simulations, but no code
# * Ripping code from here: http://hilpisch.com/Continuum_N_Body_Simulation_Numba_27072013.html#/5/1
# 
# * The difference in position appears directly proportional to the size of time step
# 
# * A time step of e-21 allows the planet to stay relatively clumped up after 10000 RK1 steps
#     * ie: Less than 10 particles were ejected into oblivion
#     
# \begin{equation}
# \frac{d^2x}{dt^2} = -GM\frac{x}{r^3}
# \end{equation}
# 
# \begin{equation}
# \frac{d^2y}{dt^2} = -GM\frac{y}{r^3}
# \end{equation}
# 
# Taken from Newman, page 386

# In[ ]:

def gravity(position, velocity):
    '''
    Runs RK4 to update velocity and position. Takes numpy arrays as input.
    '''
    t0 = time.time(); nSteps = 1000; dt = 1000        
    for step in range(1, nSteps + 1, 1):
        for i in range(len(position)):
            potentialX = 0.0; potentialY = 0.0
            for j in range(len(position)):
                if j != i:
                    deltaX = position[j,0] - position[i,0]
                    deltaY = position[j,1] - position[i,1]
                    
                    k1X = dt * -G * mDistribution[j] * deltaX / ((np.sqrt(deltaX**2 + deltaY**2))**3)
 #                   k2X = dt * -G * mDistribution[j] * (deltaX + 0.5*k1X)/ ((np.sqrt((deltaX + 0.5*k1X)**2 + deltaY**2))**3)
 #                   k3X = dt * -G * mDistribution[j] * (deltaX + 0.5*k2X)/ ((np.sqrt((deltaX + 0.5*k2X)**2 + deltaY**2))**3)
 #                   k4X = dt * -G * mDistribution[j] * (deltaX + k3X)/ ((np.sqrt((deltaX + k3X)**2 + deltaY**2))**3)
                    
                    k1Y = dt * -G * mDistribution[j] * deltaY / ((np.sqrt(deltaX**2 + deltaY**2))**3)
#                    k2Y = dt * -G * mDistribution[j] * (deltaY + 0.5*k1Y)/ ((np.sqrt((deltaY + 0.5*k1Y)**2 + deltaX**2))**3)
#                    k3Y = dt * -G * mDistribution[j] * (deltaY + 0.5*k2Y)/ ((np.sqrt((deltaY + 0.5*k2Y)**2 + deltaX**2))**3)
#                    k4Y = dt * -G * mDistribution[j] * (deltaY + k3Y)/ ((np.sqrt((deltaY + k3Y)**2 + deltaX**2))**3)

                velocity[i, 0] = k1X
                velocity[i, 1] = k1Y
#                velocity[i, 0] += 1/6 * (k1X + 2*k2X + 2*k3X + k4X)
#                velocity[i, 1] += 1/6 * (k1Y + 2*k2Y + 2*k3Y + k4Y)
        for i in range(len(xyDistribution)):
            position[i,0] += velocity[i,0] * dt
            position[i,1] += velocity[i,1] * dt


# In[ ]:

xyDistributionOld = np.copy(xyDistribution)

nbody_nb = nb.autojit(gravity)
firstrun = nbody_nb(xyDistribution, velocityDistribution)


# In[ ]:

print xyDistribution - xyDistributionOld


# [[-14817.86252546 -14254.34422445]
#  [ 16439.8530407  -10828.22849607]
#  [  -458.2766366   -6007.35716319]
#  ..., 
#  [-44791.12577057 -49775.21155739]
#  [ -7136.9473145   57910.45627022]
#  [ 10065.26782036  57367.43893051]]


# In[ ]:

plt.figure(figsize=(10,10))
plt.scatter(xyDistribution[:,0], xyDistribution[:,1], c='b')
plt.scatter(xyDistributionOld[:,0], xyDistributionOld[:,1], c='y')
plt.xlim(-0.5e11, 0.5e11)
plt.ylim(-0.5e11, 0.5e11)
plt.title("Close-up of the Planet after 10K Time Steps")
plt.plot()


# In[ ]:

plt.figure(figsize=(10,10))
plt.scatter(xyDistribution[:,0], xyDistribution[:,1], c='b')
plt.scatter(xyDistributionOld[:,0], xyDistributionOld[:,1], c='y')
plt.title("All the Particles of the Planet after 10K Time Steps")
plt.plot()


# ### Calculate Pressure Gradient
# 

# In[ ]:




# In[ ]:




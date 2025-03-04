import numpy as np

def n_body(t, y, p):
    '''
    N-body simulation using force matrix formulation.
    Each entry Fij in the force matrix represents the force from body j on body i.
    The total force on each body is computed by summing its row in the matrix.
    
    The state vector y contains:
    - First n*d elements: positions (d coordinates for each of n bodies)
    - Last n*d elements: velocities (d components for each of n bodies)
    
    Returns dydt containing:
    - First n*d elements: velocity (dr/dt = v)
    - Last n*d elements: acceleration (dv/dt = F/m)
    '''
    G = p['G']
    masses = p['m']
    n = len(masses)
    d = p['dimension']
    
    # unpack state
    positions = y[:n*d].reshape((n, d))
    velocities = y[n*d:].reshape((n, d))
    
    # initialize force matrix
    F = np.zeros((n, n, d)) # I store this as a 3d array, 1 matrix for each body
    # F structure:
    # [[[  0.    0.  ]  # No self-force (F[0,0] = 0) # Body 0
    # [  1.   -0.  ]  # Force from 1 → 0 (F[0,1])
    # [-1.   -0.  ]]  # Force from 2 → 0 (F[0,2])

    # [[-1.    0.  ]  # Force from 0 → 1 (F[1,0]) # Body 1
    # [  0.    0.  ]  # No self-force (F[1,1] = 0)
    # [-0.25 -0.  ]]  # Force from 2 → 1 (F[1,2])derivative of positions is velocities; the derivative of velocities is accelerations

    # [[  1.    0.  ]  # Force from 0 → 2 (F[2,0]) # Body 2
    # [  0.25  0.  ]  # Force from 1 → 2 (F[2,1])
    # [  0.    0.  ]]] # No self-force (F[2,2] = 0)

    # calculate forces
    for i in range(n):
        for j in range(i+1, n):
            # vector from j to i
            r_vec = positions[i] - positions[j]
            
            # gravitational force between j and i
            force = g_force(masses[i], masses[j], G, r_vec)
            
            # populate force matrix
            F[i,j] = -force  # force on body i due to body j
            F[j,i] = force   # force on body j due to body i

    # compute accelerations (sum forces on each body and divide by mass)
    acc = np.zeros_like(positions)
    for i in range(n): # iterate over the F matrix layers (1 per body)
        # sum all forces acting on body i (sum across the row)
        total_force = np.sum(F[i], axis=0)
        acc[i] = total_force / masses[i]
    
    # # if fix_first is set, the first body does not move
    # if p.get("fix_first", False):
    #     acc[0] = 0.0
    #     velocities[0] = 0.0
    
    # the derivative of positions is velocities; the derivative of velocities is accelerations
    dydt = np.concatenate((velocities.flatten(), acc.flatten()))
    # zero out the velocity derivatives for the fixed body
    if p.get("fix_first", False):
        dydt[:d] = 0.0  # set the first 'd' elements (velocity of first body) to 0

        
    return dydt


# helper func for g force
def g_force(m1, m2, g, r_vec):
    '''
    Calculate the gravitational force between two bodies
    '''
    r_hat = np.linalg.norm(r_vec)
    return g * m1 * m2 * r_vec / (r_hat**3)


# new force function for helium
def helium_force(m1, m2, g, r_vec):
    '''
    Calculate force between particles in helium atom
    m1, m2: charges (-1 for electrons, 2 for nucleus)
    r_vec: vector from m2 to m1
    Returns force vector on m1 due to m2
    '''
    r_hat = np.linalg.norm(r_vec)
    if r_hat == 0:  # Prevent division by zero
        return np.zeros_like(r_vec)
    # Coulomb force: F = q1*q2*r_vec/r^3
    # Units chosen where |F| = 1/r^2 between electrons
    return m1 * m2 * r_vec / (r_hat**3)

# new rhs function for helium
def n_body_helium(t, y, p):
    G = p['G']
    charges = p['m']  # m is charges in this case
    n = len(charges)
    d = p['dimension']
    
    positions = y[:n*d].reshape((n, d))
    velocities = y[n*d:].reshape((n, d))
    
    F = np.zeros((n, n, d))
    
    # Calculate forces
    for i in range(n):
        for j in range(i+1, n):
            r_vec = positions[i] - positions[j]
            force = helium_force(charges[i], charges[j], G, r_vec)
            F[i,j] = -force
            F[j,i] = force

    # Compute accelerations (F/m, mass = 1 for electrons)
    acc = np.zeros_like(positions)
    for i in range(n):
        total_force = np.sum(F[i], axis=0)
        acc[i] = total_force  # mass = 1
    
    if p.get("fix_first", True):
        acc[0] = 0.0
        velocities[0] = 0.0
    
    dydt = np.concatenate((velocities.flatten(), acc.flatten()))
    if p.get("fix_first", True):
        dydt[:d] = 0.0
        
    return dydt
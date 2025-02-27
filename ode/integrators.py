import numpy as np
import matplotlib.pyplot as plt

def Euler(dt, f, t, y, args):
    return y + f(t,y,*args) * dt 

def EulerCromer(dt, f, t, y, args):
    y_end = y + f(t,y,*args) * dt
    return y + f(t+dt, y_end, *args) * dt

def EulerRichardson(dt, f, t, y, args):
    y_mid = y + f(t,y,*args) * dt/2
    return y + f(t+dt/2, y_mid, *args) * dt

def RK4(dt, f, t, y, args):
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + k1*dt/2, *args)
    k3 = f(t + dt/2, y + k2*dt/2, *args)
    k4 = f(t + dt, y + k3*dt, *args)
    return y + dt *((1/6 * k1) + (1/3 * k2) + (1/3 * k3) + (1/6 * k4))

def RK45(dt, f, t, y, args):
    atol = 1e-10
    rtol = 1e-14

    c_params = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    a_params = np.array([[0, 0, 0, 0, 0, 0],
                          [1/5, 0, 0, 0, 0, 0],
                          [3/40, 9/40, 0, 0, 0, 0],
                          [44/45, -56/15, 32/9, 0, 0, 0],
                          [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
                          [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
                          [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]])
    b_params = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b_star_params = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    k = np.zeros((7, len(y)))
    # first step
    # k[0] = f(t, y, *args)
    # # second step
    # k[1] = f(t + c_params[1] * dt, y + a_params[1, 0] * dt * k[0], *args)
    # # third step
    # k[2] = f(t + c_params[2] * dt, y + dt * (a_params[2, 0] * k[0] + a_params[2, 1] * k[1]), *args)
    # # fourth step
    # k[3] = f(t + c_params[3] * dt, y + dt * (a_params[3, 0] * k[0] + a_params[3, 1] * k[1] + a_params[3, 2] * k[2]), *args)
    # # ..... we can put this in a loop

    k[0] = f(t, y, *args) # first step
    for i in range(1, 7): # iterate over the rest of the steps (1 to 6)
        y_sum = np.zeros_like(y) # this will be our sum of y's over each k step
        for j in range(i): # sum over all previous k's
            y_sum += a_params[i, j] * k[j] 
        k[i] = f(t + c_params[i] * dt, y + dt * y_sum, *args) # calculate the next k
    y_new = y + dt * np.dot(b_params, k) # calculate the new y
    y_star = y + dt * np.dot(b_star_params, k) # y_star is used to estimate the error

    scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol # scale is used to calculate the error
    error = np.linalg.norm((y_new - y_star) / scale) # error is described as a norm in the book
    # print(f'dt: ', dt)
    # print(f'error: ', error)
    # delta = np.abs(y_new - y_star) <= scale
    if error < 1:
        return y_new # if the error is small enough, we return the new y
    else:
        return RK45(dt/2, f, t, y, args) # if the error is too large, we half the step size and try again

    # return y_new, y_star


def solve_ode(f,tspan, y0, method = Euler, args=(), **options):
    """
    Given a function f that returns derivatives,
    dy / dt = f(t, y)
    and an inital state:
    y(tspan[0]) = y0
    
    This function will return the set of intermediate states of y
    from t0 (tspan[0]) to tf (tspan[1])
    
    
    
    The function is called as follows:
    
    INPUTS 
    
    f - the function handle to the function that returns derivatives of the 
        vector y at time t. The function can also accept parameters that are
        passed via *args, eg f(t,y,g) could accept the acceleration due to gravity.
        
    tspan - a indexed data type that has [t0 tf] as its two members. 
            t0 is the initial time
            tf is the final time
    
    y0 - The initial state of the system, must be passed as a numpy array.
    
    method - The method of integrating the ODEs. This week will be one of Euler, 
             Euler-Cromer, or Euler-Richardson
    
    *args - a tuple containing as many additional parameters as you would like for 
            the function handle f.
    
    **options - a dictionary containing all the keywords that might be used to control
                function behavior. For now, there is only one:
                
                first_step - the initial time step for the simulation.
    
    
    OUTPUTS
    
    t,y
    
    The returned states will be in the form of a numpy array
    t containing the times the ODEs were solved at and an array
    y with shape tsteps,N_y where tsteps is the number of steps 
    and N_y is the number of equations. Observe this makes plotting simple:
    
    plt.plot(t,y[:,0])
    
    would plot positions.
    
    """
    
    t0 = tspan[0]
    tf = tspan[1]
    y = [y0]
    t = [t0]
    dt = options.get('first_step')#,0.1)
    
    while t[-1]<tf:
        y.append(method(dt, f, t[-1], y[-1], args))
        t.append(t[-1] + dt)
    
    return np.array(t), np.array(y)
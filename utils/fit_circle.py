from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D

#-------------------------------------------------------------------------------
# Generate points on circle
# P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
#-------------------------------------------------------------------------------
def generate_circle_by_vectors(t, C, r, n, u):
    n = n/linalg.norm(n)
    u = u/linalg.norm(u)
    P_circle = r*cos(t)[:,newaxis]*u + r*sin(t)[:,newaxis]*cross(n,u) + C
    return P_circle

def generate_circle_by_angles(t, C, r, theta, phi):
    # Orthonormal vectors n, u, <n,u>=0
    n = array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    u = array([-sin(phi), cos(phi), 0])
    
    # P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
    P_circle = r*cos(t)[:,newaxis]*u + r*sin(t)[:,newaxis]*cross(n,u) + C
    return P_circle

#-------------------------------------------------------------------------------
# Generating circle
#-------------------------------------------------------------------------------
r = 2.5               # Radius
C = array([3,3,4])    # Center
theta = 45/180*pi     # Azimuth
phi   = -30/180*pi    # Zenith

t = linspace(0, 2*pi, 100)
P_gen = generate_circle_by_angles(t, C, r, theta, phi)

#-------------------------------------------------------------------------------
# Cluster of points
#-------------------------------------------------------------------------------
t = linspace(-pi, -0.25*pi, 100)
n = len(t)
P = generate_circle_by_angles(t, C, r, theta, phi)

# Add some random noise to the points
P += random.normal(size=P.shape) * 0.1

#-------------------------------------------------------------------------------
# FIT CIRCLE 2D
# - Find center [xc, yc] and radius r of circle fitting to set of 2D points
# - Optionally specify weights for points
#
# - Implicit circle function:
#   (x-xc)^2 + (y-yc)^2 = r^2
#   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
#   c[0]*x + c[1]*y + c[2] = x^2+y^2
#
# - Solution by method of least squares:
#   A*c = b, c' = argmin(||A*c - b||^2)
#   A = [x y 1], b = [x^2+y^2]
#-------------------------------------------------------------------------------
def fit_circle_2d(x, y, w=[]):
    
    A = array([x, y, ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = diag(w)
        A = dot(W,A)
        b = dot(W,b)
    
    # Solve by method of least squares
    c = linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


#-------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(P, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/linalg.norm(n0)
    n1 = n1/linalg.norm(n1)
    k = cross(n0,n1)
    k = k/linalg.norm(k)
    theta = arccos(dot(n0,n1))
    
    # Compute rotated points
    P_rot = zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*cos(theta) + cross(k,P[i])*sin(theta) + k*dot(k,P[i])*(1-cos(theta))

    return P_rot


#-------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
#-------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return arctan2(linalg.norm(cross(u,v)), dot(u,v))
    else:
        return arctan2(dot(n,cross(u,v)), dot(u,v))

def fit_circle_numpy(points):
    #-------------------------------------------------------------------------------
    # (1) Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    #-------------------------------------------------------------------------------
    P_mean = points.mean(axis=0)
    P_centered = points - P_mean
    U,s,V = linalg.svd(P_centered)

    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2,:]
    d = -dot(P_mean, normal)  # d = -<p,n>

    #-------------------------------------------------------------------------------
    # (2) Project points to coords X-Y in 2D plane
    #-------------------------------------------------------------------------------
    P_xy = rodrigues_rot(P_centered, normal, [0,0,1])

    #-------------------------------------------------------------------------------
    # (3) Fit circle in new 2D coords
    #-------------------------------------------------------------------------------
    xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])

    #--- Generate circle points in 2D
    t = linspace(0, 2*pi, 100)
    xx = xc + r*cos(t)
    yy = yc + r*sin(t)

    #-------------------------------------------------------------------------------
    # (4) Transform circle center back to 3D coords
    #-------------------------------------------------------------------------------
    C = rodrigues_rot(array([xc,yc,0]), [0,0,1], normal) + P_mean
    C = C.flatten()[None,:]

    return C, r

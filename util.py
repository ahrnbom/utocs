from math import sqrt

import carla

def loc_dist(a, b):
    return sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def vector_normalize(v:carla.Vector3D):
    norm = v.x**2 + v.y**2 + v.z**2
    new = carla.Vector3D(x=v.x/norm, y=v.y/norm, z=v.z/norm)
    return new 

# long_str(2) -> '0002'
# long_str(42, 3) -> '042'
def long_str(i:int, N:int=4):
    s = str(i)
    n = len(s)
    if n < N:
        s = '0'*(N-n) + s 
    
    return s 

# Removes 'intro' from left part of 'text', raises error if not found
def good_lstrip(text, intro):
    assert(len(intro) <= len(text))
    l = len(intro)
    first = text[:l]
    assert(first == intro)

    return text[l:]

def intr(x):
    return int(round(float(x)))

# Projective flattening, scales homogeneous coordinates so that last coordinate is always one
def pflat(x):
    if len(x.shape) == 1:
        x /= x[-1]
    else:
        x /= x[-1, :]
    return x

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
    if n < 4:
        s = '0'*(N-n) + s 
    
    return s 
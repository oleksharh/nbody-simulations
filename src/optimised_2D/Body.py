import numpy as np


###### This struct is cool and readable ########

# Runtime layout of the structured array.
# 3D vectors simply for future overprovisioning.
body_dtype = np.dtype([
    ("pos", "f8", (3,)),
    ("vel", "f8", (3,)),
    ("acc", "f8", (3,)),
    ("mass", "f8"),
])

if __name__ == "__main__":
    # For reference how to manipulate the structured array

    # This also tells Pylance that each scalar element has the fields on Body.
    bodies = np.zeros(1_000_000_000, dtype=body_dtype).view(np.recarray)

    print(bodies.shape)

    print(bodies, end="\n\n")

    # Getting i body from the array
    print(bodies[0], end="\n\n")

    # Changing i body position
    bodies[0]["pos"] = (1, 2, 3)
    print(bodies[0], end="\n\n")

    # Changing i body mass
    bodies[0]["mass"] = 100.82
    print(bodies[0], end="\n\n")

    print(bodies.mass)


    

    ########## BUT THIS IS MUCH BETTER SoA (structure of arrays over the array of structured as in body_dtype) ###############

    N = 10

    # Separate arrays (Hybrid SoA / Vectorized layout)
    pos = np.zeros((N, 3), dtype=np.float64)  # All positions
    vel = np.zeros((N, 3), dtype=np.float64)  # All velocities
    acc = np.zeros((N, 3), dtype=np.float64)  # All accelerations
    mass = np.zeros(N, dtype=np.float64)        # All masses

    print(pos[0])

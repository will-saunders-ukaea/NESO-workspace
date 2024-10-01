from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt  
import numpy as np  


def triangle_cartesian_to_barycentric(x1, y1, x2, y2, x3, y3, x, y):
    scaling = 1.0 / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    l1 = scaling * ((x2 * y3 - x3 * y2) + (y2 - y3) * x + (x3 - x2) * y)
    l2 = scaling * ((x3 * y1 - x1 * y3) + (y3 - y1) * x + (x1 - x3) * y)
    l3 = scaling * ((x1 * y2 - x2 * y1) + (y1 - y2) * x + (x2 - x1) * y)
    return l1, l2, l3

def triangle_barycentric_to_cartesian(x1, y1, x2, y2, x3, y3, l1, l2, l3):
    x = l1 * x1 + l2 * x2 + l3 * x3
    y = l1 * y1 + l2 * y2 + l3 * y3
    return x, y


class TriangleFunc:
    def __init__(self, x1, y1, x2, y2, x3, y3, func):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.func = func

        self.a1 = func(x1, y1)
        self.a2 = func(x2, y2)
        self.a3 = func(x3, y3)

    def __call__(self, x, y):
        l1, l2, l3 = triangle_cartesian_to_barycentric(
            self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, x, y)

        if ((abs(l1) + abs(l2) + abs(l3)) - 1.0) < 1.0E-10:
            return l1 * self.a1 + l2 * self.a2 + l3 * self.a3
        else:
            return None


if __name__ == "__main__":

    x1 = 0.0
    y1 = 0.0
    x2 = 2.0
    y2 = 0.0
    x3 = 0.0
    y3 = 3.0

    func = lambda x,y: 4.0 + 3.0*x + 2.0*y
    tf = TriangleFunc(x1, y1, x2, y2, x3, y3, func)
    
    N = 1000
    x_values = np.random.uniform(0.0, 3.0, N)
    y_values = np.random.uniform(0.0, 3.0, N)

    x = []
    y = []
    z = []

    for xx, yy in zip(x_values, y_values):
        sample = tf(xx, yy)
        if sample is not None:
            x.append(xx)
            y.append(yy)
            z.append(sample)

    fig = plt.figure(figsize =(16, 9))  
    ax = plt.axes(projection ='3d')  
     
    # Creating color map
    my_cmap = plt.get_cmap('hot')
       
    # Creating plot
    trisurf = ax.plot_trisurf(x, y, z,
                             cmap = my_cmap,
                             linewidth = 0.2, 
                             antialiased = True,
                             edgecolor = 'grey')  
    fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
    ax.set_title('Tri-Surface plot')
     
    # Adding labels
    ax.set_xlabel('X-axis', fontweight ='bold') 
    ax.set_ylabel('Y-axis', fontweight ='bold') 
    ax.set_zlabel('Z-axis', fontweight ='bold')
         
    # show plot
    plt.show()





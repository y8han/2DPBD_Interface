import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    Bottom = 'Bottom_'
    Bottom_A = np.load(Bottom + 'A.npy')
    Bottom_All = np.load(Bottom + 'All.npy')
    Bottom_G = np.load(Bottom + 'G.npy')
    Bottom_S = np.load(Bottom + 'S.npy')
    Upper = 'Upper_'
    Upper_A = np.load(Upper + 'A.npy')
    Upper_All = np.load(Upper + 'All.npy')
    Upper_G = np.load(Upper + 'G.npy')
    Upper_S = np.load(Upper + 'S.npy')
    plt.figure(1)
    plt.plot(Bottom_All,color='k',linewidth=2)
    plt.plot(Bottom_S, color = 'b', linewidth=2)
    plt.plot(Bottom_G, color = 'r', linewidth=2)
    plt.plot(Bottom_A, color='g', linewidth=2)
    #plt.plot(fx_groundtruth_opt,color='r',linewidth=1)
    #plt.legend(labels = ['optimalization', 'DLT', 'GT_DLT', 'GT_opt'], loc = 2)
    plt.legend(labels = ['Deformation Energy', 'Spring Elastic Energy', 'Gravitational Energy', 'Area Conservation Energy'], loc = 2)
    plt.title('Deformation Energy of The Bottom Tissue')
    plt.xlabel("Simulation Steps")
    plt.ylabel("Energy (Joule)")
    #plt.title("fx:100-steps iteration for LM opt")
    plt.grid()
    plt.savefig('Bottom.png')
    plt.show()
    plt.figure(2)
    plt.plot(Upper_All,color='k',linewidth=2)
    plt.plot(Upper_S, color = 'b', linewidth=2)
    plt.plot(Upper_G, color = 'r', linewidth=2)
    plt.plot(Upper_A, color='g', linewidth=2)
    #plt.plot(fx_groundtruth_opt,color='r',linewidth=1)
    #plt.legend(labels = ['optimalization', 'DLT', 'GT_DLT', 'GT_opt'], loc = 2)
    plt.legend(labels = ['Deformation Energy', 'Spring Elastic Energy', 'Gravitational Energy', 'Area Conservation Energy'], loc = 2)
    plt.title('Deformation Energy of The Upper Tissue')
    plt.xlabel("Simulation Steps")
    plt.ylabel("Energy (Joule)")
    #plt.title("fx:100-steps iteration for LM opt")
    plt.grid()
    plt.savefig('Upper.png')
    plt.show()


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import math
from math import factorial
from matplotlib.backends.backend_pdf import PdfPages
import os


#Smoothing function defined here
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

directory_path = os.path.dirname(os.path.abspath(__file__)) + '/CSS/'
fig_num = 1

for filename in os.listdir(directory_path):
    if not filename.endswith('.csv') or (filename != '5_3 (5).csv' and filename != '5_3 (8).csv'):
        continue
    fname = directory_path + filename
    HStress, HStrain, VStress=np.loadtxt(fname, delimiter=',',skiprows=1, unpack=True)

    #Smoothing the experimental data here
    HStress_smoothed=savitzky_golay(HStress,41,3)
    HStrain_smoothed=savitzky_golay(HStrain,41,3)
    VStress_smoothed=savitzky_golay(VStress,41,3)

    #Finding Sign change points - change from postive shear stress to negative shear stress
    a=HStress_smoothed
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    index=np.where(signchange==1)[0]  #This has list of all indices where the sign change happens
    cycles = index[1::2]   #This has list of indices where the sign changes from negative to positive.
    cycles_true=np.zeros(len(cycles))

    #Finding true indices of start of new cycles
    for i in range(0,len(cycles)):
        if (abs(HStress_smoothed[cycles[i]-1]) < HStress_smoothed[cycles[i]]):
            cycles_true[i]=cycles[i]-1
        else:
            cycles_true[i]=cycles[i]

    cycles_true=np.int64(cycles_true)

    print('Number of cycles=', len(cycles_true))

    tangent_modulus = []

    for x in range(len(cycles_true) - 1):
        stress_segment = HStress_smoothed[cycles_true[x]:cycles_true[x + 1]]
        strain_segment = HStrain_smoothed[cycles_true[x]:cycles_true[x + 1]]

        stress_diff = np.diff(stress_segment)
        strain_diff = np.diff(strain_segment)

        tangent_modulus.append(stress_diff / strain_diff)

    col = 4
    rows = 4
    plots_count = 0
    plots_per_page = 16
    
    output_path = directory_path + '/output/' + filename.removesuffix('.csv') + '.pdf'
    with PdfPages(output_path) as pdf:
        plt.figure(fig_num, figsize=(8, 8))
        plt.suptitle(filename + ': Tangent Shear Modulus v. Shear Strain')
        for x in range(len(cycles_true) - 1):
            plt.subplot(rows, 5, plots_count % plots_per_page + 1)
            stress_segment = HStress_smoothed[cycles_true[x]:cycles_true[x + 1]]
            strain_segment = HStrain_smoothed[cycles_true[x]:cycles_true[x + 1]]
            
            # Adjusting the range for tangent modulus calculation
            stress_diff = np.diff(stress_segment)
            strain_diff = np.diff(strain_segment)
            tangent_modulus_segment = stress_diff / strain_diff
            
            plt.plot(range(0, len(tangent_modulus_segment)),  # Adjusting the length of strain_segment
                    tangent_modulus_segment,
                    label='Tangent Modulus')
            
            plt.text(0.5, 0.95, x+1, horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
            plots_count += 1
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.xlabel('Shear Strain', fontsize=8)
            plt.ylabel('Tangent Shear Modulus', fontsize=8)
            plt.subplots_adjust(wspace=0.5)  # Adjust horizontal spacing between subplots
            plt.gca().yaxis.set_label_coords(-0.25, 0.5)  # Adjust y-axis label position
            plt.legend(fontsize=3)
            
            if plots_count % plots_per_page == 0:
                plt.tight_layout()
                pdf.savefig(fig_num)
                plt.close()
                fig_num += 1
                plt.figure(fig_num, figsize=(8, 8))
                plots_count = 0
                
        plt.tight_layout()
        pdf.savefig(fig_num)
        plt.close()

        fig_num += 1

        plt.figure(fig_num)
        plt.suptitle(filename + ': Shear Stress v. Vertical Stress')
        for x in range(len(cycles_true) - 1):
            plt.subplot(rows, 5, x + 1)
            plt.plot(VStress_smoothed[cycles_true[x]:cycles_true[x + 1]],
                     HStress_smoothed[cycles_true[x]:cycles_true[x + 1]])
        pdf.savefig(fig_num)
        plt.close()

    #  fig_num += 1
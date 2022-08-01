import numpy as np
import matplotlib.pyplot as plt

import analyse as ana
import na_tracer

def main():

    na = na_tracer.NetworkAnalyzer()

    resp = na.get_pna_response()

    plt.plot(resp)
    plt.show()

if __name__=="__main__":
    main()
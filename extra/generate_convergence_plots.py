#!/usr/bin/env python3
## ---------------------------------------------------------------------
## Copyright (C) 2019 - 2022 by the lifex authors.
##
## This file is part of lifex.
##
## lifex is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## lifex is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.

## You should have received a copy of the GNU Lesser General Public License
## along with lifex.  If not, see <http://www.gnu.org/licenses/>.
## ---------------------------------------------------------------------

# Author: Federica Botta <federica.botta@mail.polimi.it>.
# Author: Matteo Calafà <matteo.calafa@mail.polimi.it>.


#
# Generate plots for the convergence analysis on DG schemes.
# N.B. requires pandas and matplotlib.
#


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



errors = pd.read_csv("../../../build/PACS_Project/src/errors_Monodomain DUB_2D_w.data", sep='\t')
errors = errors[["l_inf","l_2","h_1","DG"]]
errors = np.array(errors)
errors = errors[errors[:,0]!='x',:].astype(float)
n = errors.shape[0]


if (n > 0):

    x = np.logspace(-2, -n,  num=n, base=2.0)


    k = errors[0,0]
    plt.figure()
    ax = plt.axes()
    plt.loglog(x, errors[:,0], 'tab:blue', marker = 'o', linestyle = '-', label='$L^\infty$ error')
    plt.loglog(x, k*np.power(x,1)/np.power(x[0],1), 'k', linestyle = '-.', label='$h^{-1}$')
    plt.loglog(x, k*np.power(x,2)/np.power(x[0],2), 'k', linestyle = '--', label='$h^{-2}$')
    plt.loglog(x, k*np.power(x,3)/np.power(x[0],3), 'k', linestyle = 'dotted', label = '$h^{-3}$')
    plt.xlabel('h')
    plt.ylabel('$L^\infty$ error')
    plt.legend()
    ax.set_title('Convergence test ($L^\infty$ error)')
    plt.savefig("Linf")



    k = errors[0,1]
    plt.figure()
    ax = plt.axes()
    plt.loglog(x, errors[:,1], 'tab:blue', marker = 'o', linestyle = '-', label='$L^2$ error')
    plt.loglog(x, k*np.power(x,1)/np.power(x[0],1), 'k', linestyle = '-.', label='$h^{-1}$')
    plt.loglog(x, k*np.power(x,2)/np.power(x[0],2), 'k', linestyle = '--', label='$h^{-2}$')
    plt.loglog(x, k*np.power(x,3)/np.power(x[0],3), 'k', linestyle = 'dotted', label = '$h^{-3}$')
    plt.xlabel('h')
    plt.ylabel('$L^2$ error')
    plt.legend()
    ax.set_title('Convergence test ($L^2$ error)')
    plt.savefig("L2")


    k = errors[0,2]
    plt.figure()
    ax = plt.axes()
    plt.loglog(x, errors[:,2], 'tab:blue', marker = 'o', linestyle = '-', label='$H^1$ error')
    plt.loglog(x, k*np.power(x,1)/np.power(x[0],1), 'k', linestyle = '-.', label='$h^{-1}$')
    plt.loglog(x, k*np.power(x,2)/np.power(x[0],2), 'k', linestyle = '--', label='$h^{-2}$')
    plt.loglog(x, k*np.power(x,3)/np.power(x[0],3), 'k', linestyle = 'dotted', label = '$h^{-3}$')
    plt.xlabel('h')
    plt.ylabel('$H^1$ error')
    plt.legend()
    ax.set_title('Convergence test ($H^1$ error)')
    plt.savefig("H1")


    k = errors[0,3]
    plt.figure()
    ax = plt.axes()
    plt.loglog(x, errors[:,3], 'tab:blue', marker = 'o', linestyle = '-', label='$DG$ error')
    plt.loglog(x, k*np.power(x,1)/np.power(x[0],1), 'k', linestyle = '-.', label='$h^{-1}$')
    plt.loglog(x, k*np.power(x,2)/np.power(x[0],2), 'k', linestyle = '--', label='$h^{-2}$')
    plt.loglog(x, k*np.power(x,3)/np.power(x[0],3), 'k', linestyle = 'dotted', label = '$h^{-3}$')
    plt.xlabel('h')
    plt.ylabel('$DG$ error')
    plt.legend()
    ax.set_title('Convergence test ($DG$ error)')
    plt.savefig("DG")

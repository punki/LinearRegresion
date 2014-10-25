#! /usr/bin/python
#
# This is support material for the course "Learning from Data" on edX.org
# http://lionoso.org/learningfromdata/
#
# The software is intented for course usage, no guarantee whatsoever
# Date: Sep 24, 2014
#
# Template for a LIONoso parametric table script.
#
# Generates a table based on input parameters taken from another table or from user input
#
# Syntax:
# When called without command line arguments:
# number_of_inputs
# name_of_input_1 default_value_of_input_1
# ...
# name_of_input_n default_value_of_input_n
# Otherwise, the program is invoked with the following syntax:
# script_name.py input_1 ... input_n table_row_number output_file.csv
# where table_row_number is the row from which the input values are taken (assume it to be 0 if not needed)
#
# To customize, modify the output message with no arguments given and insert task-specific code
# to insert lines (using tmp_csv.writerow) in the output table.

import sys
import os
import random

#
# If there are not enough parameters, optionally write out the number of required parameters
# followed by the list of their names and default values. One parameter per line,
# name followed by tab followed by default value.
# LIONoso will use this list to provide a user friendly interface for the component's evaluation.
#
if len(sys.argv) < 2:
    sys.stdout.write("2\nNumber of tests\t1000\nNumber of training points\t10\n")
    sys.exit(0)

#
# Retrieve the input parameters, the input row number, and the output filename.
#
in_parameters = [float(x) for x in sys.argv[1:-2]]
in_rownumber = int(sys.argv[-2])
out_filename = sys.argv[-1]

#
# Retrieve the output filename from the command line; create a temporary filename
# and open it, passing it through the CSV writer
#
tmp_filename = out_filename + "_"
tmp_file = open(tmp_filename, "w")

# ############################################################################
#
# Task-specific code goes here.
#


def fValue(xne, yne, sepFun):
    ySep = sepFun(xne)
    if yne >= ySep:
        return 1
    else:
        return -1


# return funkcja
def fFunction(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y2 - a * x2
    return lambda x: a * x + b


def computeAllF(xn, yn, f):
    fn = []
    for idx, xne in enumerate(xn):
        yne = yn[idx]
        fn.append(fValue(xne, yne, f))
    return fn


# The following function is a stub for the perceptron training function required in Exercise1-7 and following.
# It currently generates random results.
# You should replace it with your implementation of the
# perceptron algorithm (we cannot do it otherwise we solve the homework for you :)
# This functon takes the coordinates of the two points and the number of training samples to be considered.
# It returns the number of iterations needed to converge and the disagreement with the original function.
def compDisagreement(wx, wy, w0, f):
    xn = [random.uniform(-1, 1) for z in range(1000)]
    yn = [random.uniform(-1, 1) for z in range(1000)]
    fn = computeAllF(xn, yn, f)
    hn = compAllH(xn, yn, wx, wy, w0)
    miss = compMiss(hn, fn)
    return len(miss)/1000.0


def perceptron_training(x1, y1, x2, y2, training_size):
    # return (int (random.gauss(100, 10)), random.random() / training_size)

    xn = [random.uniform(-1, 1) for z in range(training_size)]
    yn = [random.uniform(-1, 1) for z in range(training_size)]
    f = fFunction(x1, y1, x2, y2)
    fn = computeAllF(xn, yn, f)
    wx = 0
    wy = 0
    w0 = 0

    iter = 1
    while True:
        hn = compAllH(xn, yn, wx, wy, w0)
        miss = compMiss(hn, fn)
        if miss == []:
            break
        mIdx = random.choice(miss)
        # learn
        wx = wx + fn[mIdx] * xn[mIdx]
        wy = wy + fn[mIdx] * yn[mIdx]
        w0 += fn[mIdx] * 1
        iter += 1

    return iter, compDisagreement(wx, wy, w0, f)


def compMiss(hn, fn):
    miss = []
    for idx, hne in enumerate(hn):
        if fn[idx] != hne:
            miss.append(idx)
    return miss


def compAllH(xn, yn, wx, wy, w0):
    h = []
    for idx, xne in enumerate(xn):
        yne = yn[idx]
        h.append(copmH(xne, yne, wx, wy, w0))
    return h


def copmH(xne, yne, wx, wy, w0):
    h = wx * xne + wy * yne + w0
    if h >= 0:
        return 1
    else:
        return -1


tests = int(in_parameters[0])
points = int(in_parameters[1])

# Write the header line in the output file, in this case the output is a 3-columns table containing the results
# of the experiments
# The syntax  name::type  is used to identify the columns and specify the type of data
header = "Test number::label,Number of iterations::number,Disagreement::number,Mean::number\n"
tmp_file.write(header)


# Repeat the experiment n times (tests parameter) and store the result of each experiment in one line of the output table
sumIter = 0

for t in range(1, tests + 1):
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    while True:
        x2 = random.uniform(-1, 1)
        y2 = random.uniform(-1, 1)
        if x1 != x2 and y1 != y2:
            break

    iterations, disagreement = perceptron_training(x1, y1, x2, y2, points)
    sumIter += iterations
    line = str(t) + ',' + str(iterations) + ',' + str(disagreement) + ',' + str(sumIter / float(tests)) + '\n'
    tmp_file.write(line)

#
# ############################################################################

#
# Close all input files and the temporary output file.
#
tmp_file.close()

#
# Rename the temporary output file into the final one.
# It's important that the output file only appears when it is complete,
# otherwise LIONoso might read an incomplete table.
#
os.rename(tmp_filename, out_filename)

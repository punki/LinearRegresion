__author__ = 'punki'
import time, random, numpy as np


def error_function(u, v):
    return (u * np.exp(v) - 2 * v * np.exp(-u)) ** 2


def d_error_u(u, v):
    return 2 * np.exp(-2 * u) * (u * np.exp(u + v) - 2 * v) * (np.exp(u + v) + 2 * v)


def d_error_v(u, v):
    return 2 * np.exp(-2 * u) * (u * np.exp(u + v) - 2) * (u * np.exp(u + v) - 2 * v)


print 'Error(1,1)={}'.format(error_function(1, 1))


def gradient_descent(alpha, u, v, ep=10 ** -14, max_iter=1000000):
    iter = 0
    while True:
        error = error_function(u, v)
        if error <= ep:
            print "Huurrrrraaaaa we are done :)"
            break
        if iter >= max_iter:
            print "We have no time to make more progress :("
            break

        iter += 1
        new_u = u - alpha * d_error_u(u, v)
        new_v = v - alpha * d_error_v(u, v)
        u = new_u
        v = new_v
    return u, v, iter


def coordinate_descent(alpha, u, v, max_iter=1000000, ep=10 ** -14):
    iter = 0
    while True:
        error = error_function(u, v)
        if error <= ep:
            print "Huurrrrraaaaa we are done :)"
            break
        if iter >= max_iter:
            print "We have no time to make more progress :("
            break

        iter += 1
        u -= alpha * d_error_u(u, v)
        v -= alpha * d_error_v(u, v)
    return u, v, iter


start_u = 1
start_v = 1
start_error = error_function(start_u, start_v)

(final_u, final_v, final_iter) = gradient_descent(0.1, start_u, start_v)
final_error = error_function(final_u, final_v)
print 'Gradient final u={}, v={}, iter={}, final error={}, start error={}'.format(final_u, final_v, final_iter, final_error,
                                                                         start_error)

(final_u, final_v, final_iter) = coordinate_descent(0.1, start_u, start_v, 150000)
final_error = error_function(final_u, final_v)
print 'Coordinate final u={}, v={}, iter={}, final error={}, start error={}'.format(final_u, final_v, final_iter, final_error,
                                                                         start_error)

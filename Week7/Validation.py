__author__ = 'punki'
import time, random, numpy as np


def load_data_from_file():
    sample_in = np.fromfile('in.dta', float, -1, ' ').reshape(-1, 3)
    sample_out = np.fromfile('out.dta', float, -1, ' ').reshape(-1, 3)
    return sample_in, sample_out


def countDiffElements(a, b):
    diffElements = 0
    for idx, he in enumerate(a):
        if b[idx] != he:
            diffElements += 1
    return diffElements


def transformation(data):
    newData = []
    for val in data:
        x1 = val[0]
        x2 = val[1]
        row = [1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]
        newData.append(row)
    return np.array(newData)


def sqr_error(h_y, y):
    return (h_y - y) ** 2


def linearRegresion(s_in, s_in_y, s_out, s_out_y, learn_w_function):
    w = learn_w_function(s_in, s_in_y)
    h_in_class = map(lambda e: 1 if e >= 0 else -1, s_in.dot(w))
    h_out_y = (s_out).dot(w)
    h_out_class = map(lambda e: 1 if e >= 0 else -1, h_out_y)
    e_in = countDiffElements(h_in_class, s_in_y) / float(len(s_in))
    e_out = countDiffElements(h_out_class, s_out_y) / float(len(s_out))
    sqr_e = sqr_error(h_out_y, s_out_y)
    return e_in, e_out, sqr_e


def with_regularization_function(x, y):
    lambda_value = 10 ** k
    a1 = x.T.dot(x)
    a2 = lambda_value * np.identity(len(a1))
    a3 = a1 + a2
    b1 = x.T.dot(y)
    return np.linalg.inv(a3).dot(b1)


# data
s_in, s_out = load_data_from_file()
without_regularization = (lambda x, y: np.linalg.pinv(x).dot(y))
# ex 1
s_train = s_in[0:25]
s_val = s_in[25:35]

for k in range(3, 8):
    # limit model, k+1 because we are choosing from Q0...Qk
    s_train_t = transformation(s_train)[:, 0:k + 1]
    s_val_t = transformation(s_val)[:, 0:k + 1]
    s_out_t = transformation(s_out)[:, 0:k + 1]
    print('model selection k={} (e_train, e_val): {} out-of-sample error={}'.format(
        k,
        linearRegresion(s_train_t, s_train[:, 2:3], s_val_t, s_val[:, 2:3], without_regularization)[0:2],
        linearRegresion(s_train_t, s_train[:, 2:3], s_out_t, s_out[:, 2:3], without_regularization)[1]))



# ex 3
print('Ex 3')
for k in range(3, 8):
    # limit model, k+1 because we are choosing from Q0...Qk
    s_train_t = transformation(s_train)[:, 0:k + 1]
    s_val_t = transformation(s_val)[:, 0:k + 1]
    s_out_t = transformation(s_out)[:, 0:k + 1]
    print('model selection k={} (e_train, e_val): {} out-of-sample error={}'.format(
        k,
        linearRegresion(s_val_t, s_val[:, 2:3], s_train_t, s_train[:, 2:3], without_regularization)[0:2],
        linearRegresion(s_val_t, s_val[:, 2:3], s_out_t, s_out[:, 2:3], without_regularization)[1]))

print('ex 7 cross validation')

# check two models h(for k==0)(x) = b  and h(for k==1)(x) = ax + b
for p in {2.394170170, 0.8555996, 4.33566130724, 2.55939646346}:
    s_in = np.array([[-1, 0], [p, 1], [1, 0]])
    sqr_error_out_for_models = []
    for k in range(0, 2):
        # leave one out cross validation
        sqr_error_out_for_model = []
        for val_idx, s_val in enumerate(s_in):
            # otherwise it would be scalar
            s_val = np.array([s_val])
            s_train = np.concatenate(((s_in[0:val_idx]), (s_in[val_idx + 1:])))
            s_train_t = transformation(s_train)[:, 0:k + 1]
            s_val_t = transformation(s_val)[:, 0:k + 1]
            sqr_e_out = linearRegresion(s_train_t, s_train[:, 1:2], s_val_t, s_val[:, 1:2], without_regularization)[2]
            sqr_error_out_for_model.append(sqr_e_out)

        avg_sqr_error_model = np.average(sqr_error_out_for_model)
        sqr_error_out_for_models.append(avg_sqr_error_model)
        print 'for k={} e={} avg_e={}'.format(k, sqr_error_out_for_model, avg_sqr_error_model)
    print 'p={} sqr_error_out_for_model={}'.format(p, sqr_error_out_for_models)
    print '\n'

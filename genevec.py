# -*-coding:utf-8-*-

'''
@File       : genevec.py
@Discription: 生成嵌入模型的初始向量类
@Author     : Guangkai Li
@Date:      : 2017/10/10
'''

import numpy as np
from scipy.linalg.misc import norm


class GeneVec(object):
    def __init__(self, n_input, n_hidden, n_output, v_dim):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dim = v_dim

    def initialize(self):
        return np.random.uniform(-6/(self.dim**0.5), 6/(self.dim**0.5))

    def assign_value(self):
        return [self.initialize() for i in range(self.dim)]

    def norm_lst(self, lst):
        var = norm(lst)
        return [l/var for l in lst]

    def genevec_vector(self):
        symp_vec, diag_vec, b1, b2 = [], [], [], []
        for i in range(self.n_input):
            symp_vec.append(self.norm_lst(self.assign_value()))
        for i in range(self.n_output):
            diag_vec.append(self.norm_lst(self.assign_value()))
        for i in range(self.n_hidden):
            b1.append(self.norm_lst(self.assign_value()))
        for i in range(self.n_output):
            b2.append(self.norm_lst(self.assign_value()))
        return np.array(symp_vec), np.array(diag_vec), np.array(b1), np.array(b2)


def main():
    v = GeneVec(2, 2, 2, 20)
    print(v.genevec_vector())


if __name__ == '__main__':
    main()

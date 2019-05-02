# -*-coding:utf-8-*-

'''
@File       : model.py
@Discription: 基于神经网络的症状/诊断嵌入模型
@Author     : Guangkai Li
@Date:      : 2017/10/10
'''

import numpy as np
from copy import deepcopy
from genevec import GeneVec as gv
from scipy.linalg import norm
from sklearn.model_selection import train_test_split


class SympDiagEmbedding(object):
    def __init__(self, x_training, y_training, x_test, y_test, v_dim=20):
        self.x_training = x_training
        self.y_training = y_training
        self.x_test = x_test
        self.y_test = y_test
        self.v_dim = v_dim

        self.n_input = self.x_training.shape[1]
        self.n_output = max(self.y_training) + 1
        self.n_hidden = int(np.sqrt(self.n_input + self.n_output)+5)

        self.symp_vec, self.diag_vec, self.b1, self.b2 = gv(
            self.n_input, self.n_hidden, self.n_output, v_dim).genevec_vector()

        self.W1 = np.random.randn(
            self.n_input, self.n_hidden)/np.sqrt(self.n_input)
        self.W2 = np.random.randn(
            self.n_hidden, self.n_output)/np.sqrt(self.n_hidden)


    def predict(self, x):
		"""
		神经网络前向传递，由输入向量得到预测结果
		
		Args:
			病历特征向量
		
		Returns:
			预测结果
		"""
        size = self.symp_vec.shape[1]
        A = np.tile(x, (size, 1))
        x_input = np.multiply(A.T, self.symp_vec)

        dis = x.dot(self.symp_vec)/sum(x)

        z1 = self.W1.T.dot(x_input) + self.b1
        a1 = np.tanh(z1)
        z2 = self.W2.T.dot(a1) + self.b2

        output = np.diag(z2.dot(self.diag_vec.T)) - \
            np.array([norm(dis - self.diag_vec[j])
                      for j in range(self.n_output)])
        exp_scores = np.exp(output)
        probs = exp_scores/np.sum(exp_scores)

        return np.argsort(probs)[-1:]

    def calculate_loss(self):
		"""
		计算损失函数
		
		Returns:
			损失
		"""
        correct_logprobs = 0
        size = self.symp_vec.shape[1]
        for i in range(len(self.y_training)):
            A = np.tile(self.x_training[i], (size, 1))
            x_input = np.multiply(A.T, self.symp_vec)

            dis = self.x_training[i].dot(
                self.symp_vec) / sum(self.x_training[i])

            z1 = self.W1.T.dot(x_input) + self.b1
            a1 = np.tanh(z1)
            z2 = self.W2.T.dot(a1) + self.b2

            output = np.diag(z2.dot(self.diag_vec.T)) - np.array(
                [norm(dis - self.diag_vec[j]) for j in range(self.n_output)])
            exp_scores = np.exp(output)
            probs = exp_scores/np.sum(exp_scores)

            correct_logprobs += -np.log(probs[self.y_training[i]])
        return correct_logprobs

    def back_propagate(self, i):
		"""
		误差反向传播训练过程
		"""
        size = self.symp_vec.shape[1]
        A = np.tile(self.x_training[i], (size, 1))

        dis = self.x_training[i].dot(self.symp_vec)/sum(self.x_training[i])
        d_symp = np.mat(self.x_training[i]).T.dot(np.mat(
            dis - self.diag_vec[self.y_training[i]]))/norm(dis - self.diag_vec[self.y_training[i]])
        d_diag = np.array([(dis - self.diag_vec[j])/norm(dis -
                                                         self.diag_vec[j]) for j in range(self.n_output)])

        x_input = np.multiply(A.T, self.symp_vec)

        z1 = self.W1.T.dot(x_input) + self.b1
        a1 = np.tanh(z1)
        z2 = self.W2.T.dot(a1) + self.b2

        output = np.diag(z2.dot(self.diag_vec.T)) - \
            np.array([norm(dis - self.diag_vec[j])
                      for j in range(self.n_output)])
        exp_scores = np.exp(output)
        probs = exp_scores/np.sum(exp_scores)

        delta = probs
        delta[self.y_training[i]] -= 1
        delta = np.mat(delta)

        B = np.tile(delta.T, (1, self.v_dim))
        propa = np.multiply(B, self.diag_vec)

        d_b2 = propa
        d_diag = np.multiply(B, z2+d_diag)
        d_W2 = a1.dot(propa.T)

        propa1 = np.multiply(self.W2.dot(propa), (1-np.power(a1, 2)))

        d_b1 = propa1
        d_W1 = x_input.dot(propa1.T)

        propa2 = self.W1.dot(propa1)

        self.W1 += -0.05 * d_W1
        self.W2 += -0.05 * d_W2
        d_symp_vec = np.multiply(A.T, propa2)

        self.symp_vec += -0.05 * d_symp_vec - 0.05 * d_symp - 0.005 * self.symp_vec
        self.b1 += -0.05 * d_b1
        self.b2 += -0.05 * d_b2
        self.diag_vec += -0.05 * d_diag - 0.005 * self.diag_vec

    def train(self, num_pass=300):
        for k in range(num_pass):
            for i in range(len(self.y_training)):
                j = np.random.randint(0, len(self.y_training)-1)
                self.back_propagate(j)
            if k % 10 == 0:
                print("loop:%s,训练集loss:%s,测试集准确率:%s,训练集准确率:%s" % (k+1, self.calculate_loss(), self.pre_test(self.x_test, self.y_test), self.pre_test(self.x_training, self.y_training)))


    def pre_test(self, X, y):
        predict_label = []
        for i in X:
            predict_label.append(self.predict(i)[0])
        return np.mean(predict_label == y)


def main():
    training_vec = np.load('./data/training_vec.npy')
    training_label = np.load('./data/training_label.npy')
    test_vec = np.load('./data/test_vec.npy')
    test_label = np.load('./data/test_label.npy')

    X_train,y_train,X_test,y_test = [],[],[],[]
    for i in range(len(training_label)):
        if sum(training_vec[i]) > 0:
            X_train.append(training_vec[i])
            y_train.append(training_label[i])
    for i in range(len(test_label)):
        if sum(test_vec[i]) > 0:
            X_test.append(test_vec[i])
            y_test.append(test_label[i])

    training_vec, training_label, test_vec, test_label = np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)

    embedd = SympDiagEmbedding(training_vec, training_label, test_vec, test_label)
    embedd.train(300)
    print(embedd.pre_test(test_vec, test_label))


if __name__ == '__main__':
    main()

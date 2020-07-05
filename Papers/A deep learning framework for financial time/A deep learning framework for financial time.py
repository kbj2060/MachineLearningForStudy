"""
	Note:
	    기존의 LSTM을 이용한 주식 예측 자료들의 lagging 현상으로
	    대부분 쓸모 없는 자료들이 많아서 위 논문을 해결책으로 삼아 구성한 프로젝트입니다.
"""

from src.predict import predict
from src.analysis import *
from src.utils import *

import rootpath
import pandas as pd
import numpy as np
import json
import plotly.offline as pyo
import plotly.graph_objects as go
import tensorflow as tf
import os

'''
requirements.txt
	pandas~=1.0.3
	pyqt5~=5.15.0
	rootpath~=0.1.1
	numpy~=1.18.4
	plotly~=4.7.1
	tensorflow~=2.1.0
	dtw~=1.4.0
	scikit-learn~=0.23.1
	requests~=2.23.0
	beautifulsoup4~=4.9.1
	tqdm~=4.46.1
	keras~=2.3.1
	setuptools~=46.4.0
	statsmodels~=0.11.1
'''

# seed 를 고정시켜 저장한 모델을 다시 불러와도 같은 값이 나오도록 합니다.
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class SystemTrading():
	"""
	Note:
	    시스템트레이딩을 목적으로 만든 클래스.

	Args:
	    batch_size (int): LSTM 및 AutoEncoder에 Input 으로 들어갈 배치 사이즈
	    time_steps (int): LSTM에 사용될 time_steps
		lstm_unit_1 (int): lstm 레이어의 units의 갯수
		lstm_unit_2 (int): lstm 레이어의 units의 갯수
		ae_lstm_1 (int): autoencoer 레이어의 units의 갯수
		ae_lstm_2 (int): autoencoer 레이어의 units의 갯수
		lr (float): learning rate
		ae_epoch (int): autoencoder의 epoch 수
		epoch (int): lstm의 epoch 수
		iter (int): lstm의 반복 수
		dropout_size_1 (int): lstm의 dropout 사이즈
		dropout_size_2 (int): lstm의 dropout 사이즈
		features (int): feature의 갯수
		label (int): label의 갯수
	"""
	def __init__(self, subject,
				 batch_size = 10,
				 time_steps = 20,
				 lstm_unit_1=64,
				 lstm_unit_2=32,
				 ae_lstm_1=10,
				 ae_lstm_2=10,
				 lr=0.001,
				 ae_epoch = 1000,
				 epoch = 70,
				 iter = 70,
				 dropout_size_1 = 0.2,
				 dropout_size_2 = 0.4,
				 features = 22,
				 label = 1):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.ae_epoch = ae_epoch
		self.epoch = epoch
		self.iter = iter
		self.lstm_unit_1 = lstm_unit_1
		self.lstm_unit_2 = lstm_unit_2
		self.ae_lstm_1 = ae_lstm_1
		self.ae_lstm_2 = ae_lstm_2
		self.dropout_size_1 = dropout_size_1
		self.dropout_size_2 = dropout_size_2
		self.lr = lr
		self.subject = subject
		self.features = features
		self.label = label


	def analysis(self):
		"""
		Note:
			analysis 함수는 총 4개의 과정으로 이루어져 있습니다.
			[ preprocessing - denoising - autoencoder - lstm - prediction ]
			preprocessing : feature 값들을 정리하고 위 논문에서 제시한 데이터들을 만들어 추가해줍니다.
			denoising : 전체 데이터의 노이즈를 제거해줍니다.
			autoencoder : stacked autoencoder를 이용해 deep feature를 재구성합니다.
			lstm : lstm을 이용하여 학습합니다.
			prediction : lstm 모델을 이용하여 예측합니다.

		Args:
		    self : 클래스 변수

		Returns:
		    int : training data의 갯수가 1000개를 넘지 못하면 리턴값을 0으로 설정.
		"""
		print("--------------- preprocessing ---------------")
		root = rootpath.detect()
		csv = pd.read_csv(root + '/stock/{s}/{s}.csv'.format(s=self.subject)).drop_duplicates()
		_stock = preprocess(csv)
		_stock = sup_indicator(_stock)
		test_num = self.batch_size * 15 + self.time_steps

		print("--------------- denoising ---------------")
		# denoising
		denoised_stock = _stock.copy()
		for col in _stock:
			denoised_stock[col] = wavelet_smooth(_stock[col].values)

		# normalization
		norm_stock = normalize(denoised_stock.copy(), test_num)

		# divied dataset
		train, val, test = divide_dataset(norm_stock, self.batch_size, self.time_steps, test_num)

		# set return condition
		# 학습 데이터가 1000개 이하 시 analysis.err 파일을 이용해 보고합니다.
		threshold = 1000
		if len(train) < threshold:
			report_error('Not Enough Dataset', 'analysis.err')
			return 0

		# divied dataset into X, y
		x_train, y_train = create_dataset(train, self.time_steps)
		x_val, y_val = create_dataset(val, self.time_steps)
		x_test, y_test = create_dataset(test, self.time_steps)

		# reshape X data for autoencoder input size
		x_train = _reshape(x_train, self.features)
		x_val = _reshape(x_val, self.features)
		x_test = _reshape(x_test, self.features)

		print("--------------- auto_encoder ---------------")
		# autoencoder modeling and learning
		# X, y 를 lstm과 다르게 train과 validation을 합쳐 학습시키고 test를 검증 데이터로 활용합니다.
		auto_encoder = AE_modeling(self.batch_size, self.time_steps, self.features,
								   self.ae_lstm_1, self.ae_lstm_2, self.lr)
		auto_encoder = AE_learning(auto_encoder, np.vstack([x_train, x_val]) , np.vstack([x_train, x_val]),
								   x_test, x_test,  self.ae_epoch, self.batch_size)

		# autoencoder predict
		# deep feature를 가진 데이터로 재탄생합니다.
		train_encoded = autoencoder_predict(auto_encoder, x_train, self.batch_size)
		validation_encoded = autoencoder_predict(auto_encoder, x_val, self.batch_size)
		test_encoded = autoencoder_predict(auto_encoder, x_test, self.batch_size)

		# autoencoder check
		# autoencoder의 loss를 plot하여 확인합니다.
		train_mae_loss = list(map(lambda x: np.mean(x), np.mean(np.abs(train_encoded - x_train), axis=1)))
		test_mae_loss = list(map(lambda x: np.mean(x), np.mean(np.abs(test_encoded - x_test), axis=1)))

		ae_assistant_plot(self.subject, self.ae_lstm_1, self.ae_lstm_2,
						  auto_encoder.history.history['loss'],
						  auto_encoder.history.history['val_loss'],
						  train_mae_loss, test_mae_loss)

		print("--------------- LSTM ---------------")
		# LSTM modeling and learning
		model = modeling(self.batch_size, self.time_steps, self.features,
						 self.dropout_size_1, self.dropout_size_2, self.lstm_unit_1, self.lstm_unit_2, self.lr)
		model = learning(model, train_encoded, y_train, validation_encoded, y_val,
						 self.iter, self.epoch, self.batch_size)

		# LSTM model save
		save_model_weight(model, self.time_steps, self.epoch, self.iter, self.batch_size,
						  self.subject, self.lstm_unit_1, self.ae_lstm_1, self.ae_lstm_2)

		print("--------------- prediction ---------------")
		# cut for prediction
		# batch size에 맞게 데이터를 잘라주어야 모델의 Input shape에 맞습니다.
		pn = norm_stock[(len(norm_stock) - self.time_steps) % self.batch_size:]

		# recreate whole dataset
		# 전체 데이터와 마지막 예측 하나를 위해 데이터셋을 만듭니다.
		X, y = create_dataset(pn, self.time_steps)
		X_f = create_dataset(pn[-(self.batch_size + self.time_steps - 1):],
							 self.time_steps, option='future')

		X = _reshape(X, self.features)
		X_f = _reshape(X_f, self.features)

		X_encoded = encoding(auto_encoder, X, self.batch_size)
		X_f_encoded = encoding(auto_encoder, X_f, self.batch_size)

		# 정리된 autoencoder를 거친 데이터셋을 만듭니다.
		all_ec = all_encoded(X_encoded, X_f_encoded, denoised_stock,
							 self.batch_size, self.time_steps)

		# prediction
		# 예측된 다음 날의 Close 값과 나머지 예측 데이터 pred와 cost를 계산하기 위한 pred_test
		future = model.predict(X_f_encoded, batch_size=self.batch_size)[-1]
		pred = model.predict(X_encoded, batch_size=self.batch_size)
		pred_test = model.predict(test_encoded, batch_size=self.batch_size)

		# 예측된 전체 데이터를 모아 denormalize하여
		# 실제 종가 데이터와 예측 종가 데이터를 받습니다.
		pred_candle, y_candle = pred_true_candle(list(pred)+list(future), _stock,
												 denoised_stock, norm_stock, self.time_steps, self.batch_size)

		# test 데이터의 비용을 계산하여 cost.json에 저장합니다.
		cost = calc_cost(pred_test, y_test)
		with open(root + 'cost.json'.format(s=self.subject), 'w+', encoding='UTF-8') as costJson:
			json_str = json.dumps({self.subject: cost}, indent=4, ensure_ascii=False)
			costJson.write(json_str)
			costJson.close()

		# 실제 데이터와 예측 데이터의 그래프를 그립니다.
		fig = draw_graph(y_candle, pred_candle, all_ec, cost)
		plot_name = "{4}_bs{3}ts{0}ep{1}it{2}lstm{5}_{6}_ae{7}_{8}".format(self.time_steps, self.epoch, self.iter, self.batch_size,
																  self.subject, self.lstm_unit_1, self.lstm_unit_2, self.ae_lstm_1, self.ae_lstm_2)
		pyo.plot(fig, filename=root+"/stock/{0}/{1}.html".format(self.subject, plot_name), auto_open=False)


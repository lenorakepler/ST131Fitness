import tensorflow as tf
import numpy as np
import math

class Optimizer():
	def __init__(self, fit_model, fit_model_kwargs, n_epochs=10000, lr=0.005, verbose=False, **kwargs):
		self.fit_model = fit_model(**fit_model_kwargs)

		self.n_epochs = n_epochs
		self.lr = lr
		self.verbose = verbose
		self.save_values = False

		self.values = []
		self.losses = []
		self.epoch_gradients = []

		self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.lr)

	def doOpt(self):
		# tf.debugging.enable_check_numerics()

		# https://www.tensorflow.org/api_docs/python/tf/debugging/check_numerics
		# try:
  		# 	tf.debugging.check_numerics(b, message='Checking b')
		# except Exception as e:
  		# 	assert "Checking b : Tensor had NaN values" in e.message

		optimizer = self.optimizer

		fit_model = self.fit_model
		phylo_loss = self.fit_model.phylo_loss(**self.fit_model.loss_kwargs)

		losses = self.losses
		values = self.values
		epoch_gradients = self.epoch_gradients

		for epoch in range(1, self.n_epochs + 1):
			with tf.GradientTape() as tape:
				c = fit_model.call()
				weights = tf.concat([tf.reshape(v, [-1]) for v in fit_model.trainable_variables], axis=-1)
				loss = phylo_loss.call(c.__dict__, weights=weights)

			# Check if loss is nan/inf
			if not tf.math.is_finite(loss):
				if self.verbose: print(f"Breaking: Loss is NaN (epoch {epoch})")
				if self.debug: breakpoint()
				break
					
			# Check if we have gotten about the same value for the past 5 epochs
			if epoch > 10:	
				if all([math.isclose(l, loss) for l in losses[-5:]]):
					if self.verbose: print(f"Breaking: Last 5 values are the same (epoch {epoch})")
					break

			values.append([v.numpy() for v in fit_model.trainable_variables])
			losses.append(loss.numpy())

			gradients = tape.gradient(loss, fit_model.trainable_variables)

			optimizer.apply_gradients(zip(gradients, fit_model.trainable_variables))

			if epoch % 500 == 0:
				if self.verbose:
					with np.printoptions(precision=4):
						# print(f"{epoch=}, loss={loss.numpy():.3f}, values={[v.numpy() for v in fit_model.trainable_variables]}")
						print(f"{epoch=}, loss={loss.numpy():.3f}")

		if self.save_values:
			self.values = values
			self.losses = losses
			self.names = [v.name for v in fit_model.trainable_variables]

		# with np.printoptions(precision=2):
		# 	print(f"{epoch=}")
		# 	print(f"{values[-1]}")
		
		min_loss = np.nanmin(losses)
		min_loss_loc = np.where(losses == min_loss)[0][0]
		offset = values[min_loss_loc]
		loss = min_loss

		print(f"{min_loss_loc=}")
		
		if self.save_values:
			self.min_loss_loc = min_loss_loc
			self.epoch_gradients = epoch_gradients
			self.offset = offset

		return {v.name.split(":")[0]: v.numpy() for v in fit_model.trainable_variables}, loss

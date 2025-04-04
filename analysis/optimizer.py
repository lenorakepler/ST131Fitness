import tensorflow as tf
import numpy as np
import math

class Optimizer():
	def __init__(self, n_epochs=10000, lr=0.005, verbose=False, **kwargs):
		self.n_epochs = n_epochs
		self.lr = lr
		self.verbose = verbose
		self.save_values = False

		self.values = []
		self.losses = []
		self.epoch_gradients = []

		self.optimizer = tf.keras.optimizers.legacy.Nadam(learning_rate=self.lr)

	def doOpt(self, fit_model, phylo_loss):
		self.fit_model = fit_model

		# tf.debugging.enable_check_numerics()

		# https://www.tensorflow.org/api_docs/python/tf/debugging/check_numerics
		# try:
  		# 	tf.debugging.check_numerics(b, message='Checking b')
		# except Exception as e:
  		# 	assert "Checking b : Tensor had NaN values" in e.message

		optimizer = self.optimizer

		losses = self.losses
		values = self.values
		epoch_gradients = self.epoch_gradients

		for epoch in range(1, self.n_epochs + 1):
			with tf.GradientTape() as tape:
				c = fit_model.call()
				poss_weights = [tf.reshape(v, [-1]) for v in fit_model.trainable_variables if v.name in fit_model.penalize]
				weights = tf.cond(len(poss_weights) > 0, lambda: tf.concat(poss_weights, axis=-1), lambda: np.array([1.01]))
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

			values.append({v.name.split(":")[0]: v.numpy() for v in fit_model.trainable_variables})
			losses.append(loss.numpy())

			gradients = tape.gradient(loss, fit_model.trainable_variables)
			gradients = [tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g for g in gradients]
			optimizer.apply_gradients(zip(gradients, fit_model.trainable_variables))

			if epoch % 500 == 0:
				if self.verbose:
					with np.printoptions(precision=4):
						# print(f"{epoch=}, loss={loss.numpy():.3f}, values={[v.numpy() for v in fit_model.trainable_variables]}")
						print(f"{epoch=}, loss={loss.numpy():.3f}")

		if self.save_values:
			self.values = values
			self.losses = losses

		# with np.printoptions(precision=2):
		# 	print(f"{epoch=}")
		# 	print(f"{values[-1]}")
		
		try:
			min_loss = np.nanmin(losses)
		except:
			print(f"{losses=}")
			return {v.name.split(":")[0]: v.numpy() for v in fit_model.trainable_variables}, np.nan

		min_loss_loc = np.where(losses == min_loss)[0][0]
		best_values = values[min_loss_loc]
		loss = min_loss

		if epoch > 1: print(f"{min_loss_loc=}")
		
		if self.save_values:
			self.min_loss_loc = min_loss_loc
			self.epoch_gradients = epoch_gradients
			self.best_values = best_values

		return best_values, loss

def validate_model(session, val_names, val_ops, batch_size=200):
	""" Validates the model stored in a session.
	Args:
		session: The session where the model is loaded.
		val_data: The validation data to use for evaluating the model.
		val_ops: The validation operations.
	Returns:
		The overall validation error for the model. """

	print ("Validating model...")
	val_num = len(val_names)
	print ("test_num: ", val_num)

	MaxTestIters = val_num/batch_size
	print ("MaxTestIters: ", MaxTestIters)

	val_loss = []
	val_err = []

	iter_start = None

	for iterTest range(MaxTestIters):

		test_start=iterTest * batch_size
		test_end = (iterTest+1) * batch_size

		batch_val_data = load_batch_from_data(val_names, dataset_path, 1000, img_ch, img_cols, img_rows, train_start = test_start, train_end = test_end)

		batch_val_data = prepare_data(batch_val_data)

		val_batch_loss, val_batch_err = sess.run([self.cost, self.err], feed_dict={self.eye_left: batch_val_data[0], \
						self.eye_right: batch_val_data[1], self.face: batch_val_data[2], \
						self.face_mask: batch_val_data[3], self.y: batch_val_data[4]})

		val_loss.append(val_batch_loss)
		val_err.append(val_batch_err)


		if iterTest % 10 == 0:
			print ('IterTest %s, val loss: %.5f, val error: %.5f' % \
										(iterTest, np.mean(val_loss), np.mean(val_err)))

			plot_loss(np.array(train_loss), np.array(train_err), np.array(Val_err), start=0, per=1, save_file="plots/testing_loss_" + str(n_epoch) + "_" + str(iterTest) + ".png")

			if iter_start:
				print ('10 iters runtime: %.1fs' % (timeit.default_timer() - iter_start))
			else:
				iter_start = timeit.default_timer()

	return np.mean(val_err)

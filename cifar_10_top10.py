EPOCHS = 10
BATCH_SIZE = 128
train_callback = TrainTimeCallback()
test_callback = TestTimeCallback()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

history_log = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
    'train_time': [],
    'test_time': []
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    # ---------- Test Phase ----------
    test_callback.on_epoch_end(epoch)
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    history_log['val_loss'].append(test_loss)
    history_log['val_accuracy'].append(test_accuracy)
    
    # ---------- Train Phase ----------
    train_callback.on_epoch_begin(epoch)
    epoch_loss = []
    epoch_accuracy = []

    for step, (x_batch, y_batch) in enumerate(train_dataset):
        metrics = model.train_on_batch(x_batch, y_batch)
        epoch_loss.append(metrics[0])
        epoch_accuracy.append(metrics[1])
        
    train_callback.on_epoch_end(epoch)
    
    avg_train_loss = np.mean(epoch_loss)
    avg_train_accuracy = np.mean(epoch_accuracy)
    
    history_log['loss'].append(avg_train_loss)
    history_log['accuracy'].append(avg_train_accuracy)
    history_log['train_time'].append(train_callback.train_times[-1])
    history_log['test_time'].append(test_callback.test_times[-1])

    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Train seconds: {train_callback.train_times[-1]:.2f}")
    print(f"Test seconds: {test_callback.test_times[-1]:.2f}")

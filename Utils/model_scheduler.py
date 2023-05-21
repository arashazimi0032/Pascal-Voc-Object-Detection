def innovative_model_scheduler(epoch, lr):
    if epoch < 80:
        return 0.001
    elif epoch < 120:
        return 0.0005
    elif epoch < 160:
        return 0.0001
    elif epoch < 200:
        return 0.00005
    else:
        return 0.00001


def scheduler_ssd(epoch, lr):
    if epoch < 100:
        return 0.002
    elif epoch < 160:
        return 0.001
    elif epoch < 240:
        return 0.0008
    elif epoch < 280:
        return 0.0005
    else:
        return 0.0001

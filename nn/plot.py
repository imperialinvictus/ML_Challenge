import matplotlib.pyplot as plt

def plot_values(model, ax):
    """
    Plot the training progress of the neural network models.
    """
    # set validation scores
    ax[0].plot(model.validation_scores_)
    ax[0].set_title('Validation Scores')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Accuracy')

    # set loss curve on the second subplot
    ax[1].plot(model.loss_curve_)
    ax[1].set_title('Loss Curve')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Loss')

def plot_single_model(model, name):
    """
    Plot the training progress of a single neural network model.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_values(model, ax)
    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

def plot_bagging_training_progress(models, name):
    """
    Plot the training progress of the bagging neural network models.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, model in enumerate(models.estimators_):
        print(models)
        plot_values(model, ax)

    # plt.suptitle(name)
    # plt.tight_layout()
    # plt.show()

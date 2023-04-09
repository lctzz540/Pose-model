import matplotlib.pyplot as plt


def exportgraph(train_loss, valid_losses, train_acc, valid_acc):
    plt.plot(train_loss, label="Train")
    plt.plot(valid_losses, label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

    plt.plot(train_acc, label="Train")
    plt.plot(valid_acc, label="Validation")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")
    plt.show()

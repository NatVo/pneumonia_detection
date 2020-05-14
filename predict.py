import torch


def load_conv_model(model_path):
    checkpoint = torch.load(model_path)

    print(f'model path:', model_path)
    print(f'Best val loss: {checkpoint["best_val_loss"]}, Best val accuracy: {checkpoint["best_val_accuracy"]}')
    print(f'Best train loss: {checkpoint["best_train_loss"]}, Best train accuracy: {checkpoint["best_train_accuracy"]}')
    print('-'*150)



if __name__ == '__main__':

    load_conv_model('./checkpoint_test.tar')
    load_conv_model('./checkpoint_val.tar')
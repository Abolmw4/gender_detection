import torch
from data import AbolfazlDataset
from network import AbolfazNework
import cv2
from torchvision.transforms import transforms
from tqdm import tqdm


model = AbolfazNework().to('cuda')
model.load_state_dict(
    torch.load("/home/mohammad/Documents/Gender_classification_train_version/weight/new_train3/Abolfazl15.pt"))
model.eval()

tr = transforms.Compose([transforms.ToTensor(),
                         transforms.Resize((224, 224)),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def calculate_accuracy(y_pred, y):
    correct = (y_pred.argmax(1).to('cuda') == y.argmax(1).to('cuda')).type(torch.float).sum()
    acc = correct / y.shape[0]
    return acc


test_data = AbolfazlDataset("/home/mohammad/Documents/Gender_classification_train_version/newtest2",
                            image_transforms=tr)

x = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)


def evaluate(model, iterator, device='cuda'):
    epoch_acc = 0.0
    model.eval()
    for (x, y) in tqdm(iterator, desc="Evaluating", leave=True):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.float32)
        label = y.to(device)
        y_pred = model(x)
        acc = calculate_accuracy(y_pred, label)
        epoch_acc += acc.item()
    return (epoch_acc / len(iterator)) * 100


print(evaluate(model, iterator=x))


# image = cv2.imread(
#     "/home/mohammad/Documents/Gender_classification_train_version/newtest2/female/WIDER_Track_ID_104_ID_61782_Similarity_0.324581_IQA_score_0.524688_Camera_Number_0_area_conf_1.273345_.jpg", 1)
# ready_image = tr(image).reshape((1, 3, 224, 224))
# with torch.no_grad():
#     output = model(ready_image.to('cuda'))
#
#     if output[0, 0] > output[0, 1]:
#         print("male")
#     else:
#         print("female")

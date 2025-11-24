import torch
import  torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchinfo import summary
import matplotlib.pyplot as plt

#보고서 제출기한: 12월 7일
#제출장소:  portal.kongju.ac.kr 

#다음문제를 pyTorch를 이용한 딥러닝으로 구현하고, 보고서 작성 제출합니다.

#1 문제: MNIST 데이터를 학습하고, 영상에서 숫자를 인식하기기 
# - MNIST  데이터 학습
# - 학습결과 저장
# - 윈도우즈에서 PyThon 응용 프로그램(TkInter, PyQt등) 작성하기
# - 성능 평가하기, 연속된 숫자는 개별인식한 후에 하나의 숫자로 인식해야 합니다.
# - 최대 3자리 정수까지 인식
# - 추가로 시간 있는 학생은  스마트폰에서 구현해보기 바랍니다(보너스 점수).

#주의1: 학생들 사이에 의견교환, 질의/응담 등 가능합니다. 단, 절대 소스를 공유해서는 안됩니다(확인합니다).
#주의2 보고서 역시 인터넷 자료 복사하면 안됩니다(확인 할 수 있는데 까지 확인합니다).
#주의3: 소스코드를 참고한 경우 반드시, 참고문헌에 교재, 웹페이지 등을 추가합니다.
#가능한 교재의 내용을 참고합니다.
#주의4. 이전년도 보고서, 다른학생 보고서 발견시 마이너스  점수입니다.
# 보고서는 15페이지(부록제외, 폰트 10) 이내로 요약 작성하여 학교 portal에 제출합니다. 


#----------------- 보고서 양식 -------------------
#겉표지(제목: 인공지능 학기말 프로젝트, 이름,  학번)

#1. 서론(문제 설명)
#2. 본론(구현내용을 코드, 그래프 등 사용하여 설명)
#3. 실험결과(화면 덤프 및 결과 분석 설명)
#4. 결론(결론, 과제 수행 느낌 등)

torch.manual_seed(1)
torch.cuda.manual_seed(1)
DEVICE='CPU'
print("DEVICE = ", DEVICE)

data_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean = 0.5, std = 0.5)])

PATH = './data'
train_data = MNIST(root = PATH, train= True, download = True, transform = data_transform)

test_ds = MNIST(root = PATH, train = False, download = True, transform = data_transform)

print('train_data.data.shape = ', train_data.data.shape)
print('test_set.data.shape = ', test_ds.data.shape)

valid_ratio = 0.2
train_size=len(train_data)
n_valid = int(train_size * valid_ratio)
n_train = train_size - n_valid
seed = torch.Generator().manual_seed(1)
train_ds, valid_ds = random_split(train_data, [n_train, n_valid], generator=seed)

print('len(train_ds) = ', len(train_ds))
print('len(valid_ds) = ', len(valid_ds))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
print('len(train_loader.dataset) = ', len(train_loader.dataset))
print('len(valid_loader.dataset) = ', len(valid_loader.dataset))
print('len(test_loader.dataset) = ', len(test_loader.dataset))

image, label = train_ds[0]
image = image.squeeze().numpy()
print('image.shape = ', image.shape)
print('label = ', label)

fig, axes = plt.subplots(nrows = 2, ncols=5, figsize=(10,4))
fig.canvas.manager.set_window_title('MNIST Sample Images')
for i in range(10):
    ax = axes[i//5, i%5]
    sample_img = train_data.data[train_data.targets==i][0].numpy()
    ax.imshow(sample_img, cmap='gray')
    ax.set_title(f'Label: {i}')
    ax.axis('off')
fig.tight_layout()
plt.show()

images, labels = next(iter(train_loader))  
images = images*0.5 + 0.5
img_grid = torchvision.utils.make_grid(images[:10], nrow=5, padding = 10, pad_value = 1).permute(1, 2, 0)
fig = plt.figure(figsize=(10,4))
plt.imshow(img_grid)
plt.axis("off")
plt.tight_layout()
plt.show()

class ConvNet(nn.Module):
    def __init__(self, nChannel = 1, nClass = 10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = nChannel, out_channels =16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5))

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*32, nClass))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def train_epoch(train_loader, model, optimizer, loss_fn):
       k = len(train_loader)
       total = 0
       correct = 0
       batch_loss = 0.0
       for X, y in train_loader:
           X, y = X.to(DEVICE), y.to(DEVICE)
           optimizer.zero_grad()
           out = model(X)

           loss = loss_fn(out, y)
           loss.backward()
           optimizer.step()
           y_pred = out.argmax(dim=1).float()
           correct += (y_pred == y).sum().item()
           batch_loss += loss.item()
           total += y.size(0)
       batch_loss /= k
       accuracy = correct / total
       return batch_loss, accuracy

def evaluate(loader, model, loss_fn,
             correct_pred = None, counts = None):
            k = len(loader)
            model.eval()
            total = 0
            correct = 0
            batch_loss = 0.0
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                y_pred = out.argmax(dim=1).float()
                correct += (y_pred == y).sum().item()

                loss = loss_fn(out, y)
                batch_loss += loss.item()
                total += y.size(0)

                if correct_pred and counts:
                    for label, pred in zip(y, y_pred):
                        if label == pred:
                            correct_pred[label] += 1
                        counts[label] += 1
            batch_loss /= k
            accuracy = correct / total
            return batch_loss, accuracy

            def main(EPOCHS = 100):
                model = ConvNet().to(DEVICE)

                summary(model, input_size=(1, 1, 28, 28))
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                loss_fn = nn.CrossEntropyLoss()

                train_losses = []
                valid_losses = []

                print('training........')
                model.train()
                for epoch in range(EPOCHS):
                        loss, acc = train_epoch(train_loader, model, 
                                                optimizer, loss_fn)
                        train_losses.append(loss)

                        val_loss, val_acc = evaluate(valid_loader, model, loss_fn)
                        valid_losses.append(val_loss)

                        if not epoch % 10 or epoch == EPOCHS - 1:
                            msg = f'Epoch: {epoch}: train_loss={loss:.4f}, '
                            msg += f'train_accuracy={acc:.4f}, '
                            msg += f'valid_loss={val_loss:.4f}, '
                            msg += f'valid_accuracy={val_acc:.4f}'
                            print(msg)
                        torch.save(model.state_dict(), './data/3001_mnist.pt')

                        corrects = [0 for i in range(10)]
                        counts = [0 for i in range(10)]
                        test_loss, test_acc = evaluate(test_loader, model, loss_fn,
                                                      corrects, counts)
                        print(f'Test Loss: {test_loss:.4f}, test_accuracy={test_acc:.4f}')

            for i, (c, n) in enumerate(zip(corrects, counts)):
                acc = c / n
                print(f'i={i}: correct:{c:.4f}, count={n:4d}: accuracy={accuracy:4f}')

                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.pyplot(train_losses, label='train_loss')
                plt.pyplot(valid_losses, label='valid_loss')
                plt.legend()
                plt.show()

                if __name__ == '__main__':
                    main()

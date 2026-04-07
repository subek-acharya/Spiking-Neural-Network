import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from spikingjelly.activation_based import surrogate, neuron, functional

import os
import time


from model_architecture.spiking_vgg_cifar import spiking_vgg16_bn_cifar
import utils

# Global variables
device = None
model = None
criterion = None
optimizer = None
trainloader = None
testloader = None
best_acc = 0
T = 4


def train(epoch):
    global model, criterion, optimizer, trainloader, device, T
    
    print('\nEpoch: %d' % epoch)
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Add time dimension: [N, C, H, W] → [T, N, C, H, W]
        inputs_seq = inputs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        
        # Forward pass
        outputs_seq = model(inputs_seq)  # [T, N, 10]
        
        # Average over time
        outputs = outputs_seq.mean(0)  # [N, 10]
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Reset membrane
        functional.reset_net(model)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(batch_idx, len(trainloader), 
                          'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc, model, criterion, testloader, device, T
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            inputs_seq = inputs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            outputs_seq = model(inputs_seq)
            outputs = outputs_seq.mean(0)
            
            loss = criterion(outputs, targets)
            functional.reset_net(model)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, len(testloader), 
                              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'T': T,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/spiking_vgg16_bn_cifar.pth')
        best_acc = acc


def main():
    global device, model, criterion, optimizer, trainloader, testloader, best_acc, T

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPUs: {torch.cuda.device_count()}")

    # Parameters
    T = 4
    batchSize = 64
    learning_rate = 0.01
    num_epochs = 200
    
    best_acc = 0
    start_epoch = 0

    # Load CIFAR-10
    trainloader = utils.GetCIFAR10Training(batchSize)
    testloader = utils.GetCIFAR10Validation(batchSize)

    print("==> Testing data loaders...")
    for images, labels in trainloader:
        print(f"Train batch shape: {images.shape}")  # [64, 3, 32, 32]
        break

    # ------------- CREATE MODEL ----------------
    print("==> Building Spiking VGG16-BN for CIFAR-10...")
    
    model = spiking_vgg16_bn_cifar(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    
    # Set multi-step mode
    functional.set_step_mode(model, 'm')
    
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    cudnn.benchmark = True

    print(f"Timesteps (T): {T}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60)
    training_start_time = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()

    # Summary
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Model: Spiking VGG16-BN (CIFAR-10)")
    print(f"Timesteps: {T}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Training time: {total_training_time/60:.2f} minutes")
    print("="*60)


if __name__ == "__main__":
    main()
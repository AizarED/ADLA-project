import matplotlib.pyplot as plt
from timer import Timer

def train(model, dataloader, criterion, optimiser, epoch, figure, axis, train_losses, timer):
    for batch_idx, batch in enumerate(dataloader):
        x = batch.cuda()
        timer.start('forward pass')
        prediction = model(x)
        timer.stop('forward pass')
        #print('Input shape:', x.shape)
        #print('Prediction shape:', prediction.shape)
        loss = criterion(prediction, x)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
        train_losses.append(loss.item())
        axis.plot(train_losses, 'b')
        figure.canvas.draw()

        if batch_idx == 10:
            #break
            pass



def evaluate(model, dataloader, criterion, epoch, figure, axis, val_losses):
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        x = batch
        prediction = model(x)
        loss = criterion(prediction, x)
        print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
        val_losses.append(loss.item())
        axis.plot(val_losses, 'g')
        figure.canvas.draw()
    model.train()

def getLossPlot():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.ion()
    plt.show()
    return fig, ax
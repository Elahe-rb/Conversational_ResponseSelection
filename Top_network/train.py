import torch
import math
import random
import preprocess_data


#train for each epoch
def train(model, loss_fn, optimizer, train_rows, batch_size, epoch, num_epochs, vocab, device, args, vocabs, models, clusters):

    print('start training ...')
    random.shuffle(train_rows)
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_acc = 0
    losses = []

    num_batches = math.ceil(len(train_rows) / batch_size)    #number of iteration
    log_interval = math.ceil(num_batches/5)

    #progress_bar = tqdm(train_data_loader)
    #for cs, rs, ys in progress_bar:
        # TODO:: check this!
        #cs.to(device)
        #rs.to(device)

    for batch in range(num_batches):
        cs, rs, ys, _ = preprocess_data.process_data(train_rows, batch, batch_size, vocab, args, device, '', '', False)
        cs_mini, rs_mini, ys_mini, clusters_mini = preprocess_data.process_data(train_rows, batch, batch_size, vocab, args, device, clusters, vocabs, True)
        scores_miniModlule = getscoresFromlowLevelNetworks(cs_mini, rs_mini, clusters_mini, models)
        output = model(cs, rs, scores_miniModlule)
        loss = loss_fn(output,ys)

        losses.append(loss.item)
        total_loss += loss.item() * ys.size(0)

        # ToDo:: also test this one
        #with torch.no_grad():
            #acc = ((output > 0).long() == ys.long()).sum().item()

        with torch.no_grad():
            pred = output >= 0.7
            num_correct = (pred == ys.byte()).sum().item()
            total_acc += num_correct

        # ToDo:: check this!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        description = 'Train::: epoch[{}/{}] batch[{}/{}] curr loss: {:.3f}, Loss: {:.3f}, Acc: {:.3f}'.format(
            epoch + 1, num_epochs,
            batch + 1, num_batches,
            loss.item(),
            total_loss / (batch_size * batch + output.size(0)),
            total_acc / (batch_size * batch + output.size(0)))
        if batch % log_interval == 0:
            print(description)

    return model, losses


def getscoresFromlowLevelNetworks(cs, rs, clusters, models):
    scores = []
    for i in range(len(clusters)):
        context = cs[i].view(1,cs.size(1),cs.size(2))
        response = rs[i].view(1,rs.size(1),rs.size(2))
        scores.append(models[clusters[i]](context, response))
    scores = torch.tensor(scores)
    return scores
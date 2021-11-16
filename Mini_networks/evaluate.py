import math
import preprocess_data


def evaluate(model,loss_fn, rows, batch_size, epoch, num_epochs, vocab, max_utt_num, max_utt_length, device, uids_rows, args, is_test):

    #eval mode to set dropout to zero
    model.eval()
    #mrr = 0.0
    acc = 0.0
    count = [0] * 10
    total_acc = 0
    total_loss = 0
    losses = []
    #i=1

    # progress_bar = tqdm(data_loader)
    # for cs, rs, ys in progress_bar:
        # TODO:: check this!
        #cs.to(device)
        #rs.to(device)
    num_batches = math.ceil(len(rows) / batch_size)  # number of iteration
    #log_interval = math.ceil(num_batches / 5)

    if not is_test:
        for batch in range(num_batches):
            cs, rs, ys = preprocess_data.process_data(rows, batch, batch_size, vocab, args, device)

            output = model(cs, rs)  # output dim: b*1     input dim: #batch*seq*featureNum
            loss = loss_fn(output, ys)

            losses.append(loss.item)
            total_loss += loss.item() * ys.size(0)
            pred = output >= 0.7
            num_correct = (pred == ys.byte()).sum().item()
            total_acc += num_correct

            description = 'valid: [{}/{}][{}/{}] curr loss: {:.3f}, Loss: {:.3f}, Acc: {:.3f}'.format(
                epoch + 1, num_epochs,
                batch + 1, num_batches,
                loss.item(),
                total_loss / (batch_size * batch + output.size(0)),
                total_acc / (batch_size * batch + output.size(0)))
            # if batch % log_interval == 0:
            # print(description)
            # print(total_loss / (batch_size * batch + output.size(0)))

        return total_loss / (batch_size * batch + output.size(0))

    else:
        for batch in range(num_batches):
            cs, rs, ys = preprocess_data.process_data(rows, batch, batch_size, vocab, args, device)

            for j in range(int(len(cs) / 10)):  # for each context in batch with its ten candidate responses
                sidx = j * 10
                each_context_result = model(cs[sidx:sidx + 10], rs[sidx:sidx + 10])
                each_context_result = [e.data.cpu().numpy() for e in each_context_result]
                better_count = sum(1 for val in each_context_result[1:] if val >= each_context_result[0])
                count[better_count] += 1  # the model selected response is in betther count position
                # mrr += np.reciprocal((ranks + 1).astype(float)).sum()
                if each_context_result[0] > 0.5:  # here acc is the number of tp+tn/total
                    acc += 1
                acc += sum(1 for val in each_context_result[1:] if val <= 0.5)
            # batch+=1
            '''
            description = (
                'Valid: [{}/{}]  R1: {:.3f} R2: {:.3f} R5: {:.3f} MRR: {:.3f} Acc: {:.3f}'.format(
                    epoch + 1, num_epochs,
                    count[0] / (batch_size * (i - 1) + len(batched_context)),
                    sum(count[:2]) / (batch_size * (i - 1) + len(batched_context)),
                    sum(count[:5]) / (batch_size * (i - 1) + len(batched_context)),
                    mrr / (batch_size * (i - 1) + len(batched_context)),
                    acc / (batch_size * (i - 1) + len(batched_context))
                ))
            if batch % log_interval == 0:
                print(description)
            i += 1
            '''

        r1 = count[0] / (batch_size * (batch - 1) + len(cs))
        r2 = sum(count[:2]) / (batch_size * (batch - 1) + len(cs))
        r5 = sum(count[:5]) / (batch_size * (batch - 1) + len(cs))
        acc = acc / ((batch_size * (batch - 1) + len(cs)) * 10)
        # "MRR": mrr / (batch_size * (i - 1) + len(batched_context)), "Name": type(model).__name__}

        return r1 * 10, r2 * 10, r5 * 10, acc * 10  # it is equal to batchsize/10  because each data contains 10 rows in batch




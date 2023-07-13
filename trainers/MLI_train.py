import torch

from loss import Angular_Isotonic_Loss


def default_train(train_loader, model, optimizer, writer, iter_counter, args):

    if args.gpu_num > 1:
        way = model.module.way
        query_shot = model.module.shots[-1]
        support_shot = model.module.shots[0]

    else:
        way = model.way
        query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()

    criterion = Angular_Isotonic_Loss(args.train_way, args.lamda, args.mrg, args.threshold).cuda()

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr', lr, iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1

        if args.gpu_num > 1:
            inp_spt = inp[:way * support_shot]
            inp_qry = inp[way * support_shot:]
            qry_num = inp_qry.shape[0]
            inp_list = []
            for i in range(args.gpu_num):
                inp_qry_fraction = inp_qry[int(qry_num/i):int(qry_num/(i+1))]
                inp_list.append(torch.cat((inp_spt, inp_qry_fraction), dim=0))
            inp = torch.cat(inp_list, dim=0)

        inp = inp.cuda()

        cos_f3,cos_f4,cos_f2 = model(inp)

        loss_f3 = criterion(cos_f3, target)
        loss_f4 = criterion(cos_f4, target)
        loss_f2 = criterion(cos_f2, target)
        loss = loss_f3 + loss_f4 + loss_f2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        scores = cos_f3 + cos_f4 + cos_f2
        _, max_index = torch.max(scores, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss_value

    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)

    writer.add_scalar('MLI_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc

import argparse
import pyhocon
from tqdm import tqdm
from itertools import combinations
from sklearn.utils import shuffle
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from models import  SpanEmbedder, SpanScorer, FullCrossEncoder
from evaluator import Evaluation
from dataset import CrossEncoderDatasetFull
from utils import *






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_pairwise_long_reg_span.json')
    long = True
    cdmlm = True
    seman = True
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.parse_file(args.config)
    tb = SummaryWriter()

    fix_seed(config)
    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config.model_path)

    # init train and dev set
    train = CrossEncoderDatasetFull(config, 'train')
    train_loader = data.DataLoader(train, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dev = CrossEncoderDatasetFull(config, 'dev')
    dev_loader = data.DataLoader(dev, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    device_ids = config.gpu_num
    device = torch.device("cuda:{}".format(device_ids[0]))

    n_gpu = torch.cuda.device_count()

    if device.type == "cuda":
        torch.cuda.set_device(device)

    ## Models' initiation
    logger.info('Init models')
    span_repr = SpanEmbedder(config, device).to(device)
    span_scorer = SpanScorer(config).to(device)
    cross_encoder_single = FullCrossEncoder(config,long=seman).to(device)
    cross_encoder = torch.nn.DataParallel(cross_encoder_single, device_ids=device_ids)

    if config.training_method in ('pipeline', 'continue') and not config.use_gold_mentions:
        span_repr.load_state_dict(torch.load(config.span_repr_path, map_location=device))
        span_scorer.load_state_dict(torch.load(config.span_scorer_path, map_location=device))


    ## Optimizer and loss function
    criterion = get_loss_function(config)
    optimizer = get_optimizer(config, [cross_encoder])
    scheduler = get_scheduler(optimizer, total_steps=config.epochs * len(train_loader))


    logger.info('Number of parameters of mention extractor: {}'.format(
        count_parameters(span_repr) + count_parameters(span_scorer)))
    logger.info('Number of parameters of the pairwise classifier: {}'.format(
        count_parameters(cross_encoder)))

    ##################################################################################
    ####                    TRAINING
    ##################################################################################



    logger.info('Number of topics: {}'.format(len(train.topics)))
    f1 = []
    for epoch in range(config.epochs):
        logger.info('Epoch: {}'.format(epoch))
        accumulate_loss = 0
        number_of_positive_pairs, number_of_pairs = 0, 0
        cross_encoder.train()
        running_loss = 0.0
        tk = tqdm(train_loader)
        # i = 0
        loss_mini_batch = 0
        optimizer.zero_grad()
        run_f1 = []
        run_prec = []
        run_rec = []
        run_acc = []
        run_loss = []
        run_f1 = 0
        run_prec = 0
        run_rec = 0
        run_acc = 0
        run_loss = 0

        for i, (batch_x, batch_y) in enumerate(tk):
            # if i == len(train_loader) - 2:
            #     break
            if not cdmlm:
                bert_tokens = cross_encoder.module.tokenizer(batch_x, pad_to_max_length=True)
            else:
                bert_tokens = cross_encoder.module.tokenizer(batch_x, pad_to_max_length=True, add_special_tokens=False)
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)
            if long or seman:
                m = input_ids.cpu()
                k = m == cross_encoder_single.vals[0]
                p = m == cross_encoder_single.vals[1]

                v = (k.int() + p.int()).bool()
                nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 4)
                q = torch.arange(m.shape[1])
                q = q.repeat(m.shape[0], 1)

                msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
                msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

                msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
                msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
                msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
                msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

                attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()

            if long:
                input_ids = input_ids[:, :4096]
                attention_mask = attention_mask[:, :4096]
                attention_mask[:,0] = 2
                attention_mask[attention_mask_g == 1] = 2
            if seman:
                arg1 = msk_0_ar.int() * msk_1_ar.int()
                arg2 = msk_2_ar.int() * msk_3_ar.int()
                arg1 = arg1[:, :4096]
                arg2 = arg2[:, :4096]
                arg1 = arg1.to(device)
                arg2 = arg2.to(device)
            else:
                arg1 = None
                arg2 = None
            scores = cross_encoder(input_ids, attention_mask, arg1, arg2)
            loss = criterion(scores, batch_y.to(device))
            loss.mean().backward()
            loss_mini_batch += loss.mean().item()
            torch.nn.utils.clip_grad_norm_(cross_encoder.parameters(), 1.0)


            accumulate_loss += loss.item()/config.batch_size
            number_of_positive_pairs += len((batch_y == 1).nonzero())
            number_of_pairs += len(batch_y)
            strict_preds = (scores > 0).to(torch.int)
            eval = Evaluation(strict_preds, batch_y.to(device))
            rec, prec, f1s, acc = eval.get_recall(), eval.get_precision(), eval.get_f1(), eval.get_accuracy()
            s = i+1
            run_f1 =run_f1*i/s+f1s/s
            run_prec = run_prec*i/s+prec/s
            run_rec = run_rec*i/s+rec/s
            run_acc = run_acc*i/s+acc/s
            run_loss = run_loss*i/s+loss.mean().item()/s

            if (i + 1) % 9== 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                cross_encoder.zero_grad()
                loss_mini_batch = 0
            if (i + 1) % 100 == 0:
                tb.add_scalar('train/loss_per_step',run_loss, i+int(len(train_loader))*epoch)
                tb.add_scalar('train/f1_per_step',run_f1,i+int(len(train_loader))*epoch)
                tb.add_scalar('train/precision_per_step', run_prec, i+int(len(train_loader))*epoch)
                tb.add_scalar('train/recall_per_step', run_rec, i+int(len(train_loader))*epoch)
                tb.add_scalar('train/accuracy_per_step', run_acc, i+int(len(train_loader))*epoch)

        tb.add_scalar('train/loss_per_epoch', np.mean(run_loss), epoch)
        tb.add_scalar('train/f1_per_epoch', np.mean(run_f1), epoch)
        tb.add_scalar('train/precision_per_epoch', np.mean(run_prec),epoch)
        tb.add_scalar('train/recall_per_epoch', np.mean(run_rec), epoch)
        tb.add_scalar('train/accuracy_per_epoch', np.mean(run_acc), epoch)

        logger.info('Number of positive/total pairs: {}/{}'.format(number_of_positive_pairs, number_of_pairs))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))


        logger.info('Evaluate on the dev set')
        all_scores, all_labels = [], []
        number_of_positive_pairs, number_of_pairs = 0, 0
        cross_encoder.eval()
        for i, (batch_x, batch_y) in enumerate(tqdm(dev_loader)):
            if not cdmlm:
                bert_tokens = cross_encoder.module.tokenizer(batch_x, pad_to_max_length=True)
            else:
                bert_tokens = cross_encoder.module.tokenizer(batch_x, pad_to_max_length=True, add_special_tokens=False)
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)
            accumulate_loss = 0
            with torch.no_grad():
                if long or seman:
                    m = input_ids.cpu()
                    k = m == cross_encoder_single.vals[0]
                    p = m == cross_encoder_single.vals[1]

                    v = (k.int() + p.int()).bool()
                    nz_indexes = v.nonzero()[:, 1].reshape(m.shape[0], 4)
                    q = torch.arange(m.shape[1])
                    q = q.repeat(m.shape[0], 1)

                    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
                    msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
                    msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

                    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
                    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
                    msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
                    msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

                    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
                if long:
                    input_ids = input_ids[:, :4096]
                    attention_mask = attention_mask[:, :4096]
                    attention_mask[:, 0] = 2
                    attention_mask[attention_mask_g == 1] = 2
                if seman:
                    arg1 = msk_0_ar.int() * msk_1_ar.int()
                    arg2 = msk_2_ar.int() * msk_3_ar.int()
                    arg1 = arg1[:, :4096]
                    arg2 = arg2[:, :4096]
                    arg1 = arg1.to(device)
                    arg2 = arg2.to(device)
                else:
                    arg1 = None
                    arg2 = None

                scores = cross_encoder(input_ids, attention_mask, arg1, arg2)
                loss = criterion(scores, batch_y.to(device))

                accumulate_loss += loss.item() / config.batch_size

            number_of_positive_pairs += len((batch_y == 1).nonzero())
            number_of_pairs += len(batch_y)


            all_scores.extend(scores.squeeze(1))
            all_labels.extend(batch_y.squeeze(1))

        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)


        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                             len(all_labels)))
        curr_f1 = eval.get_f1()
        curr_prec = eval.get_precision()
        curr_rec= eval.get_recall()
        curr_acc = eval.get_accuracy()

        logger.info('Min score: {}'.format(all_scores.min().item()))
        logger.info('Max score: {}'.format(all_scores.max().item()))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}, Accuracy: {}'.
                    format(curr_rec, curr_prec, curr_f1, curr_acc))


        out_dir = os.path.join(config.model_path, '{}_{}'.format(
            'base' if 'base' in config.bert_model else 'large', config.batch_size),
            'checkpoint_{}'.format(epoch))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        cross_encoder_single.model.save_pretrained(os.path.join(out_dir, 'bert'))
        torch.save(cross_encoder_single.linear.state_dict(), os.path.join(out_dir, 'linear'))

        f1.append(eval.get_f1())
        print(f1)
        tb.add_scalar('dev/loss_per_epoch', accumulate_loss, epoch)
        tb.add_scalar('dev/f1_per_epoch', curr_f1, epoch)
        tb.add_scalar('dev/precision_per_epoch', curr_prec, epoch)
        tb.add_scalar('dev/recall_per_epoch', curr_rec, epoch)
        tb.add_scalar('dev/accuracy_per_epoch', curr_acc, epoch)
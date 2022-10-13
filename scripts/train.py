import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import torchaudio

from openunmix import data
from openunmix import model
from openunmix import utils
from openunmix import transforms

tqdm.monitor_interval = 0


def train(args, unmix, encoder, device, train_sampler, optimizer):
    print('def train')
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        # print(f'x : {x}')
        print(f'x.shape {x.shape}')
        # x = x.reshape(1,x.shape[0]) # modified code
        print('x & y to device')
        x, y = x.to(device), y.to(device)
        print('optimizer.zero_grad as backward pass is to be done')
        optimizer.zero_grad()
        print('x to encoder')
        X = encoder(x)
        print(f'shape of encoded x (X) = {X.shape}')
        print('calling model ummix for computing estimate Y_hat')
        Y_hat = unmix(X)
        print(f'shape of Y-hat  : {Y_hat.shape}')
        print('encoding y (ie, ground truth')
        Y = encoder(y)
        print(f'shape of encoded y {Y.shape}')
        print('computing MSE loss')
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        print('computing gradients via backward passs')
        loss.backward()
        print('updating weights optimizer')
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        print(f'losses list {losses}')
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
    return losses.avg


def valid(args, unmix, encoder, device, valid_sampler):
    print('def valid')
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            X = encoder(x)
            Y_hat = unmix(X)
            Y = encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg


def get_statistics(args, encoder, dataset):
    print('def get_statistics')
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    print(f'dataset {dataset}')
    print(f'dataset_scaler {dataset_scaler}')
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        print(f'x.shape (train.get_statistics) : {x.shape}')
        print(f'type(x) : {type(x)}')
        print(f'y.shape (train.get_statistics) : {y.shape}')
        print(f'type(y) : {type(y)}')
        pbar.set_description("Compute dataset statistics")
        # x = utils.preprocess(x, rate=44100.0, model_rate=44100.0) # modified code
        
        # downmix to mono channel
        print('entering encoder....')
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1) # original statement 
        # X = encoder(x[None, None, ...]).mean(1, keepdim=False).permute(0, 2, 1)  # modified for musdb_imfs
        print(f'x.shape after encoder {x.shape}')
        print('encoder run')
        print(f'X.shape {X.shape}')
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


def main():
    print('def main')
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

    # which target do we want to train?
    parser.add_argument(
        "--target",
        type=str,
        default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )
    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="open-unmix",
        help="provide output path base folder name",
    )
    parser.add_argument("--model", type=str, help="Name or path of pretrained model to fine-tune")
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to resume training")
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument(
        "--patience",
        type=int,
        default=140,
        help="maximum number of train epochs (default: 140)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    # Model Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument("--nfft", type=int, default=4096, help="STFT fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="hidden size parameter of bottleneck layers",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz"
    )
    parser.add_argument(
        "--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=0, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args, _ = parser.parse_known_args()
    
    print('def_main 1 use CUDA or not')
    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f'repo_dir {repo_dir}')
    repo = Repo(repo_dir)
    print(f'repo {repo}')
    commit = repo.head.commit.hexsha[:7]
    print(f'commit {commit}')

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    
    print('def_main2 data.load_datasets called')
    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    # create output dir if not exist
    print('def_main3 create output dir')
    target_path = Path(args.output)
    print(f'target_path {target_path}')
    target_path.mkdir(parents=True, exist_ok=True)
    
    print('def_main4 train_sampler')
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    stft, _ = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate
    )
    print('def_main5 encoder initialized')
    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.checkpoint or args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        print('def_main6 calling get_statistics')
        scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)

    max_bin = utils.bandwidth_to_max_bin(train_dataset.sample_rate, args.nfft, args.bandwidth)

    if args.model:
        # fine tune model
        print(f"Fine-tuning model from {args.model}")
        unmix = utils.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True
        )[args.target]
        unmix = unmix.to(device)
    else:
        print('initialize model.OpenUnmix')
        unmix = model.OpenUnmix(
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_bins=args.nfft // 2 + 1,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            max_bin=max_bin,
        ).to(device)
        print('def_main7 unmix = model.OpenUnmix')

    print('def_main8 calling optimizer')
    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a checkpoint is specified: resume training
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start optimizer from scratch
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        train_loss = train(args, unmix, encoder, device, train_sampler, optimizer)
        valid_loss = valid(args, unmix, encoder, device, valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target,
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            "commit": commit,
        }

        with open(Path(target_path, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()

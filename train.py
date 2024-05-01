import argparse
import copy
import os
import pathlib as pl
from functools import partial
from multiprocessing import Pool
import matplotlib as mpl
import numpy as np
import pandas as pd
import json
import torch as t
import torchmetrics
import pytorch_lightning as plight
import yaml
import datetime
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader as dl
from tqdm.auto import tqdm
import sys
import logging
import models
import utils

parser = argparse.ArgumentParser(description="Run model training.")
parser.add_argument("yaml", type=str, nargs="?", default="main", help="Which yaml config file to use", action="store")
parser.add_argument("gpu_id", type=int, nargs="?", help="Which GPU should be used? -1 for all", action="store")

args = parser.parse_args()
print(args.yaml)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
LOGFILENAME = "full_log.log"
handler = logging.FileHandler(LOGFILENAME, mode="a")
handler_stdout = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d-%(name)s-p%(process)s-{%(pathname)s:%(lineno)d}-%(levelname)s >>> %(message)s",
    "%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
handler_stdout.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(handler_stdout)


class GenerateCallback(plight.Callback):
    def __init__(self, dirpath: str, filename: str, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.accumulated_metrics = []
        self.dirpath = dirpath
        self.filename = filename
        os.makedirs(dirpath, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if "train_loss_epoch" in trainer.logged_metrics or "train_loss" in trainer.logged_metrics:
            if trainer.current_epoch > 0 and trainer.current_epoch % self.every_n_epochs == 0:
                metrics = dict(
                    current_epoch=trainer.current_epoch,
                    train_loss_epoch=trainer.logged_metrics["train_loss_epoch"].item(),
                    val_loss_epoch=trainer.logged_metrics["val_loss"].item(),
                )
                if "acc" in trainer.logged_metrics.keys():
                    metrics["Accuracy"] = trainer.logged_metrics["acc"].item()
                self.accumulated_metrics.append(metrics)
            if len(self.accumulated_metrics):
                metrics_df = pd.DataFrame(self.accumulated_metrics)
                metrics_df.drop(["Accuracy"], axis=1).plot(x="current_epoch")
                plt.grid(visible=True, color="k")
                try:
                    plt.savefig(f"{self.dirpath}/{self.filename}_loss.png")
                    plt.close()
                    metrics_df.drop(["train_loss_epoch", "val_loss_epoch"], axis=1).plot(x="current_epoch")
                    plt.grid(visible=True, color="k")
                    plt.savefig(f"{self.dirpath}/{self.filename}_acc.png")
                except Exception as e:
                    logger.warning(e)
                plt.close()


def collect_fixation_files(
    dataset_folder_idx: int,
    system_data_dir: str,
    dataset_folders: list,
    cfg: dict,
    fixation_files: list = [],
):
    base_dir = f"{system_data_dir}/{dataset_folders[dataset_folder_idx]}"
    base_dir += f"/{cfg['processed_data_folder_name']}"
    real_data_dir = pl.Path(base_dir)
    preprocessed_files = list(real_data_dir.glob(f'*{cfg["processed_data__fixations_file_ending"]}'))

    if len(preprocessed_files):
        logger.info(
            f"Found {len(preprocessed_files)} preprocessed_files for dataset {dataset_folders[dataset_folder_idx]}"
        )
        trial_files = [
            pl.Path(
                str(f).replace(cfg["processed_data__fixations_file_ending"], cfg["processed_data__trial_file_ending"])
            )
            for f in preprocessed_files
        ]
        plot_files = [
            pl.Path(
                (str(f).replace(cfg["processed_data__fixations_file_ending"], "_fixations_words.png")).replace(
                    "processed_data", "plots"
                )
            )
            for f in preprocessed_files
        ]
        assert len(trial_files) == len(
            preprocessed_files
        ), f"len(trial_files) ({len(trial_files)}) !=  len( preprocessed_files ({len(preprocessed_files)}))"

        fixation_files_temp = [
            dict(
                subject_id=x1.stem.split("_")[2] if x1.stem.split("_")[0] == "benf" else x1.stem.split("_")[0],
                preprocessed_fixations_file=x1,
                trial_file=x2,
                dataset_index=dataset_folder_idx,
                plot_file=x3,
            )
            for x1, x2, x3 in zip(preprocessed_files, trial_files, plot_files)
        ]
    else:
        logger.warning(f"No files found for {real_data_dir}")
        return fixation_files

    fixation_files.extend(fixation_files_temp)
    return fixation_files


def make_lists(
    fpath: dict,
    samplelist: list,
    targetlist: list,
    meanlist: list,
    stdlist: list,
    trialslist: list,
    cfg: dict,
    dont_continue_lists=False,
):
    """Takes in filepath, config dict and empty or partially filled lists and adds to lists."""
    if dont_continue_lists:
        samplelist = []
        targetlist = []
        meanlist = []
        stdlist = []
        trialslist = []

    try:
        with open(fpath["trial_file"], "r") as json_file:
            trial_str = json_file.read()
            trial = json.loads(trial_str)

        if "plot_file" in fpath:
            trial["plot_file"] = fpath["plot_file"]
    except Exception as e:
        logger.warn(e)
        logger.warn(f"Skipping file {fpath['trial_file']}")
        return samplelist, targetlist, meanlist, stdlist, trialslist
    if "y_midline" in trial and len(trial["y_midline"]) < 2:
        logger.info(
            f'Skipping {fpath["preprocessed_fixations_file"]} because of len(trial["y_midline"]) {len(trial["y_midline"])}'
        )
        return samplelist, targetlist, meanlist, stdlist, trialslist
    if "subject_id" not in trial and "subject_id" in fpath:
        trial["subject_id"] = fpath["subject_id"]
    if "trial_file" not in trial and "trial_file" in fpath:
        trial["trial_file"] = str(fpath["trial_file"])
    if "preprocessed_fixations_file" not in trial and "preprocessed_fixations_file" in fpath:
        trial["preprocessed_fixations_file"] = fpath["preprocessed_fixations_file"]
    if "plot_file" not in trial and "plot_file" in fpath:
        trial["plot_file"] = fpath["plot_file"]
    dffix = pd.read_csv(fpath["preprocessed_fixations_file"])
    if dffix.shape[0] < 2:
        logger.info(f'Skipping {fpath["preprocessed_fixations_file"]} because of dffix shape {dffix.shape}')
        return samplelist, targetlist, meanlist, stdlist, trialslist
    if "duration" not in dffix.columns and "fix_duration" in dffix.columns:
        dffix["duration"] = dffix.fix_duration
    elif "corrected_start_time" in dffix.columns:
        dffix["duration"] = dffix.corrected_end_time - dffix.corrected_start_time

    if "dffix" not in trial:
        trial["dffix"] = dffix
    if "fname" not in trial:
        trial["fname"] = str(fpath["preprocessed_fixations_file"])
    if "filename" not in trial:
        trial["filename"] = str(fpath["preprocessed_fixations_file"])

    if "synctime_time" in trial and "end_time" in trial:
        trial_time = (trial["end_time"] - trial["synctime_time"]) / 1000
    elif "start_time" in trial and "end_time" in trial:
        trial_time = (trial["end_time"] - trial["start_time"]) / 1000
    else:
        trial_time = None
    if "dffix" in trial and "duration" in trial["dffix"].columns:
        total_fixation_time = trial["dffix"].duration.sum()
        trial["total_fixation_time"] = total_fixation_time
        trial["total_fixation_time_in_seconds"] = total_fixation_time / 1000
        if trial_time is None:
            trial_time = (
                trial["dffix"]["corrected_end_time"].iloc[-1] - trial["dffix"]["corrected_start_time"].iloc[0]
            ) / 1000

    trial["trial_time_in_seconds"] = trial_time

    trialslist.append(trial)

    return samplelist, targetlist, meanlist, stdlist, trialslist


def make_list_loop(fixation_files: list, cfg: dict, num_workers: int = None, use_multiprocessing=True):
    samplelist, targetlist, meanlist, stdlist, trialslist = [], [], [], [], []
    preprocessed_fixations_files = [x for x in fixation_files if "preprocessed_fixations_file" in x]

    if num_workers == 0:
        num_workers = 1

    if use_multiprocessing:
        if len(preprocessed_fixations_files):
            with Pool(processes=num_workers) as pool:
                results = pool.map(
                    partial(
                        make_lists,
                        samplelist=[],
                        targetlist=[],
                        meanlist=[],
                        stdlist=[],
                        trialslist=[],
                        cfg=cfg,
                        dont_continue_lists=True,
                    ),
                    preprocessed_fixations_files,
                )
                if results:
                    for samples, targets, means, stds, trials in results:
                        samplelist.extend(samples), targetlist.extend(targets), meanlist.extend(means), stdlist.extend(
                            stds
                        ), trialslist.extend(trials)

    else:
        if len(preprocessed_fixations_files):
            for fpath in tqdm(
                preprocessed_fixations_files,
                desc="Building lists for preprocessed fixation files",
            ):
                samplelist, targetlist, meanlist, stdlist, trialslist = make_lists(
                    fpath, samplelist, targetlist, meanlist, stdlist, trialslist, cfg
                )

    return samplelist, targetlist, meanlist, stdlist, trialslist


def get_samples_from_trials(trialslist, cfg, trialslist_unrolled, dataset_folder_idx=None):
    samplelist = []
    targetlist = []
    meanlist = []
    stdlist = []
    for sampleidx in range(len(trialslist)):
        for trialidx in tqdm(range(len(trialslist[sampleidx])), desc=f"On dataset index {dataset_folder_idx}"):
            trial = trialslist[sampleidx][trialidx]
            fpath = str(trial["plot_file"])

            trialslist[sampleidx][trialidx]["fname"] = fpath
            trialslist[sampleidx][trialidx]["plot_file"] = fpath
            trial["plot_file_words_combo_rgb"] = fpath.replace("fixations_words", "fixations_words_combo_rgb")
            trial["plot_file_words_combo_grey"] = fpath.replace("fixations_words", "fixations_words_combo_grey")
            trial["plot_file_fix_scatter_grey"] = fpath.replace("fixations_words", "fixations_fix_scatter_grey")
            trial["plot_file_word_boxes_grey"] = fpath.replace("fixations_words", "fixations_word_boxes_grey")
            trial["plot_file_words_grey"] = fpath.replace("fixations_words", "fixations_words_grey")

            dffix = trial["dffix"]
            sample_tensor = t.tensor(dffix.loc[:, cfg["sample_cols"]].to_numpy(), dtype=t.float32)

            if cfg["add_line_overlap_feature"]:
                sample_tensor = utils.add_line_overlaps_to_sample(trial, sample_tensor)
            has_nans = t.any(t.isnan(sample_tensor))
            assert not has_nans, f"NaNs found in sample tensor for {fpath}"

            if "assigned_line" not in dffix:
                target_tensor = None
            else:
                target_tensor = t.tensor(dffix.loc[:, ["assigned_line"]].to_numpy(), dtype=t.float32)

            if target_tensor is not None:
                has_nans = t.any(t.isnan(target_tensor))
                assert not has_nans, f"NaNs found in target tensor for {fpath}"

            samplelist.append(sample_tensor)
            targetlist.append(target_tensor)
            trialslist_unrolled.append(trial)

            mean_vals = t.mean(sample_tensor, dim=0).numpy()
            assert all(
                ~np.isnan(mean_vals)
            ), f"Nans in mean for {str(fpath['preprocessed_fixations_file'])} dffix.shape is {dffix.shape}, id {trial['trial_id']}"
            meanlist.append(mean_vals)
            std_vals = t.std(sample_tensor, dim=0).numpy()
            assert all(
                ~np.isnan(std_vals)
            ), f"Nans in std for {str(fpath['preprocessed_fixations_file'])} dffix.shape is {dffix.shape}, id {trial['trial_id']}"
            stdlist.append(std_vals)
    return samplelist, targetlist, trialslist_unrolled, meanlist, stdlist


def get_samples_and_trials_for_single_dset(
    cfg,
    dataset_folder_idx,
    save_path,
    system_data_dir,
    dataset_folders,
):
    (fixation_files) = collect_fixation_files(
        dataset_folder_idx,
        system_data_dir,
        dataset_folders,
        cfg,
        [],
    )
    _, _, _, _, trialslist_temp = make_list_loop(
        fixation_files,
        cfg,
        use_multiprocessing=cfg["use_multiprocessing"],
        num_workers=cfg["num_workers"],
    )

    if (
        "Sim" in cfg["dataset_folders"][dataset_folder_idx]
        and cfg["restrict_sim_data_to"] > -1
        and len(trialslist_temp) > cfg["restrict_sim_data_to"]
    ):
        trialslist_temp = trialslist_temp[: cfg["restrict_sim_data_to"]]

    (
        samplelist_temp,
        targetlist_temp,
        trialslist_temp_unrolled,
        meanlist_temp,
        stdlist_temp,
    ) = get_samples_from_trials([trialslist_temp], cfg, [], dataset_folder_idx=dataset_folder_idx)

    return (
        samplelist_temp,
        targetlist_temp,
        trialslist_temp_unrolled,
        meanlist_temp,
        stdlist_temp,
    )


def load_and_preprocess_datafiles(cfg: dict):
    system_data_dir = cfg["system_data_dir"]
    dataset_folders = cfg["dataset_folders"]
    pickle_save_str = cfg["pickle_save_str"]
    save_path = pl.Path(f"{pickle_save_str}")

    samplelist = []
    targetlist = []
    meanlist = []
    stdlist = []
    trialslist = []
    samplelist_eval = []
    targetlist_eval = []
    trialslist_eval = []
    save_path.mkdir(exist_ok=True)

    for dataset_folder_idx in cfg["dataset_folder_idx_training"]:
        (
            samplelist_temp,
            targetlist_temp,
            trialslist_temp_unrolled,
            meanlist_temp,
            stdlist_temp,
        ) = get_samples_and_trials_for_single_dset(
            cfg,
            dataset_folder_idx,
            save_path,
            system_data_dir,
            dataset_folders,
        )
        samplelist.extend(samplelist_temp)
        targetlist.extend(targetlist_temp)
        trialslist.extend(trialslist_temp_unrolled)
        meanlist.extend(meanlist_temp)
        stdlist.extend(stdlist_temp)
    if cfg["dataset_folder_idx_training"] != cfg["dataset_folder_idx_evaluation"]:
        for dataset_folder_idx in cfg["dataset_folder_idx_evaluation"]:
            (
                samplelist_temp,
                targetlist_temp,
                trialslist_temp_unrolled,
                meanlist_temp,
                stdlist_temp,
            ) = get_samples_and_trials_for_single_dset(
                cfg,
                dataset_folder_idx,
                save_path,
                system_data_dir,
                dataset_folders,
            )
            samplelist_eval.extend(samplelist_temp)
            targetlist_eval.extend(targetlist_temp)
            trialslist_eval.extend(trialslist_temp_unrolled)

    return (
        trialslist,
        trialslist_eval,
        meanlist,
        stdlist,
        samplelist,
        samplelist_eval,
        targetlist,
        targetlist_eval,
    )


def get_mean_std_values(cfg: dict, meanlist: list, stdlist: list):
    sample_means = t.tensor(np.mean(np.stack(meanlist, axis=0), axis=0), dtype=t.float32)
    sample_std = t.tensor(np.mean(np.stack(stdlist, axis=0), axis=0), dtype=t.float32)
    return sample_means, sample_std


def make_chars_coord_list(trialslist: list, use_char_bounding_boxes=False, use_words=False):
    chars_center_coords_list = []
    chars_lists_lengths = []

    for trial in trialslist:
        if use_words:
            chars_df = pd.DataFrame(trial["words_list"])
            prefix = "word"
        else:
            chars_df = pd.DataFrame(trial["chars_list"])
            prefix = "char"
        if f"{prefix}_x_center" not in chars_df.columns:
            chars_df[f"{prefix}_x_center"] = (chars_df[f"{prefix}_xmax"] - chars_df[f"{prefix}_xmin"]) / 2 + chars_df[
                f"{prefix}_xmin"
            ]
            chars_df[f"{prefix}_y_center"] = (chars_df[f"{prefix}_ymax"] - chars_df[f"{prefix}_ymin"]) / 2 + chars_df[
                f"{prefix}_ymin"
            ]
        if use_char_bounding_boxes:
            chars_center_coords = t.tensor(
                chars_df.loc[
                    :,
                    [
                        f"{prefix}_xmin",
                        f"{prefix}_ymin",
                        f"{prefix}_xmax",
                        f"{prefix}_ymax",
                    ],
                ].values,
                dtype=t.float32,
            )
        else:
            chars_center_coords = t.tensor(
                chars_df.loc[:, [f"{prefix}_x_center", f"{prefix}_y_center"]].values,
                dtype=t.float32,
            )
        chars_center_coords_list.append(chars_center_coords)
        chars_lists_lengths.append(chars_center_coords.shape[0])
    return chars_center_coords_list, chars_lists_lengths


def norm_by_line_widths(sampleidx, samplelist, trialslist, cfg, add_normalised_values_as_features):
    trial = trialslist[sampleidx]
    chars_df = pd.DataFrame(trial["chars_list"])
    linewidth = chars_df.char_xmax.max() - chars_df.char_xmin.min()
    normalised_x = samplelist[sampleidx][:, :1] / linewidth
    if add_normalised_values_as_features:
        samplelist[sampleidx] = t.cat([samplelist[sampleidx], normalised_x], dim=1)
    else:
        samplelist[sampleidx][:, :1] = normalised_x
    return samplelist, trialslist


def zero_line_numberings(trialslist: list, targetlist: list, min_dict: dict = None):
    """Checks minimum line for every target in list and
    substracts the minimum of the dataset for each target in the dataset"""

    if min_dict is None:
        min_dict = {}
    for tridx, trial in enumerate(trialslist):
        dset_name = str(pl.Path(trial["filename"]).parent)
        min_for_trial = targetlist[tridx].min().numpy().item()
        if dset_name in min_dict.keys():
            if min_for_trial < min_dict[dset_name]:
                min_dict[dset_name] = min_for_trial
        else:
            min_dict[dset_name] = min_for_trial
        if min_for_trial < 0:
            logging.warning("Found negative value")
    if len(min_dict):
        for tridx, trial in enumerate(trialslist):
            dset_name = str(pl.Path(trial["filename"]).parent)
            if dset_name in min_dict:
                targetlist[tridx] -= min_dict[dset_name]
                min_for_trial = targetlist[tridx].min().detach().numpy().item()
                if min_for_trial < 0:
                    logging.warning("Found negative value")
    return trialslist, targetlist, min_dict


def pad_sequence(
    samplelist: list,
    targetlist: list,
    sample_means: t.Tensor,
    sample_std: t.Tensor,
    max_seq_length: int,
    target_padding_number=-1,
    padding_at_end=False,
    input_padding_val=0,
    input_is_chars=False,
):
    """Normalises samples by mean and std and
    adds padding to tensors so the list can be stacked into batches."""

    if targetlist is None:
        targetlist = [None for _ in samplelist]
    samples_padded_list, targets_padded_list, padding_list = [], [], []
    sample_shape = samplelist[0].shape[1]

    for s, target in tqdm(zip(samplelist, targetlist), desc="Applying final normalization and padding"):
        if sample_shape == sample_means.shape[0] and not input_is_chars:
            s = (s - sample_means) / sample_std
        elif sample_shape == sample_std.shape[0] * 2:
            s[:, :2] = (s[:, :2] - sample_means) / sample_std
            s[:, 2:] = (s[:, 2:] - sample_means) / sample_std
        elif sample_shape == sample_std[:2].shape[0] * 2:
            s[:, :2] = (s[:, :2] - sample_means[:2]) / sample_std[:2]
            s[:, 2:] = (s[:, 2:] - sample_means[:2]) / sample_std[:2]
        elif sample_shape == sample_std[:2].shape[0]:
            s = (s - sample_means[:2]) / sample_std[:2]
        else:
            raise NotImplementedError(
                f"sample_shape {sample_shape} not compatible with sample_means {sample_means.shape}"
            )

        sequence_length_diff = max_seq_length - s.shape[0]
        padding_list.append(sequence_length_diff)

        if sequence_length_diff > 0:
            s_pad = t.ones((sequence_length_diff, s.shape[1]), dtype=s.dtype, device=s.device) * input_padding_val
            if padding_at_end:
                s_padded = t.cat((s, s_pad), dim=0)
            else:
                s_padded = t.cat((s_pad, s), dim=0)
            samples_padded_list.append(s_padded)

            if target is not None:
                target_pad = (
                    t.ones((sequence_length_diff, target.shape[1]), dtype=target.dtype, device=target.device)
                    * target_padding_number
                )
                if padding_at_end:
                    target_padded = t.cat((target, target_pad), dim=0)
                else:
                    target_padded = t.cat((target_pad, target), dim=0)
                targets_padded_list.append(target_padded)
        else:
            samples_padded_list.append(s)
            targets_padded_list.append(target)

    samples_arr_padded = t.stack(samples_padded_list, dim=0)
    if target is not None:
        target_arr = t.stack(targets_padded_list, dim=0)
    else:
        target_arr = None
    return samples_arr_padded, target_arr, padding_list


def prep_input(cfg: dict):
    (
        trialslist,
        trialslist_eval,
        meanlist,
        stdlist,
        samplelist,
        samplelist_eval,
        targetlist,
        targetlist_eval,
    ) = load_and_preprocess_datafiles(cfg)

    if len(meanlist):
        sample_means_unscaled, sample_std_unscaled = get_mean_std_values(cfg, meanlist, stdlist)

        sample_means_eval = t.stack([x.mean(dim=0) for x in samplelist_eval], dim=0).mean(dim=0)
        sample_std_eval = t.stack([x.std(dim=0) for x in samplelist_eval], dim=0).mean(dim=0)
        cfg["sample_means_unscaled"] = [
            round(float(x), 4) for x in t.detach(sample_means_unscaled).detach().cpu().numpy()
        ]
        cfg["sample_std_unscaled"] = [round(float(x), 4) for x in t.detach(sample_std_unscaled).detach().cpu().numpy()]

        cfg["sample_means_eval_unscaled"] = [
            round(float(x), 4) for x in t.detach(sample_means_eval).detach().cpu().numpy()
        ]
        cfg["sample_std_eval_unscaled"] = [round(float(x), 4) for x in t.detach(sample_std_eval).detach().cpu().numpy()]
    if cfg["use_embedded_char_pos_info"] and cfg["method_chars_into_model"] in [
        "bert",
        "dense",
    ]:
        chars_center_coords_list, chars_lists_lengths = make_chars_coord_list(
            trialslist, cfg["use_char_bounding_boxes"], cfg["use_words_coords"]
        )
        chars_center_coords_list_eval, chars_lists_lengths_eval = make_chars_coord_list(
            trialslist_eval, cfg["use_char_bounding_boxes"], cfg["use_words_coords"]
        )
        if cfg["set_max_chars_seq_len_manually"]:
            max_len_chars_list = cfg["manual_max_chars_sequence_len"]
            max_len_chars_list_in_data = max([max(chars_lists_lengths), max(chars_lists_lengths_eval)])
            assert (
                max_len_chars_list >= max_len_chars_list_in_data
            ), f"max_len_chars_list_in_data {max_len_chars_list_in_data} is larger than max_len_chars_list {max_len_chars_list}"
        else:
            max_len_chars_list = max([max(chars_lists_lengths), max(chars_lists_lengths_eval)])
        cfg["max_len_chars_list"] = max_len_chars_list
        logger.info(f'Max sequence length for chars/words used is {cfg["max_len_chars_list"]}')
    else:
        chars_center_coords_list = None
        chars_center_coords_list_eval = None
        cfg["max_len_chars_list"] = max_len_chars_list = 0

    if cfg["set_max_seq_len_manually"]:
        cfg["max_seq_length"] = int(cfg["manual_max_sequence_len"])
    else:
        seq_lengths = [x.shape[0] for x in samplelist] + [x.shape[0] for x in samplelist_eval]
        cfg["max_seq_length"] = int(np.max(seq_lengths))
    logger.info(f'Max sequence length used is {cfg["max_seq_length"]}')

    if cfg["norm_coords_by_letter_min_x_y"]:
        for sample_idx in tqdm(range(len(samplelist)), desc="Applying norm_coords_by_letter_min_x_y to samplelist"):
            (
                trialslist,
                samplelist,
                chars_center_coords_list,
            ) = utils.norm_coords_by_letter_min_x_y(
                sample_idx,
                trialslist,
                samplelist,
                chars_center_coords_list=chars_center_coords_list,
            )

        for sample_idx in tqdm(
            range(len(samplelist_eval)), desc="Applying norm_coords_by_letter_min_x_y to samplelist_eval"
        ):
            (
                trialslist_eval,
                samplelist_eval,
                chars_center_coords_list_eval,
            ) = utils.norm_coords_by_letter_min_x_y(
                sample_idx,
                trialslist_eval,
                samplelist_eval,
                chars_center_coords_list=chars_center_coords_list_eval,
            )

    if cfg["normalize_by_line_height_and_width"]:
        meanlist, stdlist = [], []
        meanlist_eval, stdlist_eval = [], []
        for sample_idx in tqdm(
            range(len(samplelist)), desc="Applying normalize_by_line_height_and_width to samplelist"
        ):
            (
                trialslist,
                samplelist,
                meanlist,
                stdlist,
                chars_center_coords_list,
            ) = utils.norm_coords_by_letter_positions(
                sample_idx,
                trialslist,
                samplelist,
                meanlist,
                stdlist,
                return_mean_std_lists=True,
                norm_by_char_averages=cfg["norm_by_char_averages"],
                chars_center_coords_list=chars_center_coords_list,
                add_normalised_values_as_features=cfg["add_normalised_values_as_features"],
            )

        for sample_idx in tqdm(
            range(len(samplelist_eval)), desc="Applying normalize_by_line_height_and_width to samplelist_eval"
        ):
            (
                trialslist_eval,
                samplelist_eval,
                meanlist_eval,
                stdlist_eval,
                chars_center_coords_list_eval,
            ) = utils.norm_coords_by_letter_positions(
                sample_idx,
                trialslist_eval,
                samplelist_eval,
                meanlist_eval,
                stdlist_eval,
                return_mean_std_lists=True,
                norm_by_char_averages=cfg["norm_by_char_averages"],
                chars_center_coords_list=chars_center_coords_list_eval,
                add_normalised_values_as_features=cfg["add_normalised_values_as_features"],
            )
    sample_means = t.stack([x.mean(dim=0) for x in samplelist], dim=0).mean(dim=0)
    sample_std = t.stack([x.std(dim=0) for x in samplelist], dim=0).mean(dim=0)

    sample_means_eval = t.stack([x.mean(dim=0) for x in samplelist_eval], dim=0).mean(dim=0)
    sample_std_eval = t.stack([x.std(dim=0) for x in samplelist_eval], dim=0).mean(dim=0)

    cfg["sample_means"] = [round(float(x), 4) for x in t.detach(sample_means).detach().cpu().numpy()]
    cfg["sample_std"] = [round(float(x), 4) for x in t.detach(sample_std).detach().cpu().numpy()]
    cfg["sample_means_eval"] = [round(float(x), 4) for x in t.detach(sample_means).detach().cpu().numpy()]
    cfg["sample_std_eval"] = [round(float(x), 4) for x in t.detach(sample_std).detach().cpu().numpy()]

    if cfg["zero_line_numbers_before_padding"]:
        mins_train = np.unique([x.numpy().min() for x in targetlist])
        mins_eval = np.unique([x.numpy().min() for x in targetlist_eval])
        if len(mins_train) > 1 or len(mins_eval) > 1 or np.any(mins_train > 0) or np.any(mins_eval > 0):
            trialslist, targetlist, min_dict = zero_line_numberings(trialslist, targetlist)
            trialslist_eval, targetlist_eval, min_dict = zero_line_numberings(
                trialslist_eval, targetlist_eval, min_dict
            )

    padding_at_end = False if cfg["model_to_use"] == "LSTM" else cfg["padding_at_end"]

    if cfg["use_embedded_char_pos_info"] and cfg["method_chars_into_model"] in [
        "bert",
        "dense",
    ]:
        chars_center_coords_padded, _, chars_padding_list = pad_sequence(
            chars_center_coords_list,
            None,
            sample_means,
            sample_std,
            max_len_chars_list,
            cfg["target_padding_number"],
            padding_at_end,
            input_padding_val=cfg["input_padding_val"],
            input_is_chars=True,
        )
        chars_center_coords_padded_eval, _, _ = pad_sequence(
            chars_center_coords_list_eval,
            None,
            sample_means,
            sample_std,
            max_len_chars_list,
            cfg["target_padding_number"],
            padding_at_end,
            input_padding_val=cfg["input_padding_val"],
            input_is_chars=True,
        )
    else:
        chars_center_coords_padded = chars_center_coords_padded_eval = None

    if cfg["loss_function"] == "OrdinalRegLoss":
        cfg["target_padding_number"] = -1
        logger.info(f'cfg["target_padding_number"] changed to {cfg["target_padding_number"]} for OrdinalRegLoss')
        if cfg["set_num_classes_manually"]:
            targets_max = cfg["ord_reg_loss_max"]
            targets_min = cfg["ord_reg_loss_min"]
        else:
            targets_max = max(
                np.max(targetlist).numpy().item(),
                np.max(targetlist_eval).numpy().item(),
            )
            targets_min = min(
                np.min(targetlist).numpy().item(),
                np.min(targetlist_eval).numpy().item(),
            )
        cfg["num_classes"] = targets_max - targets_min
        logger.info(f'cfg["num_classes"] changed to {cfg["num_classes"]} for OrdinalRegLoss')
        targetlist = [x - targets_min for x in targetlist]
        targetlist = [x / (targets_max - targets_min) for x in targetlist]
        targetlist_eval = [x - targets_min for x in targetlist_eval]
        targetlist_eval = [x / (targets_max - targets_min) for x in targetlist_eval]

    samples_arr_padded, target_arr, padding_list = pad_sequence(
        samplelist,
        targetlist,
        sample_means,
        sample_std,
        cfg["max_seq_length"],
        target_padding_number=cfg["target_padding_number"],
        padding_at_end=padding_at_end,
        input_padding_val=cfg["input_padding_val"],
    )
    samples_arr_padded_eval, target_arr_eval, padding_list_eval = pad_sequence(
        samplelist_eval,
        targetlist_eval,
        sample_means,
        sample_std,
        cfg["max_seq_length"],
        target_padding_number=cfg["target_padding_number"],
        padding_at_end=padding_at_end,
        input_padding_val=cfg["input_padding_val"],
    )

    if cfg["set_num_classes_manually"] and not cfg["loss_function"] == "OrdinalRegLoss":
        num_classes = cfg["manually_set_num_classes"]
    else:
        num_classes = max(t.unique(target_arr).shape[0], t.unique(target_arr_eval).shape[0])
    cfg["num_classes"] = int(num_classes)
    if cfg["loss_function"] == "CrossEntropyLoss":
        target_arr_onehot = target_arr.squeeze(dim=-1).to(t.long)
        target_arr_onehot_eval = target_arr_eval.squeeze(dim=-1).to(t.long)
    else:
        target_arr_onehot = target_arr
        target_arr_onehot_eval = target_arr_eval
    return (
        samples_arr_padded,
        samples_arr_padded_eval,
        cfg,
        trialslist,
        trialslist_eval,
        targetlist,
        targetlist_eval,
        padding_list,
        padding_list_eval,
        target_arr,
        target_arr_eval,
        sample_means,
        sample_std,
        chars_center_coords_padded,
        target_arr_onehot,
        chars_center_coords_padded_eval,
        target_arr_onehot_eval,
    )


def save_cfg_to_yaml(cfg: dict, results_folder_path: pl.Path):
    with open(
        results_folder_path.joinpath(
            f'{cfg["finished_cfgs_folder_name"]}/{cfg["model_to_use"]}_fin_exp_{cfg["time_experiment"]}.yaml'
        ),
        "w",
    ) as f:
        yaml.safe_dump(cfg, f)


def train(cfg):
    pl.Path(f"{cfg['results_folder']}/{cfg['loss_plots_folder_name']}").mkdir(exist_ok=True, parents=True)
    pl.Path(f"{cfg['results_folder']}/{cfg['finished_cfgs_folder_name']}").mkdir(exist_ok=True, parents=True)
    cfg["results"] = dict()
    results_folder_path = pl.Path(cfg["results_folder"])

    extra_save_str = f"{cfg['model_to_use']}"
    pl.Path(cfg["pickle_save_str"]).mkdir(exist_ok=True)
    cfg["pickle_save_str"] += "/"

    if cfg["use_reduced_set"]:
        extra_save_str += "_reduced_dataset"
        cfg["pickle_save_str"] += "reduced_dataset_"

    cfg["extra_save_str"] = extra_save_str
    plot_path = pl.Path(cfg["plot_folder_name"])
    plot_path.mkdir(exist_ok=True, parents=False)
    mpl.use("agg")

    time_experiment = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg["loop_name_str"] = time_experiment

    cfg["time_experiment"] = time_experiment
    cfg["extra_save_str"] = extra_save_str = time_experiment

    (
        samples_arr_padded,
        samples_arr_padded_eval,
        cfg,
        trialslist,
        trialslist_eval,
        targetlist,
        targetlist_eval,
        padding_list,
        padding_list_eval,
        target_arr,
        target_arr_eval,
        sample_means,
        sample_std,
        chars_center_coords_padded,
        target_arr_onehot,
        chars_center_coords_padded_eval,
        target_arr_onehot_eval,
    ) = prep_input(cfg)

    cfg["hidden_dim_bert"] = cfg["head_multiplication_factor"] * cfg["num_attention_heads"]
    if not hasattr(results_folder_path, "joinpath"):
        results_folder_path = pl.Path(results_folder_path)
    return_ims = False
    if cfg["use_embedded_char_pos_info"] and (
        cfg["method_chars_into_model"] == "resnet" or cfg["model_to_use"] == "resnet"
    ):
        return_ims = True

    train_set = utils.DSet(
        in_sequence=samples_arr_padded,
        chars_center_coords_padded=chars_center_coords_padded,
        out_categories=target_arr_onehot,
        trialslist=trialslist,
        padding_list=padding_list,
        padding_at_end=cfg["padding_at_end"],
        return_images_for_conv=return_ims,
        im_partial_string=cfg["im_partial_string"],
        input_im_shape=cfg["char_plot_shape"],
    )
    val_set = utils.DSet(
        samples_arr_padded_eval,
        chars_center_coords_padded_eval,
        target_arr_onehot_eval,
        trialslist_eval,
        padding_list=padding_list_eval,
        padding_at_end=cfg["padding_at_end"],
        return_images_for_conv=return_ims,
        im_partial_string=cfg["im_partial_string"],
        input_im_shape=cfg["char_plot_shape"],
    )

    cfg["num_train_samples"] = len(train_set)
    cfg["num_eval_samples"] = len(val_set)
    logger.info(f"Training samples: {len(train_set)}, eval samples: {len(val_set)}")

    train_loader = dl(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = dl(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    for batch in train_loader:
        x0 = batch[0]
        target0 = batch[-1]
        logger.info(f"batch[0] has shape {list(x0.shape)}")
        logger.info(f"batch[-1] has shape {list(target0.shape)}")
        if cfg["use_embedded_char_pos_info"] and cfg["method_chars_into_model"] in [
            "bert",
            "dense",
        ]:
            char_dims = batch[1].shape[-1]
        else:
            char_dims = 0
        break
    cfg["char_dims"] = char_dims

    checkpoint_cb = plight.callbacks.ModelCheckpoint(
        dirpath=f'{cfg["results_folder"]}/ckpts',
        filename=f'{cfg["extra_save_str"]}' + "_{epoch}-{val_loss:.5f}",
        save_top_k=cfg["save_top_k"],
        mode="max",
        verbose=False,
        monitor="epoch",
        save_weights_only=cfg["save_weights_only"],
    )
    gen_cb = GenerateCallback(
        dirpath=f'{cfg["results_folder"]}/loss_plots',
        filename=f'Light{cfg["extra_save_str"]}',
    )
    callbacks = [checkpoint_cb, gen_cb]
    accelerator = "cpu" if cfg["use_CPU"] else "gpu"
    model = models.LitModel(
        x0.shape,
        cfg["hidden_dim_bert"],
        cfg["num_attention_heads"],
        cfg["n_layers_BERT"],
        cfg["loss_function"],
        cfg["lr"],
        cfg["weight_decay"],
        cfg,
        cfg["use_lr_warmup"],
        cfg["use_reduce_on_plateau"],
        track_gradient_histogram=cfg["track_gradient_histogram"],
        register_forw_hook=cfg["track_activations_via_hook"],
        char_dims=cfg["char_dims"],
    )
    with t.inference_mode():
        test_model_output = model(batch)
    logger.info(f"test_model_output has shape {list(test_model_output.shape)}")
    num_params = sum([i.numel() for i in model.parameters()])
    cfg["num_model_parameters"] = int(num_params)
    max_epochs = -1
    max_steps = cfg["max_training_steps"]
    trainer = plight.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        default_root_dir=f'{cfg["results_folder"]}/ckpts',
        accelerator=accelerator,
        strategy="auto",
        devices="auto",
        callbacks=callbacks,
        logger=[
            plight.loggers.TensorBoardLogger(
                save_dir="tblogs",
                version="",
                name=f"Light{cfg['extra_save_str']}_{cfg['time_experiment']}",
            ),
        ],
        check_val_every_n_epoch=1,
        precision=cfg["precision"],
    )
    cfg["lr_initial"] = str(model.learning_rate)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    with t.inference_mode():
        pred_out = trainer.predict(model, dataloaders=val_loader, return_predictions=True)

    cfg["epochs_completed"] = trainer.current_epoch
    cfg["steps_completed"] = trainer.global_step
    save_cfg_to_yaml(cfg, results_folder_path)

    predicted_decoded = []
    reals = []
    for batch in pred_out:
        pred_temp = batch[0].squeeze().numpy()
        real_temp = batch[1].squeeze().numpy()
        if len(pred_temp.shape) < 2 or len(real_temp.shape) < 2:
            predicted_decoded.append(pred_temp)
            reals.append(real_temp)
        else:
            pred_temp = list(pred_temp)
            real_temp = list(real_temp)
            predicted_decoded.extend(pred_temp)
            reals.extend(real_temp)
    datasets_string = (
        "".join([cfg["dataset_folders"][idx] for idx in cfg["dataset_folder_idx_evaluation"]])
        .replace("/", "_")
        .replace("\\", "_")
    )
    datasets_string = datasets_string[:80]
    if len(reals[0]) > 1:
        predicted_decoded_trimmed = []
        predicted_decoded_trimmed_flattened = []
        reals_trimmed = []
        reals_trimmed_flattened = []
        hits = []
        accuracies = []
        acc_dicts = []
        if cfg["loss_function"] == "OrdinalRegLoss":
            predicted_decoded_stack = np.stack(predicted_decoded, axis=0)
            reals_stack = np.stack(reals, axis=0)
            min_real_stack = np.min(reals_stack)
            predicted_decoded_stack_trimmed = predicted_decoded_stack[reals_stack != min_real_stack]
            reals_stack_trimmed = reals_stack[reals_stack != np.min(reals_stack)]
            trimmed_acc = sum(predicted_decoded_stack_trimmed == reals_stack_trimmed) / len(reals_stack_trimmed)
        for tridx, (arr_pred, arr_real) in enumerate(zip(predicted_decoded, reals)):
            trialid = trialslist_eval[tridx]["trial_id"]
            indeces_to_use = np.arange(arr_pred.shape[0])
            if cfg["loss_function"] == "OrdinalRegLoss":
                indeces_to_use = indeces_to_use[arr_real != min_real_stack]
            else:
                indeces_to_use = indeces_to_use[arr_real != cfg["target_padding_number"]]
            trimmed_pred = arr_pred[indeces_to_use]
            trimmed_real = arr_real[indeces_to_use]
            real_max = trialslist_eval[tridx]["num_char_lines"] - 1
            real_min = 0
            trimmed_pred = np.clip(trimmed_pred, real_min, real_max)
            predicted_decoded_trimmed.append(trimmed_pred)
            trial = trialslist_eval[tridx]
            chars_df = pd.DataFrame(trial["chars_list"])
            last_char_xmax = chars_df.char_xmax.iloc[-1]
            for predidx, pred_val in enumerate(trimmed_pred):
                if pred_val > 0 and pred_val == real_max:
                    sample_arr = samples_arr_padded_eval[tridx][indeces_to_use][predidx]
                    if sample_arr[0] > last_char_xmax:
                        trimmed_pred[predidx] = pred_val - 1
            reals_trimmed.append(trimmed_real)
            hits.extend(trimmed_pred == trimmed_real)
            acc_trial = sum(trimmed_pred == trimmed_real) / len(trimmed_real)
            accuracies.append(acc_trial)
            reals_trimmed_flattened.extend(list(trimmed_real))
            predicted_decoded_trimmed_flattened.extend(list(trimmed_pred))
            acc_dicts.append(
                dict(
                    idx=tridx,
                    trialid=trialid,
                    acc=round(acc_trial * 100, ndigits=2),
                    max_predicted=trimmed_pred.max(),
                    max_real=trialslist_eval[tridx]["dffix"]["assigned_line"].max(),
                )
            )
        acc_df = pd.DataFrame(acc_dicts)
        model_str = cfg["model_to_use"]
        pl.Path(f"{results_folder_path}/inference").mkdir(exist_ok=True)
        acc_df.to_csv(
            f'{results_folder_path}/inference/acc_df_{datasets_string}_{model_str}_{cfg["time_experiment"]}.csv'
        )
        above95 = acc_df.query("acc >=95").shape[0] * 100 / acc_df.shape[0]
        cfg["results"]["trials_above_95perc_accurate"] = round(float(above95), ndigits=4)
        logger.info(f"trials_above_95perc_accurate is {above95:.2f} %")
    else:
        predicted_decoded = predicted_decoded_trimmed_flattened = [x.item() for x in predicted_decoded]
        reals = reals_trimmed_flattened = [x.item() for x in reals]
        hits = [x1 == x2 for x1, x2 in zip(predicted_decoded, reals)]
    clamped_accuracy_from_lists = np.mean(hits)
    clamped_accuracy_from_lists_from_avg_accs = np.mean(accuracies)

    predicted_decoded = t.cat([b[0] for b in pred_out], dim=0).view(-1)
    reals = t.cat([b[1] for b in pred_out], dim=0).view(-1)
    if cfg["loss_function"] == "OrdinalRegLoss":
        predicted_decoded = predicted_decoded[reals != min_real_stack]
        reals = reals[reals != min_real_stack]
        clamped_accuracy = (t.clamp(predicted_decoded, reals.min(), reals.max()) == reals).sum() / reals.shape[0]
    else:
        if len(reals[0].shape) < 2:
            clamped_accuracy = clamped_accuracy_from_lists
        else:
            clamped_accuracy = torchmetrics.functional.accuracy(
                predicted_decoded,
                reals.to(t.long),
                average="micro",
                ignore_index=cfg["target_padding_number"],
            )

    cfg["results"]["clamped_accuracy"] = round(float(clamped_accuracy), ndigits=4)
    logger.info(f"clamped_accuracy is {clamped_accuracy*100:.2f} %")

    cfg["results"]["clamped_accuracy_from_lists"] = round(float(clamped_accuracy_from_lists), ndigits=4)
    logger.info(f"clamped_accuracy_from_lists is {clamped_accuracy_from_lists*100:.2f} %")

    cfg["results"]["clamped_accuracy_from_lists_from_avg_accs"] = round(
        float(clamped_accuracy_from_lists_from_avg_accs), ndigits=4
    )
    logger.info(f"clamped_accuracy_from_lists_from_avg_accs is {clamped_accuracy_from_lists_from_avg_accs*100:.2f} %")
    return cfg["results"]


def main(yaml_file):
    import yaml

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)

    if args.gpu_id is not None:
        if args.gpu_id > -1:
            cfg["use_all_GPUs"] = False
            cfg["GPU_to_use"] = [args.gpu_id]
        else:
            cfg["use_all_GPUs"] = True
    train(cfg)


if __name__ == "__main__":
    yaml_file = f"experiments/{args.yaml}.yaml"
    main(yaml_file)

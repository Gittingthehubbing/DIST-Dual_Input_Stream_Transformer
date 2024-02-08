import zipfile
import os
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data.dataloader import DataLoader as dl
import yaml
from io import StringIO
import torch as t
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as torch_dset
from PIL import Image
import torchvision.transforms.functional as tvfunc
import json
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import pathlib as pl
import matplotlib as mpl
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import einops as eo
import copy

# import stqdm
from tqdm.auto import tqdm
import time
import requests

from matplotlib.patches import Rectangle
from matplotlib import font_manager
from models import LitModel, EnsembleModel
from loss_functions import corn_label_from_logits
import classic_correction_algos as calgo
import analysis_funcs as anf

TEMP_FOLDER = pl.Path("results")
AVAILABLE_FONTS = [x.name for x in font_manager.fontManager.ttflist]
PLOTS_FOLDER = pl.Path("plots")
TEMP_FIGURE_STIMULUS_PATH = PLOTS_FOLDER / "temp_matplotlib_plot_stimulus.png"
all_fonts = [x.name for x in font_manager.fontManager.ttflist]
mpl.use("agg")

DIST_MODELS_FOLDER = pl.Path("models")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
gradio_plots = pl.Path("plots")

event_strs = [
    "EFIX",
    "EFIX R",
    "EFIX L",
    "SSACC",
    "ESACC",
    "SFIX",
    "MSG",
    "SBLINK",
    "EBLINK",
    "BUTTON",
    "INPUT",
    "END",
    "START",
    "DISPLAY ON",
]
names_dict = {
    "SSACC": {"Descr": "Start of Saccade", "Pattern": "SSACC <eye > <stime>"},
    "ESACC": {
        "Descr": "End of Saccade",
        "Pattern": "ESACC <eye > <stime> <etime > <dur> <sxp > <syp> <exp > <eyp> <ampl > <pv >",
    },
    "SFIX": {"Descr": "Start of Fixation", "Pattern": "SFIX <eye > <stime>"},
    "EFIX": {"Descr": "End of Fixation", "Pattern": "EFIX <eye > <stime> <etime > <dur> <axp > <ayp> <aps >"},
    "SBLINK": {"Descr": "Start of Blink", "Pattern": "SBLINK <eye > <stime>"},
    "EBLINK": {"Descr": "End of Blink", "Pattern": "EBLINK <eye > <stime> <etime > <dur>"},
    "DISPLAY ON": {"Descr": "Actual start of Trial", "Pattern": "DISPLAY ON"},
}
metadata_strs = ["DISPLAY COORDS", "GAZE_COORDS", "FRAMERATE"]

ALGO_CHOICES = st.session_state["ALGO_CHOICES"] = [
    "warp",
    "regress",
    "compare",
    "attach",
    "segment",
    "split",
    "stretch",
    "chain",
    "slice",
    "cluster",
    "merge",
    "Wisdom_of_Crowds",
    "DIST",
    "DIST-Ensemble",
    "Wisdom_of_Crowds_with_DIST",
    "Wisdom_of_Crowds_with_DIST_Ensemble",
]
COLORS = px.colors.qualitative.Alphabet


class NumpyEncoder(json.JSONEncoder):
    "From https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable"

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pl.Path) or isinstance(obj, UploadedFile):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class DSet(torch_dset):
    def __init__(
        self,
        in_sequence: t.Tensor,
        chars_center_coords_padded: t.Tensor,
        out_categories: t.Tensor,
        trialslist: list,
        padding_list: list = None,
        padding_at_end: bool = False,
        return_images_for_conv: bool = False,
        im_partial_string: str = "fixations_chars_channel_sep",
        input_im_shape=[224, 224],
    ) -> None:
        super().__init__()

        self.in_sequence = in_sequence
        self.chars_center_coords_padded = chars_center_coords_padded
        self.out_categories = out_categories
        self.padding_list = padding_list
        self.padding_at_end = padding_at_end
        self.trialslist = trialslist
        self.return_images_for_conv = return_images_for_conv
        self.input_im_shape = input_im_shape
        if return_images_for_conv:
            self.im_partial_string = im_partial_string
            self.plot_files = [
                str(x["plot_file"]).replace("fixations_words", im_partial_string) for x in self.trialslist
            ]

    def __getitem__(self, index):

        if self.return_images_for_conv:
            im = Image.open(self.plot_files[index])
            if [im.size[1], im.size[0]] != self.input_im_shape:
                im = tvfunc.resize(im, self.input_im_shape)
            im = tvfunc.normalize(tvfunc.to_tensor(im), IMAGENET_MEAN, IMAGENET_STD)
        if self.chars_center_coords_padded is not None:
            if self.padding_list is not None:
                attention_mask = t.ones(self.in_sequence[index].shape[:-1], dtype=t.long)
                if self.padding_at_end:
                    if self.padding_list[index] > 0:
                        attention_mask[-self.padding_list[index] :] = 0
                else:
                    attention_mask[: self.padding_list[index]] = 0
                if self.return_images_for_conv:
                    return (
                        self.in_sequence[index],
                        self.chars_center_coords_padded[index],
                        im,
                        attention_mask,
                        self.out_categories[index],
                    )
                return (
                    self.in_sequence[index],
                    self.chars_center_coords_padded[index],
                    attention_mask,
                    self.out_categories[index],
                )
            else:
                if self.return_images_for_conv:
                    return (
                        self.in_sequence[index],
                        self.chars_center_coords_padded[index],
                        im,
                        self.out_categories[index],
                    )
                else:
                    return (self.in_sequence[index], self.chars_center_coords_padded[index], self.out_categories[index])

        if self.padding_list is not None:
            attention_mask = t.ones(self.in_sequence[index].shape[:-1], dtype=t.long)
            if self.padding_at_end:
                if self.padding_list[index] > 0:
                    attention_mask[-self.padding_list[index] :] = 0
            else:
                attention_mask[: self.padding_list[index]] = 0
            if self.return_images_for_conv:
                return (self.in_sequence[index], im, attention_mask, self.out_categories[index])
            else:
                return (self.in_sequence[index], attention_mask, self.out_categories[index])
        if self.return_images_for_conv:
            return (self.in_sequence[index], im, self.out_categories[index])
        else:
            return (self.in_sequence[index], self.out_categories[index])

    def __len__(self):
        if isinstance(self.in_sequence, t.Tensor):
            return self.in_sequence.shape[0]
        else:
            return len(self.in_sequence)


def download_url(url, target_filename):
    r = requests.get(url)
    open(target_filename, "wb").write(r.content)
    return 0


def asc_to_trial_ids(asc_file, close_gap_between_words=True):
    if "logger" in st.session_state:
        st.session_state["logger"].debug("asc_to_trial_ids entered")
    asc_encoding = ["ISO-8859-15", "UTF-8"][0]
    trials_dict, lines = file_to_trials_and_lines(
        asc_file, asc_encoding, close_gap_between_words=close_gap_between_words
    )

    trials_by_ids = {trials_dict[idx]["trial_id"]: trials_dict[idx] for idx in trials_dict["paragraph_trials"]}
    if hasattr(asc_file, "name"):
        if "logger" in st.session_state:
            st.session_state["logger"].info(f"Found {len(trials_by_ids)} trials in {asc_file.name}.")
    return trials_by_ids, lines


def get_trials_list(asc_file=None, close_gap_between_words=True):
    if "logger" in st.session_state:
        st.session_state["logger"].debug("get_trials_list entered")

    if asc_file == None:
        if "single_asc_file" in st.session_state.keys() and st.session_state["single_asc_file"] is not None:
            asc_file = st.session_state["single_asc_file"]
        else:
            if "logger" in st.session_state:
                st.session_state["logger"].warning("Asc file is None")
            return None

    if hasattr(asc_file, "name"):
        if "logger" in st.session_state:
            st.session_state["logger"].info(f"get_trials_list entered with asc_file {asc_file.name}")

    trials_by_ids, lines = asc_to_trial_ids(asc_file, close_gap_between_words=close_gap_between_words)
    trial_keys = list(trials_by_ids.keys())

    return trial_keys, trials_by_ids, lines, asc_file


def save_trial_to_json(trial, savename):
    if "dffix" in trial:
        trial.pop("dffix")
    with open(savename, "w", encoding="utf-8") as f:
        json.dump(trial, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def export_csv(dffix, trial):
    if isinstance(dffix, dict):
        dffix = dffix["value"]
    trial_id = trial["trial_id"]
    savename = TEMP_FOLDER.joinpath(pl.Path(trial["fname"]).stem)
    trial_name = f"{savename}_{trial_id}_trial_info.json"
    csv_name = f"{savename}_{trial_id}.csv"
    dffix.to_csv(csv_name)
    if "logger" in st.session_state:
        st.session_state["logger"].info(f"Saved processed data as {csv_name}")
    save_trial_to_json(trial, trial_name)
    if "logger" in st.session_state:
        st.session_state["logger"].info(f"Saved processed trial data as {trial_name}")

    return csv_name, trial_name


def get_all_classic_preds(dffix, trial, classic_algos_cfg):
    corrections = []
    for algo, classic_params in copy.deepcopy(classic_algos_cfg).items():
        dffix = calgo.apply_classic_algo(dffix, trial, algo, classic_params)
        corrections.append(np.asarray(dffix.loc[:, f"y_{algo}"]))
    return dffix, corrections


def apply_woc(dffix, trial, corrections, algo_choice):

    corrected_Y = calgo.wisdom_of_the_crowd(corrections)
    dffix.loc[:, f"y_{algo_choice}"] = corrected_Y
    dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(1)
    corrected_line_nums = [trial["y_char_unique"].index(y) for y in corrected_Y]
    dffix.loc[:, f"line_num_y_{algo_choice}"] = corrected_line_nums
    return dffix


def calc_xdiff_ydiff(line_xcoords_no_pad, line_ycoords_no_pad, line_heights, allow_multiple_values=False):
    x_diffs = np.unique(np.diff(line_xcoords_no_pad))
    if len(x_diffs) == 1:
        x_diff = x_diffs[0]
    elif not allow_multiple_values:
        x_diff = np.min(x_diffs)
    else:
        x_diff = x_diffs

    if np.unique(line_ycoords_no_pad).shape[0] == 1:
        return x_diff, line_heights[0]
    y_diffs = np.unique(np.diff(line_ycoords_no_pad))
    if len(y_diffs) == 1:
        y_diff = y_diffs[0]
    elif len(y_diffs) == 0:
        y_diff = 0
    elif not allow_multiple_values:
        y_diff = np.min(y_diffs)
    else:
        y_diff = y_diffs
    return x_diff, y_diff


def add_words(trial, close_gap_between_words=True):
    chars_list_reconstructed = []
    words_list = []
    word_start_idx = 0
    chars_df = pd.DataFrame(trial["chars_list"])
    chars_df["char_width"] = chars_df.char_xmax - chars_df.char_xmin
    space_width = chars_df.loc[chars_df["char"] == " ", "char_width"].mean()

    for idx, char_dict in enumerate(trial["chars_list"]):
        on_line_num = char_dict["assigned_line"]
        chars_list_reconstructed.append(char_dict)
        if (
            char_dict["char"] in [" ", ",", ";", ".", ":"]
            or (
                len(chars_list_reconstructed) > 2
                and (chars_list_reconstructed[-1]["char_xmin"] < chars_list_reconstructed[-2]["char_xmin"])
            )
            or len(chars_list_reconstructed) == len(trial["chars_list"])
        ):
            triggered = True
            word_xmin = chars_list_reconstructed[word_start_idx]["char_xmin"]
            word_xmax = chars_list_reconstructed[-2]["char_xmax"]
            word_ymin = chars_list_reconstructed[word_start_idx]["char_ymin"]
            word_ymax = chars_list_reconstructed[word_start_idx]["char_ymax"]
            word_x_center = (word_xmax - word_xmin) / 2 + word_xmin
            word_y_center = (word_ymax - word_ymin) / 2 + word_ymin
            word = "".join(
                [
                    chars_list_reconstructed[idx]["char"]
                    for idx in range(word_start_idx, len(chars_list_reconstructed) - 1)
                ]
            )
            assigned_line = chars_list_reconstructed[word_start_idx]["assigned_line"]

            word_dict = dict(
                word=word,
                word_xmin=word_xmin,
                word_xmax=word_xmax,
                word_ymin=word_ymin,
                word_ymax=word_ymax,
                word_x_center=word_x_center,
                word_y_center=word_y_center,
                assigned_line=assigned_line,
            )
            if char_dict["char"] != " ":
                word_start_idx = idx
            else:
                word_start_idx = idx + 1
            words_list.append(word_dict)
        else:
            triggered = False
    last_letter_in_word = word_dict["word"][-1]
    last_letter_in_chars_list_reconstructed = char_dict["char"]
    if last_letter_in_word != last_letter_in_chars_list_reconstructed:
        word_dict = dict(
            word=char_dict["char"],
            word_xmin=char_dict["char_xmin"],
            word_xmax=char_dict["char_xmax"],
            word_ymin=char_dict["char_ymin"],
            word_ymax=char_dict["char_ymax"],
            word_x_center=char_dict["char_x_center"],
            word_y_center=char_dict["char_y_center"],
            assigned_line=assigned_line,
        )
        words_list.append(word_dict)

    if close_gap_between_words:
        for widx in range(1, len(words_list)):
            if words_list[widx]["assigned_line"] == words_list[widx - 1]["assigned_line"]:
                word_sep_half_width = (words_list[widx]["word_xmin"] - words_list[widx - 1]["word_xmax"]) / 2
                words_list[widx - 1]["word_xmax"] = words_list[widx - 1]["word_xmax"] + word_sep_half_width
                words_list[widx]["word_xmin"] = words_list[widx]["word_xmin"] - word_sep_half_width

    return words_list


def asc_lines_to_trials_by_trail_id(
    lines: list, paragraph_trials_only=False, fname: str = "", close_gap_between_words=True
) -> dict:
    if hasattr(fname, "name"):
        fname = fname.name
    fps = -999
    display_coords = -999
    trials_dict = dict(paragraph_trials=[], paragraph_trial_IDs=[])
    trial_idx = -1
    removed_trial_ids = []
    for idx, l in enumerate(lines):
        parts = l.strip().split(" ")
        if "TRIALID" in l:
            trial_id = parts[-1]
            trial_idx += 1
            if trial_id[0] == "F":
                trial_is = "question"
            elif trial_id[0] == "P":
                trial_is = "practice"
            else:
                trial_is = "paragraph"
                trials_dict["paragraph_trials"].append(trial_idx)
                trials_dict["paragraph_trial_IDs"].append(trial_id)
            trials_dict[trial_idx] = dict(trial_id=trial_id, trial_id_idx=idx, trial_is=trial_is, filename=fname)
            last_trial_skipped = False

        elif "TRIAL_RESULT" in l or "stop_trial" in l:
            trials_dict[trial_idx]["trial_result_idx"] = idx
            trials_dict[trial_idx]["trial_result_timestamp"] = int(parts[0].split("\t")[1])
            if len(parts) > 2:
                trials_dict[trial_idx]["trial_result_number"] = int(parts[2])
        elif "DISPLAY COORDS" in l and isinstance(display_coords, int):
            display_coords = (float(parts[-4]), float(parts[-3]), float(parts[-2]), float(parts[-1]))
        elif "GAZE_COORDS" in l and isinstance(display_coords, int):
            display_coords = (float(parts[-4]), float(parts[-3]), float(parts[-2]), float(parts[-1]))
        elif "FRAMERATE" in l:
            l_idx = parts.index(metadata_strs[2])
            fps = float(parts[l_idx + 1])
        elif "TRIAL ABORTED" in l or "TRIAL REPEATED" in l:
            if not last_trial_skipped:
                if trial_is == "paragraph":
                    trials_dict["paragraph_trials"].remove(trial_idx)
                trial_idx -= 1
                removed_trial_ids.append(trial_id)
                last_trial_skipped = True

    if paragraph_trials_only:
        trials_dict_temp = trials_dict.copy()
        for k in trials_dict_temp.keys():
            if k not in ["paragraph_trials"] + trials_dict_temp["paragraph_trials"]:
                trials_dict.pop(k)
        if len(trials_dict_temp["paragraph_trials"]):
            trial_idx = trials_dict_temp["paragraph_trials"][-1]
        else:
            return trials_dict
    trials_dict["display_coords"] = display_coords
    trials_dict["fps"] = fps
    trials_dict["max_trial_idx"] = trial_idx
    enum = trials_dict["paragraph_trials"] if "paragraph_trials" in trials_dict.keys() else range(len(trials_dict))
    for trial_idx in enum:
        if trial_idx not in trials_dict.keys():
            continue
        chars_list = []
        if "display_coords" not in trials_dict[trial_idx].keys():
            trials_dict[trial_idx]["display_coords"] = trials_dict["display_coords"]
        trial_start_idx = trials_dict[trial_idx]["trial_id_idx"]
        trial_end_idx = trials_dict[trial_idx]["trial_result_idx"]
        trial_lines = lines[trial_start_idx:trial_end_idx]
        for idx, l in enumerate(trial_lines):
            parts = l.strip().split(" ")
            if "START" in l and " MSG" not in l:
                trials_dict[trial_idx]["start_idx"] = trial_start_idx + idx + 7
                trials_dict[trial_idx]["start_time"] = int(parts[0].split("\t")[1])
            elif "END" in l and "ENDBUTTON" not in l and " MSG" not in l:
                trials_dict[trial_idx]["end_idx"] = trial_start_idx + idx - 2
                trials_dict[trial_idx]["end_time"] = int(parts[0].split("\t")[1])
            elif "SYNCTIME" in l:
                trials_dict[trial_idx]["synctime"] = trial_start_idx + idx
                trials_dict[trial_idx]["synctime_time"] = int(parts[0].split("\t")[1])
            elif "GAZE TARGET OFF" in l:
                trials_dict[trial_idx]["gaze_targ_off_time"] = int(parts[0].split("\t")[1])
            elif "GAZE TARGET ON" in l:
                trials_dict[trial_idx]["gaze_targ_on_time"] = int(parts[0].split("\t")[1])
            elif "DISPLAY_SENTENCE" in l:  # some .asc files seem to use this
                trials_dict[trial_idx]["gaze_targ_on_time"] = int(parts[0].split("\t")[1])
            elif "REGION CHAR" in l:
                rg_idx = parts.index("CHAR")
                if len(parts[rg_idx:]) > 8:
                    char = " "
                    idx_correction = 1
                elif len(parts[rg_idx:]) == 3:
                    char = " "
                    if "REGION CHAR" not in trial_lines[idx + 1]:
                        parts = trial_lines[idx + 1].strip().split(" ")
                        idx_correction = -rg_idx - 4
                else:
                    char = parts[rg_idx + 3]
                    idx_correction = 0
                try:
                    char_dict = {
                        "char": char,
                        "char_xmin": float(parts[rg_idx + 4 + idx_correction]),
                        "char_ymin": float(parts[rg_idx + 5 + idx_correction]),
                        "char_xmax": float(parts[rg_idx + 6 + idx_correction]),
                        "char_ymax": float(parts[rg_idx + 7 + idx_correction]),
                    }
                    char_dict["char_y_center"] = (char_dict["char_ymax"] - char_dict["char_ymin"]) / 2 + char_dict[
                        "char_ymin"
                    ]
                    char_dict["char_x_center"] = (char_dict["char_xmax"] - char_dict["char_xmin"]) / 2 + char_dict[
                        "char_xmin"
                    ]
                    chars_list.append(char_dict)
                except Exception as e:
                    if "logger" in st.session_state:
                        st.session_state["logger"].warning(f"char_dict creation failed for parts {parts}")
                    if "logger" in st.session_state:
                        st.session_state["logger"].warning(e)

        if "gaze_targ_on_time" in trials_dict[trial_idx]:
            trials_dict[trial_idx]["trial_start_time"] = trials_dict[trial_idx]["gaze_targ_on_time"]
        else:
            trials_dict[trial_idx]["trial_start_time"] = trials_dict[trial_idx]["start_time"]

        if len(chars_list) > 0:
            line_ycoords = []
            for idx in range(len(chars_list)):
                chars_list[idx]["char_line_y"] = (
                    chars_list[idx]["char_ymax"] - chars_list[idx]["char_ymin"]
                ) / 2 + chars_list[idx]["char_ymin"]
                if chars_list[idx]["char_line_y"] not in line_ycoords:
                    line_ycoords.append(chars_list[idx]["char_line_y"])
            for idx in range(len(chars_list)):
                chars_list[idx]["assigned_line"] = line_ycoords.index(chars_list[idx]["char_line_y"])

            line_heights = [x["char_ymax"] - x["char_ymin"] for x in chars_list]
            line_xcoords_all = [x["char_x_center"] for x in chars_list]
            line_xcoords_no_pad = np.unique(line_xcoords_all)

            line_ycoords_all = [x["char_y_center"] for x in chars_list]
            line_ycoords_no_pad = np.unique(line_ycoords_all)

            trials_dict[trial_idx]["x_char_unique"] = list(line_xcoords_no_pad)
            trials_dict[trial_idx]["y_char_unique"] = list(line_ycoords_no_pad)
            x_diff, y_diff = calc_xdiff_ydiff(
                line_xcoords_no_pad, line_ycoords_no_pad, line_heights, allow_multiple_values=False
            )
            trials_dict[trial_idx]["x_diff"] = float(x_diff)
            trials_dict[trial_idx]["y_diff"] = float(y_diff)
            trials_dict[trial_idx]["num_char_lines"] = len(line_ycoords_no_pad)
            trials_dict[trial_idx]["line_heights"] = line_heights
            trials_dict[trial_idx]["chars_list"] = chars_list

            words_list = add_words(trials_dict[trial_idx], close_gap_between_words=close_gap_between_words)
            trials_dict[trial_idx]["words_list"] = words_list

    return trials_dict


def file_to_trials_and_lines(uploaded_file, asc_encoding: str = "ISO-8859-15", close_gap_between_words=True):
    if isinstance(uploaded_file, str) or isinstance(uploaded_file, pl.Path):
        with open(uploaded_file, "r", encoding=asc_encoding) as f:
            lines = f.readlines()
    else:
        stringio = StringIO(uploaded_file.getvalue().decode(asc_encoding))
        loaded_str = stringio.read()
        lines = loaded_str.split("\n")
    trials_dict = asc_lines_to_trials_by_trail_id(
        lines, True, uploaded_file, close_gap_between_words=close_gap_between_words
    )

    if "paragraph_trials" not in trials_dict.keys() and "trial_is" in trials_dict[0].keys():
        paragraph_trials = []
        for k in range(trials_dict["max_trial_idx"]):
            if trials_dict[k]["trial_is"] == "paragraph":
                paragraph_trials.append(k)
        trials_dict["paragraph_trials"] = paragraph_trials

    enum = (
        trials_dict["paragraph_trials"]
        if "paragraph_trials" in trials_dict.keys()
        else range(trials_dict["max_trial_idx"])
    )
    for k in enum:
        if "chars_list" in trials_dict[k].keys():
            max_line = trials_dict[k]["chars_list"][-1]["assigned_line"]
            words_on_lines = {x: [] for x in range(max_line + 1)}
            [words_on_lines[x["assigned_line"]].append(x["char"]) for x in trials_dict[k]["chars_list"]]
            sentence_list = ["".join([s for s in v]) for idx, v in words_on_lines.items()]
            text = sentence_list[0] + "\n".join([x for x in sentence_list[1:]])
            trials_dict[k]["sentence_list"] = sentence_list
            trials_dict[k]["text"] = text
            trials_dict[k]["max_line"] = max_line

    return trials_dict, lines


def get_plot_props(trial, available_fonts):
    if "font" in trial.keys():
        font = trial["font"]
        font_size = trial["font_size"]
        if font not in available_fonts:
            font = "DejaVu Sans Mono"
    else:
        font = "DejaVu Sans Mono"
        font_size = 21
    dpi = 100
    if "display_coords" in trial.keys():
        screen_res = (trial["display_coords"][2], trial["display_coords"][3])
    else:
        screen_res = (1920, 1080)
    return font, font_size, dpi, screen_res


def trial_to_dfs(
    trial: dict, lines: list, use_synctime: bool = False, save_lines_to_txt=False, cut_out_outer_fixations=False
):
    """trial should be dict of line numbers of trials.
    lines should be list of lines from .asc file."""

    if use_synctime and "synctime" in trial:
        idx0, idxend = trial["synctime"] + 1, trial["trial_result_idx"]
    else:
        idx0, idxend = trial["start_idx"], trial["end_idx"]

    line_dicts = []
    fixations_dicts = []
    blink_started = False

    fixation_started = False
    efix_count = 0
    sfix_count = 0
    sblink_count = 0

    if save_lines_to_txt:
        with open("Lines_plus500.txt", "w") as f:
            f.writelines(lines[idx0 - 500 : idxend + 500])
    eye_to_use = "R"
    for l in lines[idx0 : idxend + 1]:
        if "EFIX R" in l:
            eye_to_use = "R"
            break
        elif "EFIX L" in l:
            eye_to_use = "L"
            break
    for l in lines[idx0 : idxend + 1]:
        parts = [x.strip() for x in l.split("\t")]
        if f"EFIX {eye_to_use}" in l:
            efix_count += 1
            if fixation_started:
                if parts[1] == "." and parts[2] == ".":
                    continue
                fixations_dicts.append(
                    {
                        "start_time": float(parts[0].split()[-1].strip()),
                        "end_time": float(parts[1].strip()),
                        "duration": float(parts[2].strip()),
                        "x": float(parts[3].strip()),
                        "y": float(parts[4].strip()),
                        "pupil_size": float(parts[5].strip()),
                    }
                )
                if len(fixations_dicts) >= 2:
                    assert (
                        fixations_dicts[-1]["start_time"] > fixations_dicts[-2]["start_time"]
                    ), "start times not in order"
                fixation_started = False

        elif f"SFIX {eye_to_use}" in l:
            sfix_count += 1
            fixation_started = True
        elif f"SBLINK {eye_to_use}" in l:
            sblink_count += 1
            blink_started = True
        if not blink_started and not any([True for x in event_strs if x in l]):
            if len(parts) < 3 or (parts[1] == "." and parts[2] == "."):
                continue
            line_dicts.append(
                {
                    "idx": float(parts[0].strip()),
                    "x": float(parts[1].strip()),
                    "y": float(parts[2].strip()),
                    "p": float(parts[3].strip()),
                }
            )

        elif f"EBLINK {eye_to_use}" in l:
            blink_started = False

    df = pd.DataFrame(line_dicts)
    dffix = pd.DataFrame(fixations_dicts)
    if len(fixations_dicts) > 0:
        dffix["corrected_start_time"] = dffix.start_time - trial["trial_start_time"]
        dffix["corrected_end_time"] = dffix.end_time - trial["trial_start_time"]
        dffix["fix_duration"] = dffix.corrected_end_time.values - dffix.corrected_start_time.values
        assert all(np.diff(dffix["corrected_start_time"]) > 0), "start times not in order"
    else:
        df, pd.DataFrame(), trial

    if cut_out_outer_fixations:
        dffix = dffix[(dffix.x > -10) & (dffix.y > -10) & (dffix.x < 1050) & (dffix.y < 800)]
    trial["efix_count"] = efix_count
    trial["eye_to_use"] = eye_to_use
    trial["sfix_count"] = sfix_count
    trial["sblink_count"] = sblink_count
    return df, dffix, trial


def get_save_path(fpath, fname_ending):
    save_path = gradio_plots.joinpath(f"{fpath.stem}_{fname_ending}.png")
    return save_path


def save_im_load_convert(fpath, fig, fname_ending, mode):
    save_path = get_save_path(fpath, fname_ending)
    fig.savefig(save_path)
    im = Image.open(save_path).convert(mode)
    im.save(save_path)
    return im


def get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, dffix=None, prefix="word"):
    fig = plt.figure(figsize=(screen_res[0] / dpi, screen_res[1] / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    if dffix is not None:
        ax.set_ylim((dffix.y.min(), dffix.y.max()))
        ax.set_xlim((dffix.x.min(), dffix.x.max()))
    else:
        ax.set_ylim((words_df[f"{prefix}_y_center"].min() - y_margin, words_df[f"{prefix}_y_center"].max() + y_margin))
        ax.set_xlim((words_df[f"{prefix}_x_center"].min() - x_margin, words_df[f"{prefix}_x_center"].max() + x_margin))
    ax.invert_yaxis()
    fig.add_axes(ax)
    return fig, ax


def plot_text_boxes_fixations(
    fpath,
    dpi,
    screen_res,
    data_dir_sub,
    set_font_size: bool,
    font_size: int,
    use_words: bool,
    save_channel_repeats: bool,
    save_combo_grey_and_rgb: bool,
    dffix=None,
    trial=None,
):
    if isinstance(fpath, str):
        fpath = pl.Path(fpath)
    if use_words:
        prefix = "word"
    else:
        prefix = "char"
    if dffix is None:
        dffix = pd.read_csv(fpath)
    if trial is None:
        json_fpath = str(fpath).replace("_fixations.csv", "_trial.json")
        with open(json_fpath, "r") as f:
            trial = json.load(f)
    words_df = pd.DataFrame(trial[f"{prefix}s_list"])
    x_right = words_df[f"{prefix}_xmin"]
    x_left = words_df[f"{prefix}_xmax"]
    y_top = words_df[f"{prefix}_ymax"]
    y_bottom = words_df[f"{prefix}_ymin"]

    if f"{prefix}_x_center" not in words_df.columns:
        words_df[f"{prefix}_x_center"] = (words_df[f"{prefix}_xmax"] - words_df[f"{prefix}_xmin"]) / 2 + words_df[
            f"{prefix}_xmin"
        ]
        words_df[f"{prefix}_y_center"] = (words_df[f"{prefix}_ymax"] - words_df[f"{prefix}_ymin"]) / 2 + words_df[
            f"{prefix}_ymin"
        ]

    x_margin = words_df[f"{prefix}_x_center"].mean() / 8
    y_margin = words_df[f"{prefix}_y_center"].mean() / 4
    times = dffix.corrected_start_time - dffix.corrected_start_time.min()
    times = times / times.max()
    times = np.linspace(0.25, 1, len(times))

    if set_font_size:
        font = "monospace"
    else:
        font_size = trial["font_size"] * 27 // dpi

    font_props = FontProperties(family=font, style="normal", size=font_size)
    if save_combo_grey_and_rgb:
        fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)
        ax.scatter(dffix.x, dffix.y, alpha=times, facecolor="b")
        for idx in range(len(x_left)):
            xdiff = x_right[idx] - x_left[idx]
            ydiff = y_top[idx] - y_bottom[idx]
            rect = patches.Rectangle(
                (x_left[idx] - 1, y_bottom[idx] - 1),
                xdiff,
                ydiff,
                alpha=0.9,
                linewidth=0.8,
                edgecolor="r",
                facecolor="none",
            )  # seems to need one pixel offset
            ax.text(
                words_df[f"{prefix}_x_center"][idx],
                words_df[f"{prefix}_y_center"][idx],
                words_df[prefix][idx],
                horizontalalignment="center",
                verticalalignment="center",
                fontproperties=font_props,
                color="g",
            )
            ax.add_patch(rect)
        fname_ending = f"{prefix}s_combo_rgb"
        words_combo_rgb_im = save_im_load_convert(fpath, fig, fname_ending, "RGB")
        plt.close("all")

        fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)

        ax.scatter(dffix.x, dffix.y, facecolor="k", alpha=times)
        for idx in range(len(x_left)):
            xdiff = x_right[idx] - x_left[idx]
            ydiff = y_top[idx] - y_bottom[idx]
            rect = patches.Rectangle(
                (x_left[idx] - 1, y_bottom[idx] - 1),
                xdiff,
                ydiff,
                alpha=0.9,
                linewidth=0.8,
                edgecolor="k",
                facecolor="none",
            )  # seems to need one pixel offset
            ax.text(
                words_df[f"{prefix}_x_center"][idx],
                words_df[f"{prefix}_y_center"][idx],
                words_df[prefix][idx],
                horizontalalignment="center",
                verticalalignment="center",
                fontproperties=font_props,
            )
            ax.add_patch(rect)
        fname_ending = f"{prefix}s_combo_grey"
        words_combo_grey_im = save_im_load_convert(fpath, fig, fname_ending, "L")
        plt.close("all")

    fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)

    ax.scatter(words_df[f"{prefix}_x_center"], words_df[f"{prefix}_y_center"], s=1, facecolor="k", alpha=0.01)
    for idx in range(len(x_left)):
        ax.text(
            words_df[f"{prefix}_x_center"][idx],
            words_df[f"{prefix}_y_center"][idx],
            words_df[prefix][idx],
            horizontalalignment="center",
            verticalalignment="center",
            fontproperties=font_props,
        )
    fname_ending = f"{prefix}s_grey"
    words_grey_im = save_im_load_convert(fpath, fig, fname_ending, "L")

    plt.close("all")
    fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)

    ax.scatter(words_df[f"{prefix}_x_center"], words_df[f"{prefix}_y_center"], s=1, facecolor="k", alpha=0.1)
    for idx in range(len(x_left)):
        xdiff = x_right[idx] - x_left[idx]
        ydiff = y_top[idx] - y_bottom[idx]
        rect = patches.Rectangle(
            (x_left[idx] - 1, y_bottom[idx] - 1), xdiff, ydiff, alpha=0.9, linewidth=1, edgecolor="k", facecolor="grey"
        )  # seems to need one pixel offset
        ax.add_patch(rect)
    fname_ending = f"{prefix}_boxes_grey"
    word_boxes_grey_im = save_im_load_convert(fpath, fig, fname_ending, "L")

    plt.close("all")

    fig, ax = get_fig_ax(screen_res, dpi, words_df, x_margin, y_margin, prefix=prefix)

    ax.scatter(dffix.x, dffix.y, facecolor="k", alpha=times)
    fname_ending = "fix_scatter_grey"
    fix_scatter_grey_im = save_im_load_convert(fpath, fig, fname_ending, "L")

    plt.close("all")

    arr_combo = np.stack(
        [
            np.asarray(words_grey_im),
            np.asarray(word_boxes_grey_im),
            np.asarray(fix_scatter_grey_im),
        ],
        axis=2,
    )

    im_combo = Image.fromarray(arr_combo)
    fname_ending = f"{prefix}s_channel_sep"

    save_path = get_save_path(fpath, fname_ending)
    print(f"save_path for im combo is {save_path}")
    im_combo.save(fpath)

    if save_channel_repeats:
        arr_combo = np.stack([np.asarray(words_grey_im)] * 3, axis=2)
        im_combo = Image.fromarray(arr_combo)
        fname_ending = f"{prefix}s_channel_repeat"

        save_path = get_save_path(fpath, fname_ending)
        im_combo.save(save_path)

        arr_combo = np.stack([np.asarray(word_boxes_grey_im)] * 3, axis=2)

        im_combo = Image.fromarray(arr_combo)
        fname_ending = f"{prefix}boxes_channel_repeat"

        save_path = get_save_path(fpath, fname_ending)
        im_combo.save(save_path)

        arr_combo = np.stack([np.asarray(fix_scatter_grey_im)] * 3, axis=2)

        im_combo = Image.fromarray(arr_combo)
        fname_ending = "fix_channel_repeat"

        save_path = get_save_path(fpath, fname_ending)
        im_combo.save(save_path)


def add_line_overlaps_to_sample(trial, sample):
    char_df = pd.DataFrame(trial["chars_list"])
    line_overlaps = []
    for arr in sample:
        y_val = arr[1]
        line_overlap = t.tensor(-1, dtype=t.float32)
        for idx, (x1, x2) in enumerate(zip(char_df.char_ymin.unique(), char_df.char_ymax.unique())):
            if x1 <= y_val <= x2:
                line_overlap = t.tensor(idx, dtype=t.float32)
                break
        line_overlaps.append(line_overlap)
    line_olaps_tensor = t.stack(line_overlaps, dim=0)
    sample = t.cat([sample, line_olaps_tensor.unsqueeze(1)], dim=1)
    return sample


def norm_coords_by_letter_min_x_y(
    sample_idx: int,
    trialslist: list,
    samplelist: list,
    chars_center_coords_list: list = None,
):
    chars_df = pd.DataFrame(trialslist[sample_idx]["chars_list"])
    trialslist[sample_idx]["x_char_unique"] = chars_df.char_xmin.unique()

    min_x_chars = chars_df.char_xmin.min()
    min_y_chars = chars_df.char_ymin.min()

    norm_vector_substract = t.zeros(
        (1, samplelist[sample_idx].shape[1]), dtype=samplelist[sample_idx].dtype, device=samplelist[sample_idx].device
    )
    norm_vector_substract[0, 0] = norm_vector_substract[0, 0] + 1 * min_x_chars
    norm_vector_substract[0, 1] = norm_vector_substract[0, 1] + 1 * min_y_chars

    samplelist[sample_idx] = samplelist[sample_idx] - norm_vector_substract

    if chars_center_coords_list is not None:
        norm_vector_substract = norm_vector_substract.squeeze(0)[:2]
        if chars_center_coords_list[sample_idx].shape[-1] == norm_vector_substract.shape[-1] * 2:
            chars_center_coords_list[sample_idx][:, :2] -= norm_vector_substract
            chars_center_coords_list[sample_idx][:, 2:] -= norm_vector_substract
        else:
            chars_center_coords_list[sample_idx] -= norm_vector_substract
    return trialslist, samplelist, chars_center_coords_list


def norm_coords_by_letter_positions(
    sample_idx: int,
    trialslist: list,
    samplelist: list,
    meanlist: list = None,
    stdlist: list = None,
    return_mean_std_lists=False,
    norm_by_char_averages=False,
    chars_center_coords_list: list = None,
    add_normalised_values_as_features=False,
):
    chars_df = pd.DataFrame(trialslist[sample_idx]["chars_list"])
    trialslist[sample_idx]["x_char_unique"] = chars_df.char_xmin.unique()

    min_x_chars = chars_df.char_xmin.min()
    max_x_chars = chars_df.char_xmax.max()

    norm_vector_multi = t.ones(
        (1, samplelist[sample_idx].shape[1]), dtype=samplelist[sample_idx].dtype, device=samplelist[sample_idx].device
    )
    if norm_by_char_averages:
        chars_list = trialslist[sample_idx]["chars_list"]
        char_widths = np.asarray([x["char_xmax"] - x["char_xmin"] for x in chars_list])
        char_heights = np.asarray([x["char_ymax"] - x["char_ymin"] for x in chars_list])
        char_widths_average = np.mean(char_widths[char_widths > 0])
        char_heights_average = np.mean(char_heights[char_heights > 0])

        norm_vector_multi[0, 0] = norm_vector_multi[0, 0] * char_widths_average
        norm_vector_multi[0, 1] = norm_vector_multi[0, 1] * char_heights_average

    else:
        line_height = min(np.unique(trialslist[sample_idx]["line_heights"]))
        line_width = max_x_chars - min_x_chars
        norm_vector_multi[0, 0] = norm_vector_multi[0, 0] * line_width
        norm_vector_multi[0, 1] = norm_vector_multi[0, 1] * line_height
    assert ~t.any(t.isnan(norm_vector_multi)), "Nan found in char norming vector"

    norm_vector_multi = norm_vector_multi.squeeze(0)
    if add_normalised_values_as_features:
        norm_vector_multi = norm_vector_multi[norm_vector_multi != 1]
        normed_features = samplelist[sample_idx][:, : norm_vector_multi.shape[0]] / norm_vector_multi
        samplelist[sample_idx] = t.cat([samplelist[sample_idx], normed_features], dim=1)
    else:
        samplelist[sample_idx] = samplelist[sample_idx] / norm_vector_multi  #  in case time or pupil size is included
    if chars_center_coords_list is not None:
        norm_vector_multi = norm_vector_multi[:2]
        if chars_center_coords_list[sample_idx].shape[-1] == norm_vector_multi.shape[-1] * 2:
            chars_center_coords_list[sample_idx][:, :2] /= norm_vector_multi
            chars_center_coords_list[sample_idx][:, 2:] /= norm_vector_multi
        else:
            chars_center_coords_list[sample_idx] /= norm_vector_multi
    if return_mean_std_lists:
        mean_val = samplelist[sample_idx].mean(axis=0).cpu().numpy()
        meanlist.append(mean_val)
        std_val = samplelist[sample_idx].std(axis=0).cpu().numpy()
        stdlist.append(std_val)
        assert ~any(np.isnan(mean_val)), "Nan found in mean_val"
        assert ~any(np.isnan(mean_val)), "Nan found in std_val"

        return trialslist, samplelist, meanlist, stdlist, chars_center_coords_list
    return trialslist, samplelist, chars_center_coords_list


def remove_compile_from_model(model):
    if hasattr(model.project, "_orig_mod"):
        model.project = model.project._orig_mod
        model.chars_conv = model.chars_conv._orig_mod
        model.chars_classifier = model.chars_classifier._orig_mod
        model.layer_norm_in = model.layer_norm_in._orig_mod
        model.bert_model = model.bert_model._orig_mod
        model.linear = model.linear._orig_mod
    else:
        print(f"remove_compile_from_model not done since model.project {model.project} has no orig_mod")
    return model


def remove_compile_from_dict(state_dict):
    for key in list(state_dict.keys()):
        newkey = key.replace("._orig_mod.", ".")
        state_dict[newkey] = state_dict.pop(key)
    return state_dict


def add_text_to_ax(
    chars_list,
    ax,
    font_to_use="DejaVu Sans Mono",
    fontsize=21,
    prefix="char",
    plot_boxes=True,
    plot_text=True,
    box_annotations=None,
):
    font_props = FontProperties(family=font_to_use, style="normal", size=fontsize)
    if not plot_boxes and not plot_text:
        return None
    if box_annotations is None:
        enum = chars_list
    else:
        enum = zip(chars_list, box_annotations)
    for v in enum:
        if box_annotations is not None:
            v, annot_text = v
        x0, y0 = v[f"{prefix}_xmin"], v[f"{prefix}_ymin"]
        xdiff, ydiff = v[f"{prefix}_xmax"] - v[f"{prefix}_xmin"], v[f"{prefix}_ymax"] - v[f"{prefix}_ymin"]
        if plot_text:
            ax.text(
                v[f"{prefix}_x_center"],
                v[f"{prefix}_y_center"],
                v[prefix],
                horizontalalignment="center",
                verticalalignment="center",
                fontproperties=font_props,
            )
        if plot_boxes:
            ax.add_patch(Rectangle((x0, y0), xdiff, ydiff, edgecolor="grey", facecolor="none", lw=0.8, alpha=0.4))
        if box_annotations is not None:
            ax.annotate(
                str(annot_text),
                (x0 + xdiff / 2, y0),
                horizontalalignment="center",
                verticalalignment="center",
                fontproperties=FontProperties(family=font_to_use, style="normal", size=fontsize / 1.5),
            )


def plot_fixations_and_text(
    dffix: pd.DataFrame,
    trial: dict,
    plot_prefix="chars_",
    show=False,
    returnfig=False,
    save=False,
    savelocation="plot.png",
    font_to_use="DejaVu Sans Mono",
    fontsize=20,
    plot_classic=True,
    plot_boxes=True,
    plot_text=True,
    fig_size=(14, 8),
    dpi=300,
    turn_axis_on=True,
    algo_choice="slice",
):
    fig, ax = plt.subplots(1, 1, figsize=fig_size, tight_layout=True, dpi=dpi)
    if f"{plot_prefix}list" in trial.keys():
        add_text_to_ax(
            trial[f"{plot_prefix}list"],
            ax,
            font_to_use,
            fontsize=fontsize,
            prefix=plot_prefix[:-2],
            plot_boxes=plot_boxes,
            plot_text=plot_text,
        )
    ax.plot(dffix.x, dffix.y, "kX", label="Raw Fixations", alpha=0.9)

    if plot_classic and f"line_num_{algo_choice}" in dffix.columns:
        ax.scatter(
            dffix.x,
            dffix[f"y_{algo_choice}"],
            marker="*",
            color="tab:green",
            label=f"{algo_choice} Prediction",
            alpha=0.9,
        )
        for x_before, y_before, x_after, y_after in zip(
            dffix.x.values, dffix[f"y_{algo_choice}"].values, dffix.x, dffix.y
        ):
            arr_delta_x = x_after - x_before
            arr_delta_y = y_after - y_before
            ax.arrow(x_before, y_before, arr_delta_x, arr_delta_y, color="tab:green", alpha=0.6)
    ax.set_ylabel("y (pixel)")
    ax.set_xlabel("x (pixel)")

    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    if not turn_axis_on:
        ax.axis("off")
    if save:
        plt.savefig(savelocation, dpi=dpi)
    if show:
        plt.show()
    if returnfig:
        return fig
    else:
        plt.close()
        return None


def make_folders(gradio_temp_folder, gradio_temp_unzipped_folder, gradio_plots):
    gradio_temp_folder.mkdir(exist_ok=True)
    gradio_temp_unzipped_folder.mkdir(exist_ok=True)
    gradio_plots.mkdir(exist_ok=True)
    return 0


def get_classic_cfg(fname):
    with open(fname, "r") as f:
        jsonsstring = f.read()
    classic_algos_cfg = json.loads(jsonsstring)
    classic_algos_cfg["slice"] = classic_algos_cfg["slice"]
    classic_algos_cfg = classic_algos_cfg
    return classic_algos_cfg


def find_and_load_model(model_date="20240104-223349"):
    model_cfg_file = list(DIST_MODELS_FOLDER.glob(f"*{model_date}*.yaml"))
    if len(model_cfg_file) == 0:
        if "logger" in st.session_state:
            st.session_state["logger"].warning(f"No model cfg yaml found for {model_date}")
        return None, None
    model_cfg_file = model_cfg_file[0]
    with open(model_cfg_file) as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["system_type"] = "linux"
    model_file = list(pl.Path("models").glob(f"*{model_date}*.ckpt"))[0]
    model = load_model(model_file, model_cfg)

    return model, model_cfg


def load_model(model_file, cfg):
    try:
        model_loaded = t.load(model_file, map_location="cpu")
        if "hyper_parameters" in model_loaded.keys():
            model_cfg_temp = model_loaded["hyper_parameters"]["cfg"]
        else:
            model_cfg_temp = cfg
        model_state_dict = model_loaded["state_dict"]
    except Exception as e:
        if "logger" in st.session_state:
            st.session_state["logger"].warning(e)
        if "logger" in st.session_state:
            st.session_state["logger"].warning(f"Failed to load {model_file}")
        return None
    model = LitModel(
        [1, 500, 3],
        model_cfg_temp["hidden_dim_bert"],
        model_cfg_temp["num_attention_heads"],
        model_cfg_temp["n_layers_BERT"],
        model_cfg_temp["loss_function"],
        1e-4,
        model_cfg_temp["weight_decay"],
        model_cfg_temp,
        model_cfg_temp["use_lr_warmup"],
        model_cfg_temp["use_reduce_on_plateau"],
        track_gradient_histogram=model_cfg_temp["track_gradient_histogram"],
        register_forw_hook=model_cfg_temp["track_activations_via_hook"],
        char_dims=model_cfg_temp["char_dims"],
    )
    model = remove_compile_from_model(model)
    model_state_dict = remove_compile_from_dict(model_state_dict)
    with t.no_grad():
        model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    model.freeze()
    return model


def set_up_models(dist_models_folder):
    out_dict = {}
    if "logger" in st.session_state:
        st.session_state["logger"].info("Loading Ensemble")
    dist_models_with_norm = list(dist_models_folder.glob("*normalize_by_line_height_and_width_True*.ckpt"))
    dist_models_without_norm = list(dist_models_folder.glob("*normalize_by_line_height_and_width_False*.ckpt"))
    DIST_MODEL_DATE_WITH_NORM = dist_models_with_norm[0].stem.split("_")[1]

    models_without_norm_df = [find_and_load_model(m_file.stem.split("_")[1]) for m_file in dist_models_without_norm]
    models_with_norm_df = [find_and_load_model(m_file.stem.split("_")[1]) for m_file in dist_models_with_norm]

    model_cfg_without_norm_df = [x[1] for x in models_without_norm_df if x[1] is not None][0]
    model_cfg_with_norm_df = [x[1] for x in models_with_norm_df if x[1] is not None][0]

    models_without_norm_df = [x[0] for x in models_without_norm_df if x[0] is not None]
    models_with_norm_df = [x[0] for x in models_with_norm_df if x[0] is not None]

    ensemble_model_avg = EnsembleModel(
        models_without_norm_df, models_with_norm_df, learning_rate=0.0058, use_simple_average=True
    )
    out_dict["ensemble_model_avg"] = ensemble_model_avg

    out_dict["model_cfg_without_norm_df"] = model_cfg_without_norm_df
    out_dict["model_cfg_with_norm_df"] = model_cfg_with_norm_df

    single_DIST_model, single_DIST_model_cfg = find_and_load_model(model_date=DIST_MODEL_DATE_WITH_NORM)
    out_dict["DIST_MODEL_DATE_WITH_NORM"] = DIST_MODEL_DATE_WITH_NORM
    out_dict["single_DIST_model"] = single_DIST_model
    out_dict["single_DIST_model_cfg"] = single_DIST_model_cfg
    return out_dict


def prep_data_for_dist(model_cfg, dffix, trial=None):
    if "logger" in st.session_state:
        st.session_state["logger"].debug("prep_data_for_dist entered")
    if trial is None:
        trial = st.session_state["trial"]
    if isinstance(dffix, dict):
        dffix = dffix["value"]
    sample_tensor = t.tensor(dffix.loc[:, model_cfg["sample_cols"]].to_numpy(), dtype=t.float32)

    if model_cfg["add_line_overlap_feature"]:
        sample_tensor = add_line_overlaps_to_sample(trial, sample_tensor)

    has_nans = t.any(t.isnan(sample_tensor))
    assert not has_nans, "NaNs found in sample tensor"
    samplelist_eval = [sample_tensor]
    trialslist_eval = [trial]
    chars_center_coords_list_eval = None
    if model_cfg["norm_coords_by_letter_min_x_y"]:
        for sample_idx, _ in enumerate(samplelist_eval):
            trialslist_eval, samplelist_eval, chars_center_coords_list_eval = norm_coords_by_letter_min_x_y(
                sample_idx,
                trialslist_eval,
                samplelist_eval,
                chars_center_coords_list=chars_center_coords_list_eval,
            )

    if model_cfg["normalize_by_line_height_and_width"]:
        meanlist_eval, stdlist_eval = [], []
        for sample_idx, _ in enumerate(samplelist_eval):
            (
                trialslist_eval,
                samplelist_eval,
                meanlist_eval,
                stdlist_eval,
                chars_center_coords_list_eval,
            ) = norm_coords_by_letter_positions(
                sample_idx,
                trialslist_eval,
                samplelist_eval,
                meanlist_eval,
                stdlist_eval,
                return_mean_std_lists=True,
                norm_by_char_averages=model_cfg["norm_by_char_averages"],
                chars_center_coords_list=chars_center_coords_list_eval,
                add_normalised_values_as_features=model_cfg["add_normalised_values_as_features"],
            )
    sample_tensor = samplelist_eval[0]
    sample_means = t.tensor(model_cfg["sample_means"], dtype=t.float32)
    sample_std = t.tensor(model_cfg["sample_std"], dtype=t.float32)
    sample_tensor = (sample_tensor - sample_means) / sample_std
    sample_tensor = sample_tensor.unsqueeze(0)

    if "logger" in st.session_state:
        st.session_state["logger"].info(f"Using path {trial['plot_file']} for plotting")
    plot_text_boxes_fixations(
        fpath=trial["plot_file"],
        dpi=250,
        screen_res=(1024, 768),
        data_dir_sub=None,
        set_font_size=True,
        font_size=4,
        use_words=False,
        save_channel_repeats=False,
        save_combo_grey_and_rgb=False,
        dffix=dffix,
        trial=trial,
    )

    val_set = DSet(
        sample_tensor,
        None,
        t.zeros((1, sample_tensor.shape[1])),
        trialslist_eval,
        padding_list=[0],
        padding_at_end=model_cfg["padding_at_end"],
        return_images_for_conv=True,
        im_partial_string=model_cfg["im_partial_string"],
        input_im_shape=model_cfg["char_plot_shape"],
    )
    val_loader = dl(val_set, batch_size=1, shuffle=False, num_workers=0)
    return val_loader, val_set


def fold_in_seq_dim(out, y=None):
    batch_size, seq_len, num_classes = out.shape

    out = eo.rearrange(out, "b s c -> (b s) c", s=seq_len)
    if y is None:
        return out, None
    if len(y.shape) > 2:
        y = eo.rearrange(y, "b s c -> (b s) c", s=seq_len)
    else:
        y = eo.rearrange(y, "b s -> (b s)", s=seq_len)
    return out, y


def logits_to_pred(out, y=None):
    seq_len = out.shape[1]
    out, y = fold_in_seq_dim(out, y)
    preds = corn_label_from_logits(out)
    preds = eo.rearrange(preds, "(b s) -> b s", s=seq_len)
    if y is not None:
        y = eo.rearrange(y.squeeze(), "(b s) -> b s", s=seq_len)
        y = y
    return preds, y


def get_DIST_preds(dffix, trial, models_dict=None):
    algo_choice = "DIST"

    if models_dict is None:
        if st.session_state["single_DIST_model"] is None or st.session_state["single_DIST_model_cfg"] is None:
            st.session_state["single_DIST_model"], st.session_state["single_DIST_model_cfg"] = find_and_load_model(
                model_date=st.session_state["DIST_MODEL_DATE_WITH_NORM"]
            )

            if "logger" in st.session_state:
                st.session_state["logger"].info("Model is None, reiniting model")
        else:
            model = st.session_state["single_DIST_model"]
        loader, dset = prep_data_for_dist(st.session_state["single_DIST_model_cfg"], dffix, trial)
    else:
        model = models_dict["single_DIST_model"]
        loader, dset = prep_data_for_dist(models_dict["single_DIST_model_cfg"], dffix, trial)
    batch = next(iter(loader))

    if "cpu" not in str(model.device):
        batch = [x.cuda() for x in batch]
    try:
        out = model(batch)
        preds, y = logits_to_pred(out, y=None)
        if "logger" in st.session_state:
            st.session_state["logger"].debug(
                f"y_char_unique are {trial['y_char_unique']} for trial {trial['trial_id']}"
            )
        if "logger" in st.session_state:
            st.session_state["logger"].debug(f"trial keys are {trial.keys()} for trial {trial['trial_id']}")
        if "logger" in st.session_state:
            st.session_state["logger"].debug(
                f"chars_list has len {len(trial['chars_list'])} for trial {trial['trial_id']}"
            )
        if "logger" in st.session_state:
            st.session_state["logger"].debug(f"y_char_unique  {trial['y_char_unique']} for trial {trial['trial_id']}")
        if len(trial["y_char_unique"]) < 1:
            y_char_unique = pd.DataFrame(trial["chars_list"]).char_y_center.sort_values().unique()
        else:
            y_char_unique = trial["y_char_unique"]
        num_lines = trial["num_char_lines"] - 1
        preds = t.clamp(preds, 0, num_lines).squeeze().cpu().numpy()
        y_pred_DIST = [y_char_unique[idx] for idx in preds]

        dffix[f"line_num_{algo_choice}"] = preds
        dffix[f"y_{algo_choice}"] = np.round(y_pred_DIST, decimals=1)
        dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(1)
    except Exception as e:
        if "logger" in st.session_state:
            st.session_state["logger"].warning(f"Exception on model(batch) for DIST \n{e}")
    return dffix


def get_DIST_ensemble_preds(
    dffix,
    trial,
    model_cfg_without_norm_df,
    model_cfg_with_norm_df,
    ensemble_model_avg,
):
    algo_choice = "DIST-Ensemble"
    loader_without_norm, dset_without_norm = prep_data_for_dist(model_cfg_without_norm_df, dffix, trial)
    loader_with_norm, dset_with_norm = prep_data_for_dist(model_cfg_with_norm_df, dffix, trial)
    batch_without_norm = next(iter(loader_without_norm))
    batch_with_norm = next(iter(loader_with_norm))
    out = ensemble_model_avg((batch_without_norm, batch_with_norm))
    preds, y = logits_to_pred(out[0]["out_avg"], y=None)
    if len(trial["y_char_unique"]) < 1:
        y_char_unique = pd.DataFrame(trial["chars_list"]).char_y_center.sort_values().unique()
    else:
        y_char_unique = trial["y_char_unique"]
    num_lines = trial["num_char_lines"] - 1
    preds = t.clamp(preds, 0, num_lines).squeeze().cpu().numpy()
    if "logger" in st.session_state:
        st.session_state["logger"].debug(f"preds are {preds} for trial {trial['trial_id']}")
    y_pred_DIST = [y_char_unique[idx] for idx in preds]

    dffix[f"line_num_{algo_choice}"] = preds
    dffix[f"y_{algo_choice}"] = np.round(y_pred_DIST, decimals=1)
    dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(1)
    return dffix


def get_EDIST_preds_with_model_check(dffix, trial, ensemble_model_avg=None, models_dict=None):

    if models_dict is None:
        if ensemble_model_avg is None and "ensemble_model_avg" not in st.session_state:
            if "logger" in st.session_state:
                st.session_state["logger"].info("Ensemble Model is None, reiniting model")
            dist_models_with_norm = DIST_MODELS_FOLDER.glob("*normalize_by_line_height_and_width_True*.ckpt")
            dist_models_without_norm = DIST_MODELS_FOLDER.glob("*normalize_by_line_height_and_width_False*.ckpt")

            models_without_norm_df = [
                find_and_load_model(m_file.stem.split("_")[1]) for m_file in dist_models_without_norm
            ]
            models_with_norm_df = [find_and_load_model(m_file.stem.split("_")[1]) for m_file in dist_models_with_norm]

            model_cfg_without_norm_df = [x[1] for x in models_without_norm_df if x[1] is not None][0]
            model_cfg_with_norm_df = [x[1] for x in models_with_norm_df if x[1] is not None][0]

            models_without_norm_df = [x[0] for x in models_without_norm_df if x[0] is not None]
            models_with_norm_df = [x[0] for x in models_with_norm_df if x[0] is not None]

            ensemble_model_avg = EnsembleModel(
                models_without_norm_df, models_with_norm_df, learning_rate=0.0, use_simple_average=True
            )
            st.session_state["ensemble_model_avg"] = ensemble_model_avg
            st.session_state["model_cfg_without_norm_df"] = model_cfg_without_norm_df
            st.session_state["model_cfg_with_norm_df"] = model_cfg_with_norm_df
        else:
            model_cfg_without_norm_df = st.session_state["model_cfg_without_norm_df"]
            model_cfg_with_norm_df = st.session_state["model_cfg_with_norm_df"]
            ensemble_model_avg = st.session_state["ensemble_model_avg"]
        dffix = get_DIST_ensemble_preds(
            dffix,
            trial,
            st.session_state["model_cfg_without_norm_df"],
            st.session_state["model_cfg_with_norm_df"],
            st.session_state["ensemble_model_avg"],
        )
    else:
        dffix = get_DIST_ensemble_preds(
            dffix,
            trial,
            models_dict["model_cfg_without_norm_df"],
            models_dict["model_cfg_with_norm_df"],
            models_dict["ensemble_model_avg"],
        )
    return dffix


def correct_df(
    dffix,
    algo_choice,
    trial=None,
    for_multi=False,
    ensemble_model_avg=None,
    is_outside_of_streamlit=False,
    classic_algos_cfg=None,
    models_dict=None,
):
    if is_outside_of_streamlit:
        stqdm = tqdm
    else:
        from stqdm import stqdm
    if classic_algos_cfg is None:
        classic_algos_cfg = st.session_state["classic_algos_cfg"]
    if trial is None and not for_multi:
        trial = st.session_state["trial"]
    if "logger" in st.session_state:
        st.session_state["logger"].info(f"Applying {algo_choice} to fixations for trial {trial['trial_id']}")

    if isinstance(dffix, dict):
        dffix = dffix["value"]
    if "x" not in dffix.keys() or "x" not in dffix.keys():
        if "logger" in st.session_state:
            st.session_state["logger"].warning(f"x or y not in dffix")
        if "logger" in st.session_state:
            st.session_state["logger"].warning(dffix.columns)
        return dffix
    if isinstance(algo_choice, list):
        algo_choices = algo_choice
        repeats = range(len(algo_choice))
    else:
        algo_choices = [algo_choice]
        repeats = range(1)
    for algoIdx in stqdm(repeats, desc="Applying correction algorithms"):
        algo_choice = algo_choices[algoIdx]
        st_proc = time.process_time()
        st_wall = time.time()

        if algo_choice == "DIST":
            dffix = get_DIST_preds(dffix, trial, models_dict=models_dict)

        elif algo_choice == "DIST-Ensemble":
            dffix = get_EDIST_preds_with_model_check(dffix, trial, models_dict=models_dict)
        elif algo_choice == "Wisdom_of_Crowds_with_DIST":
            dffix, corrections = get_all_classic_preds(dffix, trial, classic_algos_cfg)
            dffix = get_DIST_preds(dffix, trial, models_dict=models_dict)
            for _ in range(3):
                corrections.append(np.asarray(dffix.loc[:, "y_DIST"]))
            dffix = apply_woc(dffix, trial, corrections, algo_choice)
        elif algo_choice == "Wisdom_of_Crowds_with_DIST_Ensemble":
            dffix, corrections = get_all_classic_preds(dffix, trial, classic_algos_cfg)
            dffix = get_EDIST_preds_with_model_check(dffix, trial, ensemble_model_avg, models_dict=models_dict)
            for _ in range(3):
                corrections.append(np.asarray(dffix.loc[:, "y_DIST-Ensemble"]))
            dffix = apply_woc(dffix, trial, corrections, algo_choice)
        elif algo_choice == "Wisdom_of_Crowds":
            dffix, corrections = get_all_classic_preds(dffix, trial, classic_algos_cfg)
            dffix = apply_woc(dffix, trial, corrections, algo_choice)

        else:
            algo_cfg = classic_algos_cfg[algo_choice]
            dffix = calgo.apply_classic_algo(dffix, trial, algo_choice, algo_cfg)
            dffix[f"y_{algo_choice}_correction"] = (dffix.loc[:, f"y_{algo_choice}"] - dffix.loc[:, "y"]).round(1)

        et_proc = time.process_time()
        time_proc = et_proc - st_proc
        et_wall = time.time()
        time_wall = et_wall - st_wall
        if "logger" in st.session_state:
            st.session_state["logger"].info(f"time_proc {algo_choice} {time_proc}")
        if "logger" in st.session_state:
            st.session_state["logger"].info(f"time_wall {algo_choice} {time_wall}")
    if for_multi:
        return dffix
    else:
        if "start_time" in dffix.columns:
            dffix = dffix.drop(axis=1, labels=["start_time", "end_time"])
        return dffix, export_csv(dffix, trial)

def set_font_from_chars_list(trial):
    
    if "chars_list" in trial:
        chars_df = pd.DataFrame(trial["chars_list"])
        line_diffs = np.diff(chars_df.char_y_center.unique())
        y_diffs = np.unique(line_diffs)
        if len(y_diffs) == 1:
            y_diff = y_diffs[0]
        else:
            y_diff = np.min(y_diffs)
        y_diff = round(y_diff * 2) / 2

    else:
        y_diff = 1 / 0.333 * 18
    font_size = y_diff * 0.333  # pixel to point conversion
    return round((font_size)*4,ndigits=0)/4

def get_font_and_font_size_from_trial(trial):
    font_face, font_size, dpi, screen_res = get_plot_props(trial, AVAILABLE_FONTS)

    if font_size is None and "font_size" in trial:
        font_size = trial["font_size"]
    elif font_size is None:
        font_size = set_font_from_chars_list(trial)
    return font_face, font_size


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def matplotlib_plot_df(
    dffix,
    trial,
    algo_choice,
    stimulus_prefix="word",
    desired_dpi=300,
    fix_to_plot=[],
    stim_info_to_plot=["Words", "Word boxes"],
    box_annotations=None,
):
    chars_df = pd.DataFrame(trial["chars_list"]) if "chars_list" in trial else None

    if chars_df is not None:
        font_face, font_size = get_font_and_font_size_from_trial(trial)
        font_size = font_size * 0.65
    else:
        st.warning("No character or word information available to plot")

    if "display_coords" in trial:
        desired_width_in_pixels = trial["display_coords"][2] + 1
        desired_height_in_pixels = trial["display_coords"][3] + 1
    else:
        desired_width_in_pixels = 1920
        desired_height_in_pixels = 1080

    figure_width = desired_width_in_pixels / desired_dpi
    figure_height = desired_height_in_pixels / desired_dpi

    fig = plt.figure(figsize=(figure_width, figure_height), dpi=desired_dpi)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    if "font" in trial and trial["font"] in AVAILABLE_FONTS:
        font_to_use = trial["font"]
    else:
        font_to_use = "DejaVu Sans Mono"
    if "font_size" in trial:
        font_size = trial["font_size"]
    else:
        font_size = 20

    if f"{stimulus_prefix}s_list" in trial:
        add_text_to_ax(
            trial[f"{stimulus_prefix}s_list"],
            ax,
            font_to_use,
            prefix=stimulus_prefix,
            fontsize=font_size / 3.89,
            plot_text=False,
            plot_boxes=True if "Word boxes" in stim_info_to_plot else False,
            box_annotations=box_annotations,
        )

    if "chars_list" in trial:
        add_text_to_ax(
            trial["chars_list"],
            ax,
            font_to_use,
            prefix="char",
            fontsize=font_size / 3.89,
            plot_text=True if "Words" in stim_info_to_plot else False,
            plot_boxes=False,
            box_annotations=None,
        )

    if "Uncorrected Fixations" in fix_to_plot:
        ax.plot(dffix.x, dffix.y, label="Raw fixations", color="blue", alpha=0.6, linewidth=0.6)

        x0 = dffix.x.iloc[range(len(dffix.x) - 1)].values
        x1 = dffix.x.iloc[range(1, len(dffix.x))].values
        y0 = dffix.y.iloc[range(len(dffix.y) - 1)].values
        y1 = dffix.y.iloc[range(1, len(dffix.y))].values
        xpos = x0
        ypos = y0
        xdir = x1 - x0
        ydir = y1 - y0
        for X, Y, dX, dY in zip(xpos, ypos, xdir, ydir):
            ax.annotate(
                "",
                xytext=(X, Y),
                xy=(X + 0.001 * dX, Y + 0.001 * dY),
                arrowprops=dict(arrowstyle="fancy", color="blue"),
                size=8,
                alpha=0.3,
            )
    if "Corrected Fixations" in fix_to_plot:
        if isinstance(algo_choice, list):
            algo_choices = algo_choice
            repeats = range(len(algo_choice))
        else:
            algo_choices = [algo_choice]
            repeats = range(1)
        for algoIdx in repeats:
            algo_choice = algo_choices[algoIdx]
            if f"y_{algo_choice}" in dffix.columns:
                ax.plot(
                    dffix.x,
                    dffix.loc[:, f"y_{algo_choice}"],
                    label="Raw fixations",
                    color=COLORS[algoIdx],
                    alpha=0.6,
                    linewidth=0.6,
                )

                x0 = dffix.x.iloc[range(len(dffix.x) - 1)].values
                x1 = dffix.x.iloc[range(1, len(dffix.x))].values
                y0 = dffix.loc[:, f"y_{algo_choice}"].iloc[range(len(dffix.loc[:, f"y_{algo_choice}"]) - 1)].values
                y1 = dffix.loc[:, f"y_{algo_choice}"].iloc[range(1, len(dffix.loc[:, f"y_{algo_choice}"]))].values
                xpos = x0
                ypos = y0
                xdir = x1 - x0
                ydir = y1 - y0
                for X, Y, dX, dY in zip(xpos, ypos, xdir, ydir):
                    ax.annotate(
                        "",
                        xytext=(X, Y),
                        xy=(X + 0.001 * dX, Y + 0.001 * dY),
                        arrowprops=dict(arrowstyle="fancy", color=COLORS[algoIdx]),
                        size=8,
                        alpha=0.3,
                    )

    ax.set_xlim((0, desired_width_in_pixels))
    ax.set_ylim((0, desired_height_in_pixels))
    ax.invert_yaxis()

    return fig, desired_width_in_pixels, desired_height_in_pixels


def plotly_plot_with_image(
    dffix,
    trial,
    algo_choice,
    to_plot_list=["Uncorrected Fixations", "Words", "corrected fixations", "Word boxes"],
    scale_factor=0.5,
):
    fig, img_width, img_height = matplotlib_plot_df(
        dffix, trial, algo_choice, desired_dpi=300, fix_to_plot=[], stim_info_to_plot=to_plot_list
    )
    fig.savefig(TEMP_FIGURE_STIMULUS_PATH)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[img_height * scale_factor, 0],
            mode="markers",
            marker_opacity=0,
            name="scale_helper",
        )
    )

    fig.update_xaxes(visible=False, range=[0, img_width * scale_factor])

    fig.update_yaxes(
        visible=False,
        range=[img_height * scale_factor, 0],
        scaleanchor="x",
    )
    if "Words" in to_plot_list or "Word boxes" in to_plot_list:
        imsource = Image.open(str(TEMP_FIGURE_STIMULUS_PATH))
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=0,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=imsource,
            )
        )

    if "Uncorrected Fixations" in to_plot_list:
        duration_scaled = dffix.duration - dffix.duration.min()
        duration_scaled = ((duration_scaled / duration_scaled.max()) - 0.5) * 3
        duration = sigmoid(duration_scaled) * 50 * scale_factor
        fig.add_trace(
            go.Scatter(
                x=dffix.x * scale_factor,
                y=dffix.y * scale_factor,
                mode="markers+lines+text",
                name="Raw fixations",
                marker=dict(
                    color=COLORS[-1],
                    symbol="arrow",
                    size=duration.values,
                    angleref="previous",
                    line=dict(color="black", width=duration.values / 10),
                ),
                line_width=2 * scale_factor,
                text=np.arange(len(dffix.x)),
                textposition="middle right",
                textfont=dict(
                    family="sans serif",
                    size=18 * scale_factor,
                ),
                hoverinfo="text+x+y",
                opacity=0.9,
            )
        )

    if "Corrected Fixations" in to_plot_list:
        if isinstance(algo_choice, list):
            algo_choices = algo_choice
            repeats = range(len(algo_choice))
        else:
            algo_choices = [algo_choice]
            repeats = range(1)
        for algoIdx in repeats:
            algo_choice = algo_choices[algoIdx]
            if f"y_{algo_choice}" in dffix.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dffix.x * scale_factor,
                        y=dffix.loc[:, f"y_{algo_choice}"] * scale_factor,
                        mode="markers",
                        name=f"{algo_choice} corrected",
                        marker_color=COLORS[algoIdx],
                        marker_size=10 * scale_factor,
                        hoverinfo="text+x+y",
                        opacity=0.75,
                    )
                )

    fig.update_layout(
        plot_bgcolor=None,
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=0.8),
    )

    for trace in fig["data"]:
        if trace["name"] == "scale_helper":
            trace["showlegend"] = False
    return fig


def plot_y_corr(dffix, algo_choice, margin=dict(t=40, l=10, r=10, b=1)):
    num_datapoints = len(dffix.x)

    layout = dict(
        plot_bgcolor="white",
        autosize=True,
        margin=margin,
        xaxis=dict(
            title="Fixation Index",
            linecolor="black",
            range=[-1, num_datapoints + 1],
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        yaxis=dict(
            title="y correction",
            side="left",
            linecolor="black",
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        legend=dict(orientation="v", yanchor="middle", y=0.95, xanchor="left", x=1.05),
    )
    if isinstance(dffix, dict):
        dffix = dffix["value"]
    algo_string = algo_choice[0] if isinstance(algo_choice, list) else algo_choice
    if f"y_{algo_string}_correction" not in dffix.columns:
        st.session_state["logger"].warning("No correction column found in dataframe")
        return go.Figure(layout=layout)
    if isinstance(dffix, dict):
        dffix = dffix["value"]

    fig = go.Figure(layout=layout)

    if isinstance(algo_choice, list):
        algo_choices = algo_choice
        repeats = range(len(algo_choice))
    else:
        algo_choices = [algo_choice]
        repeats = range(1)
    for algoIdx in repeats:
        algo_choice = algo_choices[algoIdx]
        fig.add_trace(
            go.Scatter(
                x=np.arange(num_datapoints),
                y=dffix.loc[:, f"y_{algo_choice}_correction"],
                mode="markers",
                name=f"{algo_choice} y correction",
                marker_color=COLORS[algoIdx],
                marker_size=3,
                showlegend=True,
            )
        )
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")

    return fig


def download_example_ascs(EXAMPLES_FOLDER, EXAMPLES_ASC_ZIP_FILENAME, OSF_DOWNLAOD_LINK, EXAMPLES_FOLDER_PATH):
    if not os.path.isdir(EXAMPLES_FOLDER):
        os.mkdir(EXAMPLES_FOLDER)

    if not os.path.exists(EXAMPLES_ASC_ZIP_FILENAME):
        download_url(OSF_DOWNLAOD_LINK, EXAMPLES_ASC_ZIP_FILENAME)
        # os.system(f'''wget -O {EXAMPLES_ASC_ZIP_FILENAME} -c --read-timeout=5 --tries=0 "{OSF_DOWNLAOD_LINK}"''')

    if os.path.exists(EXAMPLES_ASC_ZIP_FILENAME):
        if EXAMPLES_FOLDER_PATH.exists():
            EXAMPLE_ASC_FILES = [x for x in EXAMPLES_FOLDER_PATH.glob("*.asc")]
        if len(EXAMPLE_ASC_FILES) != 4:
            try:
                with zipfile.ZipFile(EXAMPLES_ASC_ZIP_FILENAME, "r") as zip_ref:
                    zip_ref.extractall(EXAMPLES_FOLDER)
            except Exception as e:
                st.session_state["logger"].warning(e)
                st.session_state["logger"].warning(f"Extracting {EXAMPLES_ASC_ZIP_FILENAME} failed")

        EXAMPLE_ASC_FILES = [x for x in EXAMPLES_FOLDER_PATH.glob("*.asc")]
    return EXAMPLE_ASC_FILES


def process_trial_choice_single_csv(trial, algo_choice, file=None):
    trial_id = trial["trial_id"]
    if "dffix" in trial:
        dffix = trial["dffix"]
    else:
        if file is None:
            file = st.session_state["single_csv_file"]
        trial["plot_file"] = str(PLOTS_FOLDER.joinpath(f"{file.name}_{trial_id}_2ndInput_chars_channel_sep.png"))
        trial["fname"] = str(file.name)
        dffix = trial["dffix"] = st.session_state["trials_by_ids_single_csv"][trial_id]["dffix"]

    font, font_size, dpi, screen_res = get_plot_props(trial, AVAILABLE_FONTS)
    chars_df = pd.DataFrame(trial["chars_list"])
    trial["chars_df"] = chars_df.to_dict()
    trial["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())
    if algo_choice is not None:
        dffix, _ = correct_df(dffix, algo_choice, trial)
    return dffix, trial, dpi, screen_res, font, font_size


def add_default_font_and_character_props_to_state(trial):
    chars_list = trial["chars_list"]
    chars_df = pd.DataFrame(trial["chars_list"])
    line_diffs = np.diff(chars_df.char_y_center.unique())
    y_diffs = np.unique(line_diffs)
    if len(y_diffs) == 1:
        y_diff = y_diffs[0]
    else:
        y_diff = np.min(y_diffs)
    y_diff = round(y_diff * 2) / 2
    x_txt_start = chars_list[0]["char_xmin"]
    y_txt_start = chars_list[0]["char_y_center"]

    font_face, font_size = get_font_and_font_size_from_trial(trial)

    line_height = y_diff
    return y_diff, x_txt_start, y_txt_start, font_face, font_size, line_height

def get_all_measures(trial, dffix, prefix, use_corrected_fixations=True, correction_algo="warp"):
    if use_corrected_fixations:
        dffix_copy = copy.deepcopy(dffix)
        dffix_copy["y"] = dffix_copy[f"y_{correction_algo}"]
    else:
        dffix_copy = dffix
    initial_landing_position_own_vals = anf.initial_landing_position_own(trial, dffix_copy, prefix).set_index(
        f"{prefix}_index"
    )
    second_pass_duration_own_vals = anf.second_pass_duration_own(trial, dffix_copy, prefix).set_index(f"{prefix}_index")
    number_of_fixations_own_vals = anf.number_of_fixations_own(trial, dffix_copy, prefix).set_index(f"{prefix}_index")
    initial_fixation_duration_own_vals = anf.initial_fixation_duration_own(trial, dffix_copy, prefix).set_index(
        f"{prefix}_index"
    )
    first_of_many_duration_own_vals = anf.first_of_many_duration_own(trial, dffix_copy, prefix).set_index(
        f"{prefix}_index"
    )
    total_fixation_duration_own_vals = anf.total_fixation_duration_own(trial, dffix_copy, prefix).set_index(
        f"{prefix}_index"
    )
    gaze_duration_own_vals = anf.gaze_duration_own(trial, dffix_copy, prefix).set_index(f"{prefix}_index")
    go_past_duration_own_vals = anf.go_past_duration_own(trial, dffix_copy, prefix).set_index(f"{prefix}_index")
    initial_landing_distance_own_vals = anf.initial_landing_distance_own(trial, dffix_copy, prefix).set_index(
        f"{prefix}_index"
    )
    landing_distances_own_vals = anf.landing_distances_own(trial, dffix_copy, prefix).set_index(f"{prefix}_index")
    number_of_regressions_in_own_vals = anf.number_of_regressions_in_own(trial, dffix_copy, prefix).set_index(
        f"{prefix}_index"
    )
    own_measure_df = pd.concat(
        [
            df.drop(prefix, axis=1)
            for df in [
                number_of_fixations_own_vals,
                initial_fixation_duration_own_vals,
                first_of_many_duration_own_vals,
                total_fixation_duration_own_vals,
                gaze_duration_own_vals,
                go_past_duration_own_vals,
                second_pass_duration_own_vals,
                initial_landing_position_own_vals,
                initial_landing_distance_own_vals,
                landing_distances_own_vals,
                number_of_regressions_in_own_vals,
            ]
        ],
        axis=1,
    )
    own_measure_df[prefix] = number_of_fixations_own_vals[prefix]
    first_column = own_measure_df.pop(prefix)
    own_measure_df.insert(0, prefix, first_column)
    own_measure_df.insert(0, f"{prefix}_num", np.arange((own_measure_df.shape[0])))
    return own_measure_df
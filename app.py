import copy
from PIL import Image
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import os

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import pathlib as pl
import json
import logging
import zipfile
from stqdm import stqdm
import jellyfish as jf
import lovely_tensors
import shutil
import eyekit_measures as ekm
import zipfile

import utils as ut

os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

st.set_page_config("Correction", page_icon=":eye:", layout="wide")

AVAILABLE_FONTS = st.session_state["AVAILABLE_FONTS"] = ut.AVAILABLE_FONTS

DEFAULT_PLOT_FONT = "DejaVu Sans Mono"
EXAMPLES_FOLDER = "./testfiles/"
EXAMPLES_ASC_ZIP_FILENAME = "asc_files.zip"
OSF_DOWNLAOD_LINK = "https://osf.io/download/us97f/"
EXAMPLES_FOLDER_PATH = pl.Path(EXAMPLES_FOLDER)


lovely_tensors.monkey_patch()


def make_folders(gradio_temp_folder, gradio_temp_unzipped_folder, gradio_plots):
    return ut.make_folders(gradio_temp_folder, gradio_temp_unzipped_folder, gradio_plots)


TEMP_FOLDER = st.session_state["TEMP_FOLDER"] = ut.TEMP_FOLDER
gradio_temp_unzipped_folder = st.session_state["gradio_temp_unzipped_folder"] = pl.Path("unzipped")

PLOTS_FOLDER = st.session_state["PLOTS_FOLDER"] = pl.Path("plots")
TEMP_FIGURE_STIMULUS_PATH = PLOTS_FOLDER.joinpath("temp_matplotlib_plot_stimulus.png")
make_folders(TEMP_FOLDER, gradio_temp_unzipped_folder, PLOTS_FOLDER)


@st.cache_data
def get_classic_cfg(fname):
    return ut.get_classic_cfg(fname)


classic_algos_cfg = st.session_state["classic_algos_cfg"] = get_classic_cfg("algo_cfgs_all.json")

DIST_MODELS_FOLDER = st.session_state["DIST_MODELS_FOLDER"] = pl.Path("models")
COLORS = st.session_state["COLORS"] = px.colors.qualitative.Alphabet
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


st.session_state["colnames_custom_csv_fix"] = {
    "x_col_name_fix": "x",
    "y_col_name_fix": "y",
    "x_col_name_fix_stim": "char_x_center",
    "x_start_col_name_fix_stim": "char_xmin",
    "x_end_col_name_fix_stim": "char_xmax",
    "y_col_name_fix_stim": "char_y_center",
    "y_start_col_name_fix_stim": "char_ymin",
    "y_end_col_name_fix_stim": "char_ymax",
    "char_col_name_fix_stim": "char",
    "trial_id_col_name_fix": "trial_id",
    "trial_id_col_name_stim": "trial_id",
    "subject_col_name_fix": "subid",
    "subject_col_name_stim": "subid",
    "line_num_col_name_stim": "assigned_line",
    "time_start_col_name_fix": "start",
    "time_stop_col_name_fix": "stop",
}

if "results" not in st.session_state:
    st.session_state["results"] = {}


@st.cache_resource
def load_model(model_file, cfg):
    return ut.load_model(model_file, cfg)


@st.cache_resource
def find_and_load_model(model_date="20240104-223349"):
    return ut.find_and_load_model(model_date)


def create_logger(name, level="DEBUG", file=None):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    if sum([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]) == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d-%(name)s-p%(process)s-{%(pathname)s:%(lineno)d}-%(levelname)s >>> %(message)s",
                "%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(ch)
    if file is not None:
        if sum([isinstance(handler, logging.FileHandler) for handler in logger.handlers]) == 0:
            ch = logging.FileHandler(file, "w")
            ch.setFormatter(
                logging.Formatter(
                    "%(asctime)s.%(msecs)03d-%(name)s-p%(process)s-{%(pathname)s:%(lineno)d}-%(levelname)s >>> %(message)s",
                    "%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(ch)
    logger.debug("Logger added")
    return logger


if "logger" not in st.session_state:
    st.session_state["logger"] = create_logger(name="app", level="DEBUG", file="log_for_app.log")


@st.cache_data
def download_example_ascs(EXAMPLES_FOLDER, EXAMPLES_ASC_ZIP_FILENAME, OSF_DOWNLAOD_LINK, EXAMPLES_FOLDER_PATH):
    return ut.download_example_ascs(EXAMPLES_FOLDER, EXAMPLES_ASC_ZIP_FILENAME, OSF_DOWNLAOD_LINK, EXAMPLES_FOLDER_PATH)


EXAMPLE_ASC_FILES = download_example_ascs(
    EXAMPLES_FOLDER, EXAMPLES_ASC_ZIP_FILENAME, OSF_DOWNLAOD_LINK, EXAMPLES_FOLDER_PATH
)


def asc_to_trial_ids(asc_file, close_gap_between_words=True):
    return ut.asc_to_trial_ids(asc_file, close_gap_between_words)


@st.cache_data
def get_trials_list(asc_file=None, close_gap_between_words=True):
    return ut.get_trials_list(asc_file, close_gap_between_words)


@st.cache_data
def prep_data_for_dist(model_cfg, dffix, trial=None):
    return ut.prep_data_for_dist(model_cfg, dffix, trial)


def save_trial_to_json(trial, savename):
    return ut.save_trial_to_json(trial, savename)


def export_csv(dffix, trial):
    return ut.export_csv(dffix, trial)


@st.cache_data
def get_DIST_preds(dffix, trial):
    return ut.get_DIST_preds(dffix, trial)


@st.cache_data
def get_EDIST_preds_with_model_check(dffix, trial, ensemble_model_avg=None):
    return ut.get_EDIST_preds_with_model_check(dffix, trial, ensemble_model_avg)


def get_all_classic_preds(dffix, trial):
    return ut.get_all_classic_preds(dffix, trial)


def apply_woc(dffix, trial, corrections, algo_choice):
    return ut.apply_woc(dffix, trial, corrections, algo_choice)


@st.cache_data
def correct_df(
    dffix,
    algo_choice,
    trial=None,
    for_multi=False,
    ensemble_model_avg=None,
):
    return ut.correct_df(
        dffix,
        algo_choice,
        trial,
        for_multi,
        ensemble_model_avg,
    )


@st.cache_data
def get_font_and_font_size_from_trial(trial):
    return ut.get_font_and_font_size_from_trial(trial)


@st.cache_data
def add_default_font_and_character_props_to_state(trial):
    return ut.add_default_font_and_character_props_to_state(trial)


@st.cache_data
def get_plot_props(trial, available_fonts):
    return ut.get_plot_props(trial, available_fonts)


def process_trial_choice(trial_id, algo_choice):
    if isinstance(trial_id, dict):
        trial_id = trial_id["value"]
    trials_by_ids = st.session_state["trials_by_ids"]
    trial = trials_by_ids[trial_id]
    if "chars_list" in trial:
        (
            y_diff,
            x_txt_start,
            y_txt_start,
            font_face,
            _,
            line_height,
        ) = add_default_font_and_character_props_to_state(trial)
        font_size = ut.set_font_from_chars_list(trial)

        st.session_state["y_diff_for_eyekit"] = y_diff
        st.session_state["x_txt_start_for_eyekit"] = x_txt_start
        st.session_state["y_txt_start_for_eyekit"] = y_txt_start
        st.session_state["font_face_for_eyekit"] = font_face
        st.session_state["font_size_for_eyekit"] = font_size
        st.session_state["line_height_for_eyekit"] = line_height

    if "dffix" in trial:
        dffix = trial["dffix"]
    else:
        asc_file = st.session_state["asc_file"]
        trial["plot_file"] = str(PLOTS_FOLDER.joinpath(f"{asc_file.stem}_{trial_id}_2ndInput_chars_channel_sep.png"))
        trial["fname"] = str(asc_file.name).split(".")[0]
        df, dffix, trial = ut.trial_to_dfs(trial, st.session_state["lines"], use_synctime=True)
        st.session_state["logger"].info(f"dffix.columns after trial_to_dfs {dffix.columns}")

    font, font_size, dpi, screen_res = ut.get_plot_props(trial, AVAILABLE_FONTS)
    st.session_state["trial"] = trial
    if "chars_list" in trial:
        chars_df = pd.DataFrame(trial["chars_list"])
        trial["chars_df"] = chars_df.to_dict()
        trial["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())
    if algo_choice is not None and ("chars_list" in trial or "words_list" in trial):
        dffix, _ = correct_df(dffix, algo_choice, trial)
    else:
        st.warning("🚨 Stimulus information needed for fixation correction 🚨")

    return dffix, trial, dpi, screen_res, font, font_size


@st.cache_data
def process_trial_choice_single_csv(trial, algo_choice, file=None):
    return ut.process_trial_choice_single_csv(trial, algo_choice, file=file)


def quick_dffix_save(dffix, savename):
    dffix.to_csv(savename)
    st.session_state["logger"].info(f"Saved processed data as {savename}")


def save_trial_to_json(trial, savename):
    if "dffix" in trial:
        trial.pop("dffix")
    with open(savename, "w", encoding="utf-8") as f:
        json.dump(trial, f, ensure_ascii=False, indent=4, cls=ut.NumpyEncoder)


@st.cache_data
def process_trial(trial, asc_file_stem, lines, algo_choice, for_multi=False):
    trial_id = trial["trial_id"]
    trial["plot_file"] = str(PLOTS_FOLDER.joinpath(f"{asc_file_stem}_{trial_id}_2ndInput_chars_channel_sep.png"))
    trial["fname"] = str(asc_file_stem)
    font, font_size, dpi, screen_res = ut.get_plot_props(trial, AVAILABLE_FONTS)
    trial["font"] = font
    trial["font_size"] = font_size
    trial["dpi"] = dpi
    trial["screen_res"] = screen_res
    df, dffix, trial = ut.trial_to_dfs(trial, lines, use_synctime=True)
    if dffix.empty:
        return pd.DataFrame(), trial

    chars_df = pd.DataFrame(trial["chars_list"])
    trial["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())

    trial["chars_df"] = chars_df.to_dict()
    trial["y_char_unique"] = list(chars_df.char_y_center.sort_values().unique())
    if algo_choice is not None:
        dffix = correct_df(dffix, algo_choice, trial, for_multi)

    return dffix, trial


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
    return ut.add_text_to_ax(
        chars_list,
        ax,
        font_to_use=font_to_use,
        fontsize=fontsize,
        prefix=prefix,
        plot_boxes=plot_boxes,
        plot_text=plot_text,
        box_annotations=box_annotations,
    )


@st.cache_data
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
    return ut.matplotlib_plot_df(
        dffix,
        trial,
        algo_choice,
        stimulus_prefix=stimulus_prefix,
        desired_dpi=desired_dpi,
        fix_to_plot=fix_to_plot,
        stim_info_to_plot=stim_info_to_plot,
        box_annotations=box_annotations,
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


@st.cache_data
def plotly_plot_with_image(
    dffix,
    trial,
    algo_choice,
    to_plot_list=["Uncorrected Fixations", "Words", "corrected fixations", "Word boxes"],
    scale_factor=0.5,
):
    return ut.plotly_plot_with_image(
        dffix,
        trial,
        algo_choice,
        to_plot_list=to_plot_list,
        scale_factor=scale_factor,
    )


@st.cache_data
def plot_y_corr(dffix, algo_choice):
    return ut.plot_y_corr(dffix, algo_choice)


def plotly_df(
    dffix=None, trial=None, algo_choice=None, to_plot_list=["fixations", "characters", "corrected fixations"], title=""
):
    if dffix is None:
        dffix = st.session_state["dffix"]
    if algo_choice is None:
        algo_choice = st.session_state["algo_choice"]

    st.session_state["logger"].info(f"Plotting {to_plot_list}")

    num_datapoints = dffix.index
    if trial is None:
        if title in st.session_state["results"]:
            chars_df = pd.DataFrame(st.session_state["results"][title]["trial"]["chars_list"])
        else:
            chars_df = pd.DataFrame(st.session_state["trial"]["chars_df"])
    else:
        chars_df = pd.DataFrame(trial["chars_list"]) if "chars_list" in trial else None
    if chars_df is not None:
        font_face, font_size = get_font_and_font_size_from_trial(trial)
        font_size = font_size * 0.65  # guess for scaling
        xmin = chars_df.char_x_center.min()
        xmax = chars_df.char_x_center.max()
        ymin = chars_df.char_y_center.min()
        ymax = chars_df.char_y_center.max()
    else:
        st.warning("No character or word information available to plot")
        xmin = dffix.x.min()
        xmax = dffix.x.max()
        ymin = dffix.y.min()
        ymax = dffix.y.max()

    layout = dict(
        plot_bgcolor="white",
        autosize=True,
        margin=dict(t=1, l=10, r=10, b=1),
        xaxis=dict(
            title="x-coordinate",
            linecolor="black",
            range=[xmin - 100, xmax + 100],
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        yaxis=dict(
            title="y-coordinate",
            range=[ymax + 100, ymin - 100],
            linecolor="black",
            showgrid=False,
            mirror="all",
            showline=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=0.8),
    )

    fig = go.Figure(layout=layout)

    if "Uncorrected Fixations" in to_plot_list:
        duration_scaled = dffix.duration - dffix.duration.min()
        duration = ((duration_scaled + 0.1) / duration_scaled.median()) * 5
        fig.add_trace(
            go.Scatter(
                x=dffix.x,
                y=dffix.y,
                mode="markers+lines+text",
                name="Raw fixations",
                marker=dict(
                    symbol="arrow",
                    size=duration.values,
                    angleref="previous",
                ),
                line_width=1.2,
                text=num_datapoints,
                textposition="middle right",
                textfont=dict(
                    family="sans serif",
                    size=9,
                ),
                hoverinfo="text+x+y",
                opacity=0.6,
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
                        x=dffix.x,
                        y=dffix.loc[:, f"y_{algo_choice}"],
                        mode="markers",
                        name=f"{algo_choice} corrected",
                        marker_color=st.session_state["COLORS"][algoIdx],
                        marker_size=5,
                        hoverinfo="text+x+y",
                        opacity=0.75,
                    )
                )
    if "Characters" in to_plot_list and chars_df is not None:
        fig.add_trace(
            go.Scatter(
                x=chars_df.char_x_center,
                y=chars_df.char_y_center,
                mode="markers+text",
                name="",
                showlegend=False,
                text=chars_df.char,
                textposition="middle center",
                marker=dict(color="black", size=0.1),
                textfont=dict(family=font_face, size=font_size, color="Black"),
            )
        )

    if "Character boxes (slow to plot)" in to_plot_list and chars_df is not None:
        num = 0
        for k, row in stqdm(chars_df.iterrows(), "Adding boxes"):
            fig.add_shape(
                type="rect",
                x0=row.char_xmin,
                y0=row.char_ymin,
                x1=row.char_xmax,
                y1=row.char_ymax,
                line=dict(color=st.session_state["COLORS"][-1], width=1),
            )
            num += 1
    return fig


def save_to_zips(folder, pattern, savename):
    if os.path.exists(TEMP_FOLDER.joinpath(savename)):
        mode = "a"
    else:
        mode = "w"
    for idx, f in enumerate(folder.glob(pattern)):
        with zipfile.ZipFile(TEMP_FOLDER.joinpath(savename), mode=mode) as archive:
            archive.write(f)
        st.session_state["logger"].info(f"Written {f} to zip {TEMP_FOLDER.joinpath(savename)}")
        if idx == 1:
            mode = "a"
    st.session_state["logger"].info("Done zipping")


def process_multiple_asc(asc_files):
    algo_choice = st.session_state["algo_choice_multi"]
    if algo_choice is not None and "DIST" in algo_choice:
        model, model_cfg = find_and_load_model(model_date=st.session_state["DIST_MODEL_DATE_WITH_NORM"])
        model = st.session_state["single_DIST_model"]
        model_cfg = st.session_state["single_DIST_model_cfg"]
        st.session_state["logger"].info(f"process_multiple_asc loaded model")
    else:
        model, model_cfg = None, None
    zipfiles_with_results = []
    st.session_state["logger"].info(f"found asc_files {asc_files}")

    for asc_file in stqdm(asc_files, desc="Processing asc files"):
        st.session_state["logger"].info(f"processing asc_file {asc_file}")
        asc_file_stem = pl.Path(asc_file.name).stem
        trials_by_ids, lines = asc_to_trial_ids(asc_file)
        for trial_id, trial in stqdm(trials_by_ids.items(), desc=f"\nProcessing trials in {asc_file_stem}"):
            dffix, trial = process_trial(
                trial,
                asc_file_stem,
                lines,
                algo_choice,
                True,
            )

            st.session_state["logger"].debug(f"dffix.columns after process trial {dffix.columns}")
            if dffix.empty:
                st.session_state["logger"].warning(f"Dataframe for {trial_id} is empty, skipping")
                continue
            st.session_state["results"][f"{asc_file_stem}_{trial_id}"] = {
                "trial": trial,
                "dffix": dffix,
            }
            st.session_state["logger"].debug(f"Added {asc_file_stem}_{trial_id} to st.session_state")
            quick_dffix_save(dffix, TEMP_FOLDER.joinpath(f"{asc_file_stem}_{trial_id}.csv"))
            save_trial_to_json(trial, TEMP_FOLDER.joinpath(f"{asc_file_stem}_{trial_id}.json"))
            ut.plot_fixations_and_text(
                dffix,
                trial,
                save=True,
                savelocation=TEMP_FOLDER.joinpath(f"{asc_file_stem}_{trial_id}.png"),
                algo_choice=algo_choice,
                turn_axis_on=False,
            )
        if os.path.exists(TEMP_FOLDER.joinpath(f"{asc_file_stem}.zip")):
            os.remove(TEMP_FOLDER.joinpath(f"{asc_file_stem}.zip"))
        save_to_zips(TEMP_FOLDER, f"{asc_file_stem}*.csv", f"{asc_file_stem}.zip")
        save_to_zips(TEMP_FOLDER, f"{asc_file_stem}*.json", f"{asc_file_stem}.zip")
        save_to_zips(TEMP_FOLDER, f"{asc_file_stem}*.png", f"{asc_file_stem}.zip")
        zipfiles_with_results += [str(x) for x in TEMP_FOLDER.glob(f"{asc_file_stem}*.zip")]
    results_keys = list(st.session_state["results"].keys())
    st.session_state["logger"].debug(f"results_keys are {results_keys}")
    st.session_state["trial_choices_multi"] = results_keys
    st.session_state["zipfiles_with_results"] = zipfiles_with_results
    return (zipfiles_with_results, results_keys)


@st.cache_data
def get_trials_and_lines_from_asc_files(asc_files):
    list_of_trial_lists = []
    list_of_lines = []
    total_num_trials = 0

    asc_files_to_do = []
    for filename_full in asc_files:
        if hasattr(filename_full, "name") and not isinstance(filename_full, pl.Path):
            file = filename_full.name
            st.session_state["logger"].info(f"Filename is {file}, filename_full is {filename_full}")
        else:
            file = filename_full
        if not isinstance(file, str):
            file_stem = pl.Path(file.name).stem
        else:
            file_stem = pl.Path(file).stem
        savefolder = gradio_temp_unzipped_folder.joinpath(file_stem)
        st.session_state["logger"].info(f"Operating on file {file}")
        if ".zip" in file:
            with zipfile.ZipFile(filename_full, "r") as z:
                z.extractall(str(savefolder))
        elif ".tar" in file:
            shutil.unpack_archive(file, savefolder, "tar")
        elif ".asc" in file:
            asc_files_to_do.append(filename_full)
        else:
            st.session_state["logger"].warning(f"Unsopported file format found in files")
        newfiles = [str(x) for x in savefolder.glob(f"*.asc")]
        asc_files_to_do += newfiles
    st.session_state["logger"].info(f"asc_files_to_do is {asc_files_to_do}")

    for asc_file in asc_files_to_do:
        trials_by_ids, lines = asc_to_trial_ids(asc_file)
        total_num_trials += len(trials_by_ids)
        list_of_trial_lists.append(trials_by_ids)
        list_of_lines.append(lines)
    st.session_state["list_of_trial_lists"] = list_of_trial_lists
    st.session_state["list_of_lines"] = list_of_lines
    process_multiple_asc(st.session_state["multi_asc_filelist"])


def process_trial_choice_and_update_df_multi():
    trial_id = st.session_state["trial_id_multi"]
    dffix = st.session_state["results"][trial_id]["dffix"]
    if "start_time" in dffix.columns:
        dffix = dffix.drop(axis=1, labels=["start_time", "end_time"])
    st.session_state["dffix_multi"] = dffix
    st.session_state["trial_multi"] = st.session_state["results"][trial_id]["trial"]


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def make_trial_from_stimulus_df(
    stim_plot_df,
    filename,
    trial_id,
):
    chars_list = []
    words_list = []
    word_start_idx = 0
    for idx, row in stim_plot_df.reset_index().iterrows():
        char_dict = dict(
            char_xmin=row[st.session_state["x_start_col_name_fix_stim"]],
            char_xmax=row[st.session_state["x_end_col_name_fix_stim"]],
            char_ymin=row[st.session_state["y_start_col_name_fix_stim"]],
            char_ymax=row[st.session_state["y_end_col_name_fix_stim"]],
            char_x_center=row[st.session_state["x_col_name_fix_stim"]],
            char_y_center=row[st.session_state["y_col_name_fix_stim"]],
            char=row[st.session_state["char_col_name_fix_stim"]],
            assigned_line=int(row[st.session_state["line_num_col_name_stim"]]),
        )
        chars_list.append(char_dict)

        if len(chars_list) > 1 and (
            char_dict["char"] == " "
            or (len(chars_list) > 2 and (chars_list[-1]["char_xmin"] < chars_list[-2]["char_xmin"]))
        ):
            word_dict = dict(
                word_xmin=chars_list[word_start_idx]["char_xmin"],
                word_xmax=chars_list[-2]["char_xmax"],
                word_ymin=chars_list[word_start_idx]["char_ymin"],
                word_ymax=chars_list[word_start_idx]["char_ymax"],
                word_x_center=(chars_list[-2]["char_xmax"] - chars_list[word_start_idx]["char_xmin"]) / 2
                + chars_list[word_start_idx]["char_xmin"],
                word_y_center=(chars_list[word_start_idx]["char_ymax"] - chars_list[word_start_idx]["char_ymin"]) / 2
                + chars_list[word_start_idx]["char_ymin"],
                word="".join([chars_list[idx]["char"] for idx in range(word_start_idx, len(chars_list) - 1)]),
            )

            if char_dict["char"] != " ":
                word_start_idx = idx
            else:
                word_start_idx = idx + 1
            words_list.append(word_dict)

    line_heights = [x["char_ymax"] - x["char_ymin"] for x in chars_list]
    line_xcoords_all = [x["char_x_center"] for x in chars_list]
    line_xcoords_no_pad = np.unique(line_xcoords_all)

    line_ycoords_all = [x["char_y_center"] for x in chars_list]
    line_ycoords_no_pad = np.unique(line_ycoords_all)

    trial = dict(
        filename=filename,
        y_midline=[float(x) for x in list(stim_plot_df[st.session_state["y_col_name_fix_stim"]].unique())],
        num_char_lines=len(stim_plot_df[st.session_state["y_col_name_fix_stim"]].unique()),
        y_diff=[
            float(x) for x in list(np.unique(np.diff(stim_plot_df[st.session_state["y_start_col_name_fix_stim"]])))
        ],
        trial_id=trial_id,
        chars_list=chars_list,
        words_list=words_list,
        trial_is="paragraph",
        text="".join([x["char"] for x in chars_list]),
    )

    trial["x_char_unique"] = [float(x) for x in list(line_xcoords_no_pad)]
    trial["y_char_unique"] = list(map(float, list(line_ycoords_no_pad)))
    x_diff, y_diff = ut.calc_xdiff_ydiff(
        line_xcoords_no_pad, line_ycoords_no_pad, line_heights, allow_multiple_values=False
    )
    trial["x_diff"] = float(x_diff)
    trial["y_diff"] = float(y_diff)
    trial["num_char_lines"] = len(line_ycoords_no_pad)
    trial["line_heights"] = list(map(float, line_heights))
    trial["chars_list"] = chars_list

    return trial


@st.cache_data
def get_fixations_file_trials_list(fixations_df, stimulus):
    if isinstance(stimulus, pd.DataFrame):
        stimulus[st.session_state["line_num_col_name_stim"]] -= stimulus[
            st.session_state["line_num_col_name_stim"]
        ].min()
        stimulus.rename(
            {
                st.session_state["x_col_name_fix_stim"]: "char_x_center",
                st.session_state["x_start_col_name_fix_stim"]: "char_xmin",
                st.session_state["x_end_col_name_fix_stim"]: "char_xmax",
                st.session_state["y_col_name_fix_stim"]: "char_y_center",
                st.session_state["y_start_col_name_fix_stim"]: "char_ymin",
                st.session_state["y_end_col_name_fix_stim"]: "char_ymax",
                st.session_state["char_col_name_fix_stim"]: "char",
                st.session_state["trial_id_col_name_stim"]: "trial_id",
            },
            axis=1,
            inplace=True,
        )

    fixations_df.rename(
        mapper={
            st.session_state["x_col_name_fix"]: "x",
            st.session_state["y_col_name_fix"]: "y",
            st.session_state["time_start_col_name_fix"]: "corrected_start_time",
            st.session_state["time_stop_col_name_fix"]: "corrected_end_time",
            st.session_state["trial_id_col_name_fix"]: "trial_id",
        },
        axis=1,
        inplace=True,
    )

    fixations_df["duration"] = fixations_df.corrected_end_time - fixations_df.corrected_start_time
    if "trial_id" in stimulus:
        fixations_df["trial_id"] = stimulus["trial_id"]
    if "trial_id" in fixations_df:
        if st.session_state["has_multiple_subject"]:
            fixations_df["trial_id"] = [
                f"{id}_{num}"
                for id, num in zip(
                    fixations_df[st.session_state["subject_col_name_fix"]],
                    fixations_df[st.session_state["trial_id_col_name_fix"]],
                )
            ]
        trial_keys = list(fixations_df[st.session_state["trial_id_col_name_fix"]].unique())
        st.session_state["logger"].info(f"Found keys {trial_keys} for {st.session_state['single_csv_file'].name}")
    else:
        st.session_state["logger"].warning(f"trial id column not found assigning trial id trial_0.")
        st.warning(f"trial id column not found assigning trial id trial_0.")
        fixations_df["trial_id"] = "trial_0"
    st.session_state["fixations_df"] = fixations_df
    trials_by_ids = {}

    for trial_id, subdf in fixations_df.groupby("trial_id"):
        if isinstance(stimulus, pd.DataFrame):
            stim_df = stimulus[stimulus.trial_id == trial_id]

            stim_df = stim_df.dropna(axis=0, how="any")
            subdf = subdf.dropna(axis=0, how="any")
            subdf = subdf.reset_index(drop=True)
            stim_df = stim_df.reset_index(drop=True)
            assert not stim_df.empty, "stimulus df is empty"
            trial = make_trial_from_stimulus_df(
                stim_df,
                st.session_state["single_csv_file_stim"].name,
                trial_id,
            )
        else:
            trial = stimulus
        trial["dffix"] = subdf
        trial["fname"] = f"{trial_id}"
        trial["plot_file"] = str(
            st.session_state["PLOTS_FOLDER"].joinpath(f"{trial_id}_2ndInput_chars_channel_sep.png")
        )
        trials_by_ids[trial_id] = trial

    return trials_by_ids, trial_keys


def try_reading_csv(file):
    stringio = StringIO(file.getvalue().decode("utf-8"))
    colname_mapping = {}
    try:
        df = pd.read_csv(stringio)
        st.session_state["logger"].info(f"\n{df.head()}")
        col_list = df.columns
        assert len(col_list) > 1
        return df
    except Exception as e:
        st.session_state["logger"].warn(e)
        try:
            df = pd.read_csv(StringIO(file.getvalue().decode("utf-8")), delimiter="\t")
            col_list = df.columns
            assert len(col_list) > 1
            return df
        except Exception as e:
            st.session_state["logger"].warn(e)
            return None


@st.cache_data
def guess_col_names_fix(file=None):
    if file is None:
        file = st.session_state["single_csv_file"]
    if file is None:
        return None

    first_line = next(iter(StringIO(file.getvalue().decode("utf-8"))))
    res = re.findall(r"[^()0-9-]+", first_line)
    for delim in [",", "\t", ";"]:
        first_line = first_line.split(delim)
        if len(first_line) > 2:
            break
        else:
            first_line = first_line[0]
    scores_lists = {}
    for k, v in st.session_state["colnames_custom_csv_fix"].items():
        scores_lists[v] = []
        for word in first_line:
            scores_lists[v].append(jf.levenshtein_distance(v, word))
    scores_df = pd.DataFrame(scores_lists)
    scores_df.idxmin(axis=0)
    df = try_reading_csv(file)
    if df.shape[1] > 1:
        return df
    else:
        return None


@st.cache_data
def guess_col_names_stim(file=None):
    if file is None:
        file = st.session_state["single_csv_file_stim"]
    if file is None:
        return None
    if ".json" in file.name:
        json_string = file.getvalue().decode("utf-8")
        trial = json.loads(json_string)
        return trial
    else:
        df = try_reading_csv(file)

        if df.shape[1] > 1:
            return df
        else:
            return None


@st.cache_resource
def set_up_models(dist_models_folder):
    return ut.set_up_models(dist_models_folder)

@st.cache_data
def get_eyekit_measures(_txt, _seq, get_char_measures=False):
    return ekm.get_eyekit_measures(_txt, _seq, get_char_measures=get_char_measures)


@st.cache_data
def get_all_measures(trial, dffix, prefix, use_corrected_fixations=True, correction_algo="warp"):
    return ut.get_all_measures(trial, dffix, prefix, use_corrected_fixations=use_corrected_fixations, correction_algo=correction_algo)


assert "ALGO_CHOICES" in st.session_state, f"st.session_state not initialized\n{list(st.session_state.keys())}"

set_up_models_out = set_up_models(DIST_MODELS_FOLDER)
st.session_state.update(set_up_models_out)


st.title("Fixation data vertical alignment")
st.header("👀 Read asc file or files and plot fixations 👀")
st.markdown("[Contact Us](mailto:pvwork13+hgf@gmail.com)")
st.markdown("[Read about DIST model](https://arxiv.org/abs/2311.06095)")

single_file_tab, multi_file_tab = st.tabs(["Single File 📁", "Multiple Files 📁 📁"])

single_file_tab_asc_tab, single_file_tab_csv_tab = single_file_tab.tabs([".asc files", "custom files"])

single_file_tab_asc_tab.subheader(
    "Upload an .asc file and select a trial. Then select a correction algorithm and plot/download the results"
)


def change_which_file_is_used_and_clear_results():
    if "dffix" in st.session_state:
        del st.session_state["dffix"]
    if "trial" in st.session_state:
        del st.session_state["trial"]
    if st.session_state["single_file_tab_asc_tab_example_use_example_or_uploaded_file_choice"] == "Example File":
        st.session_state["single_asc_file_asc"] = st.session_state["single_file_tab_asc_tab_example_file_choice"]
    else:
        st.session_state["single_asc_file_asc"] = st.session_state["single_asc_uploaded_file"]


with single_file_tab_asc_tab.form("single_file_tab_asc_tab_load_example_form"):
    single_asc_file_asc_uploaded = st.file_uploader(
        "Select .asc File", accept_multiple_files=False, key="single_asc_uploaded_file", type=["asc"]
    )
    close_gap_between_words_single_asc = st.checkbox(
        label="Should spaces between words be included in word bounding box?",
        value=False,
        key="close_gap_between_words_single_asc",
    )

    if os.path.isfile(EXAMPLE_ASC_FILES[0]):
        example_file_choice = st.selectbox(
            "Select example file", options=EXAMPLE_ASC_FILES, key="single_file_tab_asc_tab_example_file_choice"
        )
        use_example_or_uploaded_file_choice = st.radio(
            "Should the uploaded file be used or the selected example file?",
            index=1,
            options=["Uploaded File", "Example File"],
            key="single_file_tab_asc_tab_example_use_example_or_uploaded_file_choice",
        )

    upload_file_button = st.form_submit_button(
        label="Load selected data.", on_click=change_which_file_is_used_and_clear_results
    )


if "single_asc_file_asc" in st.session_state and st.session_state["single_asc_file_asc"] is not None:
    trial_choices_single_asc, trials_by_ids, lines, asc_file = get_trials_list(
        st.session_state["single_asc_file_asc"], close_gap_between_words=close_gap_between_words_single_asc
    )
    st.session_state["trials_by_ids"] = trials_by_ids
    st.session_state["trial_choices"] = trial_choices_single_asc
    st.session_state["lines"] = lines
    st.session_state["asc_file"] = asc_file
    if trial_choices_single_asc:
        with single_file_tab_asc_tab.form(key="single_file_tab_asc_tab_trial_select_form"):
            col_a1, col_a2 = st.columns((1, 2))
            with col_a1:
                trial_choice = st.selectbox(
                    "Which trial should be corrected?",
                    trial_choices_single_asc,
                    key="trial_id",
                    index=0,
                )
            with col_a2:
                st.multiselect(
                    "Choose correction algorithm",
                    ALGO_CHOICES,
                    key="algo_choice",
                    default=[ALGO_CHOICES[0]],
                )
            process_trial_btn = st.form_submit_button("Load and correct trial")

        if process_trial_btn:
            single_file_tab_asc_tab.write(f'You selected: {st.session_state["trial_id"]}')
            dffix, trial, dpi, screen_res, font, font_size = process_trial_choice(
                trial_choice, st.session_state["algo_choice"]
            )

            st.session_state["dffix"] = dffix
            st.session_state["trial"] = trial
            st.session_state["dpi"] = dpi
            st.session_state["screen_res"] = screen_res
            st.session_state["font"] = font
            st.session_state["font_size"] = font_size

            export_csv(dffix, trial)

        if "dffix" in st.session_state and "trial" in st.session_state:
            df_expander_single = single_file_tab_asc_tab.expander("Show Dataframe", False)
            plot_expander_single = single_file_tab_asc_tab.expander("Show Plots", True)
            df_expander_single.dataframe(st.session_state["dffix"])

            csv = convert_df(st.session_state["dffix"])

            df_expander_single.download_button(
                "Download fixation dataframe",
                csv,
                f'{st.session_state["trial_id"]}.csv',
                "text/csv",
                key="download-csv-single",
            )

            plotting_checkboxes_single = plot_expander_single.multiselect(
                "Select what gets plotted",
                ["Uncorrected Fixations", "Corrected Fixations", "Words", "Word boxes"],
                key="plotting_checkboxes_single",
                default=["Uncorrected Fixations", "Corrected Fixations", "Words", "Word boxes"],
            )
            scale_factor_single_asc = plot_expander_single.number_input(
                label="Scale factor for stimulus image", min_value=0.01, max_value=3.0, value=0.5, step=0.1
            )
            plot_expander_single.plotly_chart(
                plotly_plot_with_image(
                    st.session_state["dffix"],
                    st.session_state["trial"],
                    to_plot_list=plotting_checkboxes_single,
                    algo_choice=st.session_state["algo_choice"],
                    scale_factor=scale_factor_single_asc,
                ),
                use_container_width=False,
            )
            plot_expander_single.plotly_chart(
                plot_y_corr(st.session_state["dffix"], st.session_state["algo_choice"]), use_container_width=True
            )

            if "chars_list" in st.session_state["trial"]:
                analysis_expander_single_asc = single_file_tab_asc_tab.expander("Show Analysis results", True)
                use_corrected_fixations_tickbox = analysis_expander_single_asc.checkbox(
                    "Use corrected",
                    True,
                    "use_corrected_fixations_tickbox",
                    help="Whether to use the corrected or uncorrected fixations for the analysis.",
                )
                eyekit_tab, own_analysis_tab = analysis_expander_single_asc.tabs(
                    ["Analysis using eyekit", "Analysis without eyekit"]
                )
                with eyekit_tab:
                    st.markdown("Analysis powered by [eyekit](https://jwcarr.github.io/eyekit/)")
                    st.markdown(
                        "Please adjust parameters below to align fixations with stimulus using the sliders.Eyekit analysis is based on this alignment."
                    )
                    a_c1, a_c2, a_c3, a_c4, a_c5, a_c6 = st.columns(6)
                    if "Consolas" in AVAILABLE_FONTS:
                        font_index = AVAILABLE_FONTS.index("Consolas")
                    elif "Courier New" in AVAILABLE_FONTS:
                        font_index = AVAILABLE_FONTS.index("Courier New")
                    elif "DejaVu Sans Mono" in AVAILABLE_FONTS:
                        font_index = AVAILABLE_FONTS.index("DejaVu Sans Mono")
                    else:
                        font_index = 0
                    font_face = a_c1.selectbox(
                        label="Select Font",
                        options=AVAILABLE_FONTS,
                        index=font_index,
                        key="font_face_for_eyekit_single_asc",
                    )
                    algo_choice_single_asc_eyekit = a_c1.selectbox(
                        "Algorithm", st.session_state["algo_choice"], index=0, key="algo_choice_single_asc_eyekit"
                    )
                    sliders_on_tickbox = a_c6.checkbox(
                        "Sliders", True, "single_asc_eyekit_sliders_checkbox", help="Turns sliders on and off"
                    )

                    if "font_size_for_eyekit" not in st.session_state:
                        (
                            y_diff,
                            x_txt_start,
                            y_txt_start,
                            _,
                            _,
                            line_height,
                        ) = add_default_font_and_character_props_to_state(st.session_state["trial"])
                        font_size = ut.set_font_from_chars_list(st.session_state["trial"])
                        st.session_state["y_diff_for_eyekit"] = y_diff
                        st.session_state["x_txt_start_for_eyekit"] = x_txt_start
                        st.session_state["y_txt_start_for_eyekit"] = y_txt_start
                        st.session_state["font_face_for_eyekit"] = font_face
                        st.session_state["font_size_for_eyekit"] = font_size
                        st.session_state["line_height_for_eyekit"] = line_height
                    if sliders_on_tickbox:
                        font_size = a_c2.select_slider(
                            "Font Size",
                            np.arange(5, 36, 0.25),
                            st.session_state["font_size_for_eyekit"],
                            key="font_size_for_eyekit_single_asc",
                        )
                        x_txt_start = a_c3.select_slider(
                            "x",
                            np.arange(300, 601, 1),
                            round(st.session_state["x_txt_start_for_eyekit"]),
                            key="x_txt_start_for_eyekit_single_asc",
                            help="x coordinate of first character",
                        )
                        y_txt_start = a_c4.select_slider(
                            "y",
                            np.arange(100, 501, 1),
                            round(st.session_state["y_txt_start_for_eyekit"]),
                            key="y_txt_start_for_eyekit_single_asc",
                            help="y coordinate of first character",
                        )
                        line_height = a_c5.select_slider(
                            "Line height",
                            np.arange(0, 151, 1),
                            round(st.session_state["line_height_for_eyekit"]),
                            key="line_height_for_eyekit_single_asc",
                        )
                    else:
                        font_size = a_c2.number_input(
                            "Font Size",
                            None,
                            None,
                            round(st.session_state["font_size_for_eyekit"], ndigits=0),
                            key="font_size_for_eyekit_single_asc",
                        )
                        x_txt_start = a_c3.number_input(
                            "x",
                            None,
                            None,
                            round(st.session_state["x_txt_start_for_eyekit"]),
                            key="x_txt_start_for_eyekit_single_asc",
                            help="x coordinate of first character",
                        )
                        y_txt_start = a_c4.number_input(
                            "y",
                            None,
                            None,
                            round(st.session_state["y_txt_start_for_eyekit"]),
                            key="y_txt_start_for_eyekit_single_asc",
                            help="y coordinate of first character",
                        )
                        line_height = a_c5.number_input(
                            "Line height",
                            None,
                            None,
                            round(st.session_state["line_height_for_eyekit"]),
                            key="line_height_for_eyekit_single_asc",
                        )

                    fixation_sequence, textblock, screen_size = ekm.get_fix_seq_and_text_block(
                        st.session_state["dffix"],
                        st.session_state["trial"],
                        x_txt_start=st.session_state["x_txt_start_for_eyekit_single_asc"],
                        y_txt_start=st.session_state["y_txt_start_for_eyekit_single_asc"],
                        font_face=st.session_state["font_face_for_eyekit_single_asc"],
                        font_size=st.session_state["font_size_for_eyekit_single_asc"],
                        line_height=line_height,
                        use_corrected_fixations=st.session_state["use_corrected_fixations_tickbox"],
                        correction_algo=st.session_state["algo_choice_single_asc_eyekit"],
                    )
                    eyekitplot_img = ekm.eyekit_plot(textblock, fixation_sequence, screen_size)
                    st.image(eyekitplot_img, "Fixations and stimulus as used for anaylsis")

                    with open(
                        f'results/fixation_sequence_eyekit_{st.session_state["trial"]["trial_id"]}.json', "r"
                    ) as f:
                        fixation_sequence_json = json.load(f)
                    fixation_sequence_json_str = json.dumps(fixation_sequence_json)

                    st.download_button(
                        "Download fixations in eyekits format",
                        fixation_sequence_json_str,
                        f'fixation_sequence_eyekit_{st.session_state["trial"]["trial_id"]}.json',
                        "json",
                        key="download_eyekit_fix_json_single_asc",
                    )

                    with open(f'results/textblock_eyekit_{st.session_state["trial"]["trial_id"]}.json', "r") as f:
                        textblock_json = json.load(f)
                    textblock_json_str = json.dumps(textblock_json)

                    st.download_button(
                        "Download stimulus in eyekits format",
                        textblock_json_str,
                        f'textblock_eyekit_{st.session_state["trial"]["trial_id"]}.json',
                        "json",
                        key="download_eyekit_text_json_single_asc",
                    )

                    word_measures_df, character_measures_df = get_eyekit_measures(
                        textblock, fixation_sequence, get_char_measures=False
                    )

                    st.dataframe(word_measures_df, use_container_width=True, hide_index=True)
                    word_measures_df_csv = convert_df(word_measures_df)

                    word_measures_df_download_btn = st.download_button(
                        "Download word measures data",
                        word_measures_df_csv,
                        f'{st.session_state["trial"]["trial_id"]}_word_measures_df.csv',
                        "text/csv",
                        key="word_measures_df_download_btn",
                    )
                    measure_words = st.selectbox(
                        "Select measure to visualize", list(ekm.MEASURES_DICT.keys()), key="measure_words"
                    )
                    st.image(ekm.plot_with_measure(textblock, fixation_sequence, screen_size, measure_words))
                with own_analysis_tab:
                    st.markdown(
                        "This analysis method does not require manual alignment and works when the automated stimulus coordinates are correct."
                    )
                    own_word_measures = get_all_measures(
                        st.session_state["trial"],
                        st.session_state["dffix"],
                        prefix="word",
                        use_corrected_fixations=st.session_state["use_corrected_fixations_tickbox"],
                        correction_algo=st.session_state["algo_choice_single_asc_eyekit"],
                    )
                    st.dataframe(own_word_measures, use_container_width=True, hide_index=True)
                    own_word_measures_csv = convert_df(own_word_measures)

                    word_measures_df_download_btn = st.download_button(
                        "Download word measures data",
                        own_word_measures_csv,
                        f'{st.session_state["trial"]["trial_id"]}_own_word_measures_df.csv',
                        "text/csv",
                        key="own_word_measures_df_download_btn",
                    )
                    fix_to_plot = (
                        ["Corrected Fixations"]
                        if st.session_state["use_corrected_fixations_tickbox"]
                        else ["Uncorrected Fixations"]
                    )
                    own_word_measures_fig, desired_width_in_pixels, desired_height_in_pixels = matplotlib_plot_df(
                        st.session_state["dffix"],
                        st.session_state["trial"],
                        st.session_state["algo_choice"],
                        stimulus_prefix="word",
                        box_annotations=own_word_measures[measure_words],
                        fix_to_plot=fix_to_plot,
                    )
                    st.pyplot(own_word_measures_fig)
            else:
                single_file_tab_asc_tab.warning("🚨 Stimulus information needed for analysis 🚨")

single_file_tab_csv_tab.markdown(
    "#### Upload one .csv file for the fixations and one .json or .csv file for the stimulus information and select a trial. Then select a correction algorithm and plot/download the results"
)

with single_file_tab_csv_tab.expander("Upload and preview data", expanded=True):
    csv_upl_col1, csv_upl_col2 = st.columns(2)
    single_csv_file = csv_upl_col1.file_uploader(
        "Select .csv file containing the fixation data",
        accept_multiple_files=False,
        key="single_csv_file",
        type={"csv", "txt", "dat"},
    )
    single_csv_stim_file = csv_upl_col2.file_uploader(
        "Select .csv or .json file containing the stimulus data",
        accept_multiple_files=False,
        key="single_csv_file_stim",
        type={"json", "csv", "txt", "dat"},
    )

    if single_csv_file:
        st.session_state["dffix_single_csv"] = guess_col_names_fix(single_csv_file)
        if st.session_state["dffix_single_csv"] is not None:
            csv_upl_col1.dataframe(
                st.session_state["dffix_single_csv"], use_container_width=True, hide_index=True, height=200
            )
    if single_csv_stim_file:
        st.session_state["stimdf_single_csv"] = guess_col_names_stim(single_csv_stim_file)
        if ".json" in single_csv_stim_file.name:
            st.session_state["colnames_stim"] = st.session_state["stimdf_single_csv"].keys()
        else:
            st.session_state["colnames_stim"] = st.session_state["stimdf_single_csv"].columns
        if st.session_state["stimdf_single_csv"] is not None:
            if ".json" in single_csv_stim_file.name:
                csv_upl_col2.json(st.session_state["stimdf_single_csv"])
            else:
                csv_upl_col2.dataframe(
                    st.session_state["stimdf_single_csv"], use_container_width=True, hide_index=True, height=200
                )

if single_csv_file and single_csv_stim_file:
    with single_file_tab_csv_tab.expander("Column names for csv files", expanded=True):
        with st.form("Column names for csv files"):
            st.markdown("### Please set column/key names for csv/json files")
            st.markdown("#### Fixation file column names:")
            c1, c2, c3 = st.columns(3)
            x_col_name_fix = c1.text_input("x coordinate", key="x_col_name_fix", value="x")
            y_col_name_fix = c2.text_input("y coordinate", key="y_col_name_fix", value="y")
            subject_col_name_fix = c1.text_input("subject id", key="subject_col_name_fix", value="sub_id")
            trial_id_col_name_fix = c3.text_input("trial id", key="trial_id_col_name_fix", value="trial_id")
            time_start_col_name_fix = c2.text_input(
                "fixation start time", key="time_start_col_name_fix", value="corrected_start_time"
            )
            time_stop_col_name_fix = c3.text_input(
                "fixation end time", key="time_stop_col_name_fix", value="corrected_end_time"
            )
            st.markdown("#### Stimulus file column/key names:")
            c1, c2, c3 = st.columns(3)
            x_col_name_fix_stim = c1.text_input("x coordinate", key="x_col_name_fix_stim", value="char_x_center")
            y_col_name_fix_stim = c2.text_input("y coordinate", key="y_col_name_fix_stim", value="char_y_center")
            x_start_col_name_fix_stim = c3.text_input(
                "x min of interest areas", key="x_start_col_name_fix_stim", value="char_xmin"
            )
            x_end_col_name_fix_stim = c1.text_input(
                "x max of interest areas", key="x_end_col_name_fix_stim", value="char_xmax"
            )
            y_start_col_name_fix_stim = c2.text_input(
                "y min of interest areas", key="y_start_col_name_fix_stim", value="char_ymin"
            )
            y_end_col_name_fix_stim = c3.text_input(
                "x max of interest areas", key="y_end_col_name_fix_stim", value="char_ymax"
            )
            char_col_name_fix_stim = c1.text_input(
                "content of interest area", key="char_col_name_fix_stim", value="char"
            )
            line_num_col_name_stim = c3.text_input(
                "line number for interest areas", key="line_num_col_name_stim", value="assigned_line"
            )
            subject_col_name_stim = c1.text_input("subject id", key="subject_col_name_stim", value="sub_id")
            trial_id_col_name_stim = c2.text_input("trial id", key="trial_id_col_name_stim", value="trial_id")
            has_multiple_subject = c2.checkbox("multiple subject in file", key="has_multiple_subject")
            form_submitted = st.form_submit_button("Confirm column/key names")


if single_csv_file and single_csv_stim_file:
    process_custom_csvs_button = single_file_tab_csv_tab.button(
        "Load data from files",
    )
    if process_custom_csvs_button or "trial_keys_single_csv" in st.session_state:
        trials_by_ids, trial_keys = get_fixations_file_trials_list(
            st.session_state["dffix_single_csv"], st.session_state["stimdf_single_csv"]
        )

        st.session_state["trials_by_ids_single_csv"] = trials_by_ids
        st.session_state["trial_keys_single_csv"] = trial_keys
        with single_file_tab_csv_tab.form(key="trial_selection_algo_selection_form_single_csv"):
            col_a1, col_a2 = st.columns((1, 2))
            with col_a1:
                trial_choice = st.selectbox(
                    "Which trial should be corrected?",
                    st.session_state["trial_keys_single_csv"],
                    key="trial_id_selected_custom_csv",
                    index=0,
                )
            with col_a2:
                algo_choice_single_csv = st.multiselect(
                    "Choose correction algorithm",
                    ALGO_CHOICES,
                    key="algo_choice_single_csv",
                    default=[ALGO_CHOICES[0]],
                )
            process_trial_btn = st.form_submit_button("Correct trial")
        if "trial_id_selected_custom_csv" in st.session_state and "algo_choice_single_csv" in st.session_state:
            trial = st.session_state["trials_by_ids_single_csv"][trial_choice]
            dffix, trial, dpi, screen_res, font, font_size = process_trial_choice_single_csv(
                trial, algo_choice_single_csv
            )
            st.session_state["trial_single_csv"] = trial
            csv = convert_df(dffix)

            single_file_tab_csv_tab.download_button(
                "Download corrected fixation data",
                csv,
                f'{trial["trial_id"]}.csv',
                "text/csv",
                key="download-csv-custom-csv",
            )
            with single_file_tab_csv_tab.expander("Show corrected fixation data", expanded=True):
                st.dataframe(dffix, use_container_width=True, hide_index=True, height=200)
            with single_file_tab_csv_tab.expander("Show fixation plots", expanded=True):
                plotting_checkboxes_single_single_csv = st.multiselect(
                    "Select what gets plotted",
                    ["Uncorrected Fixations", "Corrected Fixations", "Words", "Word boxes"],
                    key="plotting_checkboxes_single_single_csv",
                    default=["Uncorrected Fixations", "Corrected Fixations", "Words", "Word boxes"],
                )

                st.plotly_chart(
                    plotly_plot_with_image(
                        dffix,
                        trial,
                        to_plot_list=plotting_checkboxes_single_single_csv,
                        algo_choice=algo_choice_single_csv,
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(plot_y_corr(dffix, algo_choice_single_csv), use_container_width=True)


multi_file_tab.subheader("Upload multiple .asc files. Then select a correction algorithm and download the results.")

with multi_file_tab.form("Upload files to be processed and select algorithm"):
    multifile_col, multi_algo_col = st.columns((1, 1))

    with multifile_col:
        st.file_uploader(
            "Upload .asc Files", accept_multiple_files=True, key="multi_asc_filelist", type=["asc", "tar", "zip"]
        )
    with multi_algo_col:
        st.multiselect(
            "Choose correction algorithms",
            ALGO_CHOICES,
            key="algo_choice_multi",
            default=[ALGO_CHOICES[0]],
        )
    process_trial_btn_multi = st.form_submit_button("Load and correct asc files")
    if process_trial_btn_multi:
        get_trials_and_lines_from_asc_files(st.session_state["multi_asc_filelist"])

if "zipfiles_with_results" in st.session_state:
    multi_res_col1, multi_res_col2 = multi_file_tab.columns(2)

    chosen_zip = multi_res_col1.selectbox("Choose results to download", st.session_state["zipfiles_with_results"])
    st.session_state["logger"].info(f"Download button for {chosen_zip}")
    st.session_state["logger"].info(st.session_state["zipfiles_with_results"])
    zipnamestem = pl.Path(chosen_zip).stem
    with open(chosen_zip, "rb") as f:
        multi_res_col2.download_button(f"Download {zipnamestem}", f, file_name=f"results_{zipnamestem}.zip")


if "trial_choices_multi" in st.session_state:
    multi_plotting_options_col1, multi_plotting_options_col2 = multi_file_tab.columns(2)

    trial_choice_multi = multi_plotting_options_col1.selectbox(
        "Which trial should be plotted?",
        st.session_state["trial_choices_multi"],
        key="trial_id_multi",
        placeholder="Select trial to display and plot",
        on_change=process_trial_choice_and_update_df_multi,
    )

    plotting_checkboxes_multi = multi_plotting_options_col2.multiselect(
        "Select what gets plotted",
        ["Uncorrected Fixations", "Corrected Fixations", "Words", "Word boxes"],
        default=["Uncorrected Fixations", "Corrected Fixations", "Words", "Word boxes"],
    )

    if trial_choice_multi and "dffix_multi" in st.session_state:
        df_expander_multi = multi_file_tab.expander("Show Dataframe", False)
        plot_expander_multi = multi_file_tab.expander("Show Plots", True)

        df_expander_multi.dataframe(st.session_state["dffix_multi"])
        dffix_multi = st.session_state["dffix_multi"]
        trial_multi = st.session_state["trial_multi"]

        plot_expander_multi.plotly_chart(
            plotly_plot_with_image(
                dffix_multi, trial_multi, st.session_state["algo_choice_multi"], to_plot_list=plotting_checkboxes_multi
            ),
            use_container_width=True,
        )
        plot_expander_multi.plotly_chart(
            plot_y_corr(dffix_multi, st.session_state["algo_choice_multi"]), use_container_width=True
        )

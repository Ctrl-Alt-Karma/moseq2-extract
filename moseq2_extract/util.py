"""
General utility functions throughout the extract package.
"""
import os
import re
import cv2
import math
import json
import h5py
import click
import tarfile
import warnings
import datetime
import subprocess
import numpy as np
from glob import glob
from copy import deepcopy
import ruamel.yaml as yaml
from typing import Pattern
from cytoolz import valmap
from moseq2_extract.io.image import write_image
from moseq2_extract.io.video import get_movie_info
from os.path import join, exists, splitext, basename, abspath, dirname


# Semantic version tags for the behaviours whose OUTPUT changed relative to
# the upstream 2021 code. Bump the relevant tag whenever a change alters the
# numbers written to disk, so a file records which policy produced it and data
# from before and after a fix can be told apart.
EXTRACT_OUTPUT_POLICIES = {
    "scalar_px_to_mm": "pinhole",          # was small-angle approximation
    "area_units": "mm2-xy-product",        # was linear mm/px factor
    "background_dropout": "masked",        # dropouts excluded from bg median
    "roi_plane_fit": "lstsq-inlier-refit",  # was 3-point RANSAC hypothesis
    "hampel_filter": "mad-threshold-both-axes",
    "depth_range_detection": "histogram-mode",
    "velocity_3d_px": "invalid-mixed-units-omitted",
}


def _package_git_sha(package_file):
    """Best-effort git commit of an installed package (editable installs)."""
    try:
        pkg_dir = dirname(abspath(package_file))
        return (
            subprocess.check_output(
                ["git", "-C", pkg_dir, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_pipeline_provenance():
    """
    Build a provenance record describing the code that produced an output.

    Returns:
    provenance (dict): package version, git commit (best effort), write time,
        and the semantic output-policy tags in EXTRACT_OUTPUT_POLICIES.
    """

    try:
        from moseq2_extract import __version__ as version
    except Exception:
        version = "unknown"

    return {
        "package": "moseq2-extract",
        "version": version,
        "git_sha": _package_git_sha(__file__),
        "written": datetime.datetime.now().isoformat(),
        "policies": dict(EXTRACT_OUTPUT_POLICIES),
    }


def write_pipeline_provenance(h5_file, path="metadata/extraction/pipeline"):
    """
    Write the provenance record as a JSON string dataset into an open h5 file.

    Args:
    h5_file (h5py.File): open, writable h5 file.
    path (str): dataset path to write the JSON provenance to.
    """

    provenance = json.dumps(get_pipeline_provenance())
    if path in h5_file:
        del h5_file[path]
    h5_file.create_dataset(path, data=np.string_(provenance))
    h5_file[path].attrs["description"] = (
        "JSON provenance: pipeline version, git commit and output-policy tags "
        "identifying the code that produced this file"
    )


def filter_warnings(func):
    """
    Applies warnings.simplefilter() to ignore warnings when
     running the main gui functionaity in a Jupyter Notebook.
     The function will filter out: yaml.error.UnsafeLoaderWarning, FutureWarning and UserWarning.

    Args:
    func (function): function to silence enclosed warnings.

    Returns:
    apply_warning_filters (func): Returns passed function after warnings filtering is completed.
    """
    def apply_warning_filters(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
            warnings.simplefilter(action='ignore', category=FutureWarning)
            warnings.simplefilter(action='ignore', category=UserWarning)
            return func(*args, **kwargs)
    return apply_warning_filters


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name):
    """
    Override default CLI variables with the values contained within the config.yaml being passed.

    Args:
    config_file_param_name (str): path to config file.

    Returns:
    custom_command_class (function): decorator function to update click.Command parameters with the config_file
    parameter values.
    """

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            # grab the config file
            config_file = ctx.params[config_file_param_name]
            param_defaults = {p.human_readable_name: p.default for p in self.params
                              if isinstance(p, click.core.Option)}
            param_defaults = {k: tuple(v) if type(v) is list else v for k, v in param_defaults.items()}
            param_cli = {k: tuple(v) if type(v) is list else v for k, v in ctx.params.items()}

            if config_file is not None:

                config_data = read_yaml(config_file)
                # set config_data['output_file'] ['output_dir'] ['input_dir'] to None to avoid overwriting previous files
                # assuming users would either input their own paths or use the default path
                config_data['input_dir'] = None
                config_data['output_dir'] = None
                config_data['output_file'] = None

                # modified to only use keys that are actually defined in options and the value is not not none
                config_data = {k: tuple(v) if isinstance(v, yaml.comments.CommentedSeq) else v
                               for k, v in config_data.items() if k in param_defaults.keys() and v is not None}

                # find differences btw config and param defaults
                diffs = set(param_defaults.items()) ^ set(param_cli.items())

                # combine defaults w/ config data
                combined = {**param_defaults, **config_data}

                # update cli params that are non-default
                keys = [d[0] for d in diffs]
                for k in set(keys):
                    combined[k] = ctx.params[k]

                ctx.params = combined
                
                # add new parameters to the original config file
                config_data = read_yaml(config_file)
                
                # remove flags from combined so the flag values in config.yaml won't get overwritten
                flag_list = ['manual_set_depth_range', 'use_plane_bground', 'progress_bar', 'delete', 'compute_raw_scalars', 'skip_completed', 'skip_checks', 'get_cmd', 'run_cmd']
                combined = {k:v for k, v in combined.items() if k not in flag_list}
                # combine original config data and the combined params prioritizing the combined
                config_data = {**config_data, **combined}
                # with open(config_file, 'w') as f:
                #     yaml.safe_dump(config_data, f)

            return super().invoke(ctx)

    return custom_command_class

def set_bground_to_plane_fit(bground_im, plane, output_dir):
    """
    Replaces median-computed background image with plane fit.
    Only occurs if config_data['use_plane_bground'] == True.

    Args:
    bground_im (numpy.ndarray): Background image computed via median value in each pixel of depth video.
    plane (numpy.ndarray): Computed ROI Plane using RANSAC.
    output_dir (str): Path to write updated background image to.

    Returns:
    bground_im (numpy.ndarray): The background image.
    """

    xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
    coords = np.vstack((xx.ravel(), yy.ravel()))

    plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
    plane_im = plane_im.reshape(bground_im.shape)

    write_image(join(output_dir, 'bground.tiff'), plane_im, scale=True)

    return plane_im

def get_frame_range_indices(trim_beginning, trim_ending, nframes):
    """
    Compute the total number of frames to be extracted, and find the start and end indices.

    Args:
    trim_beginning (int): number of frames to remove from beginning of recording
    trim_ending (int): number of frames to remove from ending of recording
    nframes (int): total number of requested frames to extract

    Returns:
    nframes (int): total number of frames to extract
    first_frame_idx (int): index of the frame to begin extraction from
    last_frame_idx (int): index of the last frame in the extraction
    """
    assert all((trim_ending >= 0, trim_beginning >= 0)) , "frame_trim arguments must be greater than or equal to 0!"

    first_frame_idx = 0
    if trim_beginning > 0 and trim_beginning < nframes:
        first_frame_idx = trim_beginning

    last_frame_idx = nframes
    if first_frame_idx < (nframes - trim_ending) and trim_ending > 0:
        last_frame_idx = nframes - trim_ending

    total_frames = last_frame_idx - first_frame_idx

    return total_frames, first_frame_idx, last_frame_idx

def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    """
    Generates batches used to chunk videos prior to extraction.

    Args:
    nframes (int): total number of frames
    chunk_size (int): the number of desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns:
    out (list): the list of batches
    """

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out

def load_timestamps(timestamp_file, col=0, alternate=False):
    """
    Read timestamps from space delimited text file for timestamps.

    Args:
    timestamp_file (str): path to timestamp file
    col (int): column in ts file read.
    alternate (boolean): specified if timestamps were saved in a csv file. False means txt file and True means csv file.

    Returns:
    ts (1D array): list of timestamps
    """

    ts = []
    try:
        with open(timestamp_file, 'r') as f:
            for line in f:
                cols = line.split()
                ts.append(float(cols[col]))
        ts = np.array(ts)
    except TypeError as e:
        # try iterating directly
        for line in timestamp_file:
            cols = line.split()
            ts.append(float(cols[col]))
        ts = np.array(ts)
    except FileNotFoundError as e:
        ts = None
        warnings.warn('Timestamp file was not found! Make sure the timestamp file exists is named '
            '"depth_ts.txt" or "timestamps.csv".')
        warnings.warn('This could cause issues for large number of dropped frames during the PCA step while '
            'imputing missing data.')

    # if timestamps were saved in a csv file
    if alternate:
        ts = ts * 1000

    return ts

def detect_avi_file(finfo):
    """
    Detect the camera type by comparing the read video resolution with known
     outputted dimensions of different camera types.

    Args:
    finfo (dict): dictionary containing the file metadata,

    Returns:
    detected (str): name of the detected camera type.
    """

    detected = 'azure'
    potential_camera_dims = {
        'kinect': [[512, 424]],
        'realsense': [[640, 480]],
        'azure': [[640, 576],
                  [320, 288],
                  [512, 512],
                  [1024, 1024]]
    }

    # Check dimensions
    if finfo is not None:
        if list(finfo['dims']) in potential_camera_dims['azure']:
            # Default Azure dimensions
            detected = 'azure'
        elif list(finfo['dims']) in potential_camera_dims['realsense']:
            # Realsense D415 output dimensions
            detected = 'realsense'
        elif list(finfo['dims']) in potential_camera_dims['kinect']:
            # Kinect output dimensions
            detected = 'kinect'
        else:
            warnings.warn('Could not infer camera type, using default Azure parameters.')

    return detected

def detect_and_set_camera_parameters(config_data, input_file=None):
    """
    Read the camera type and info and set the bg_roi_weights to the precomputed values.
    If camera_type is None, function will assume kinect is used.

    Args:
    config_data (dict): dictionary containing all input parameters.
    input_file (str): path to raw depth file

    Returns:
    config_data (dict): updated dictionary with bg-roi-weights to use for extraction.
    """

    # Auto-setting background weights
    camera_type = config_data.get('camera_type')
    finfo = config_data.get('finfo')

    default_parameters = {
        'kinect': {
            'bg_roi_weights': (1, .1, 1),
            'pixel_format': 'gray16le',
            'movie_dtype': '<u2'
        },
        'azure': {
            'bg_roi_weights': (10, 0.1, 1),
            'pixel_format': 'gray16be',
            'movie_dtype': '>u2'
        },
        'realsense': {
            'bg_roi_weights': (10, 0.1, 1),
            'pixel_format': 'gray16le',
            'movie_dtype': '<u2',
        },
    }

    if type(input_file) is tarfile.TarFile:
        detected = 'kinect'
    elif camera_type == 'auto' and input_file is not None:
        if input_file.endswith('.dat'):
            detected = 'kinect'
        elif input_file.endswith('.mkv'):
            detected = 'azure'
        elif input_file.endswith('.avi'):
            if finfo is None:
                finfo = get_movie_info(input_file,
                                       mapping=config_data.get('mapping', 0),
                                       threads=config_data.get('threads', 8)
                                       )
            detected = detect_avi_file(finfo)
        else:
            warnings.warn('Extension not recognized, trying default Kinect v2 parameters.')
            detected = 'kinect'

        # set the params
        config_data.update(**default_parameters[detected])
    elif camera_type in default_parameters:
        # update the config with the corresponding param set
        config_data.update(**default_parameters[camera_type])
    else:
        warnings.warn('Warning, make sure the following parameters are set to best handle your camera type: '
                      '"bg_roi_weights", "pixel_format", "movie_dtype"')

    return config_data

def check_filter_sizes(config_data):
    """
    Ensure spatial and temporal filter kernel sizes are odd numbers.

    Args:
    config_data (dict): a dictionary holding all extraction parameters

    Returns:
    config_data (dict): Updated configuration dict

    """

    # Ensure filter kernel sizes are odd
    if config_data['spatial_filter_size'][0] % 2 == 0 and config_data['spatial_filter_size'][0] > 0:
        warnings.warn("Spatial Filter Size must be an odd number. Incrementing value by 1.")
        config_data['spatial_filter_size'][0] += 1
    if config_data['temporal_filter_size'][0] % 2 == 0 and config_data['temporal_filter_size'][0] > 0:
        config_data['temporal_filter_size'][0] += 1
        warnings.warn("Spatial Filter Size must be an odd number. Incrementing value by 1.")

    return config_data

def generate_missing_metadata(sess_dir, sess_name):
    """
    Generate metadata.json with default avlues for session that does not already include one.

    Args:
    sess_dir (str): Path to session directory to create metadata.json file in.
    sess_name (str): Session Name to set the metadata SessionName.

    Returns:
    """

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': '', f'SessionName': f'{sess_name}',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}

    with open(join(sess_dir, 'metadata.json'), 'w') as fp:
        json.dump(sample_meta, fp)

def load_metadata(metadata_file):
    """
    Load metadata from session metadata.json file.

    Args:
    metadata_file (str): path to metadata file

    Returns:
    metadata (dict): metadata dictionary of JSON contents
    """

    try:
        if not exists(metadata_file):
            generate_missing_metadata(dirname(metadata_file), basename(dirname(metadata_file)))

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except TypeError:
        # try loading directly
        metadata = json.load(metadata_file)

    return metadata

def load_found_session_paths(input_dir, exts):
    """
    Find all depth files with the specified extension recursively in input directory.

    Args:
    input_dir (str): path to project base directory holding all the session sub-folders.
    exts (list or str): list of extensions to search for, or a single extension in string form.

    Returns:
    files (list): sorted list of all paths to found depth files
    """

    if not isinstance(exts, (tuple, list)):
        exts = [exts]

    files = []
    for ext in exts:
        files.extend(glob(join(input_dir, '*/*' + ext), recursive=True))

    return sorted(files)

def get_strels(config_data):
    """
    Get dictionary object of cv2 StructuringElements for image filtering given
    a dict of configurations parameters.

    Args:
    config_data (dict): dict containing cv2 Structuring Element parameters

    Returns:
    str_els (dict): dict containing cv2 StructuringElements used for image filtering
    """

    str_els = {
        'strel_dilate': select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate'])),
        'strel_erode': select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_erode'])),
        'strel_tail': select_strel(config_data['tail_filter_shape'], tuple(config_data['tail_filter_size'])),
        'strel_min': select_strel(config_data['cable_filter_shape'], tuple(config_data['cable_filter_size']))
    }

    return str_els

def select_strel(string='e', size=(10, 10)):
    """
    Returns structuring element of specified shape.

    Args:
    string (str): string to indicate whether to use ellipse or rectangle
    size (tuple): size of structuring element

    Returns:
    strel (cv2.StructuringElement): selected cv2 StructuringElement to use in video filtering or ROI dilation/erosion.
    """

    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel

def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    """
    Converts x, y coordinates in pixel space to mm.

    Args:
    coords (list): list of x,y pixel coordinates
    resolution (tuple): image dimensions in pixels, (width, height)
    field_of_view (tuple): camera angular field of view in DEGREES, (horizontal,
        vertical). These are view angles, not focal lengths -- the focal lengths
        are derived from them below.
    true_depth (float): detected true depth

    Returns:
    new_coords (list): x,y coordinates in mm

    Note:
    Focal length is derived from the NOMINAL field of view, not from a
    per-camera calibration. Every caller in this package uses the defaults above
    (only true_depth is passed) and there is no CLI or config path to override
    them, so supplying real calibrated intrinsics is NOT currently supported end
    to end. Calibrated fx/fy would be more accurate -- the factory values have
    fx == fy, which no FOV-derived pair does -- but wiring that through the
    extraction config is future work, not an available option today.
    """

    # http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    # http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    # http://smeenk.com/kinect-field-of-view-comparison/

    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    # Pinhole focal length: f = (resolution / 2) / tan(fov / 2). The previous
    # code used deg2rad(fov/2) in place of tan(deg2rad(fov/2)) -- the small-angle
    # approximation -- which overestimates the focal length by ~10-13% and so
    # underestimates every mm displacement by the same amount. On a 512x424
    # Kinect2 the pinhole values (fx=361.6, fy=367.2) land within ~2% of the
    # factory-calibrated intrinsics (~368), where the small-angle values
    # (fx=415.5, fy=404.9) are ~10-13% off. See the Note in the docstring: these
    # are nominal-FOV derived, not calibrated per camera.
    fw = resolution[0] / (2 * np.tan(np.deg2rad(field_of_view[0] / 2)))
    fh = resolution[1] / (2 * np.tan(np.deg2rad(field_of_view[1] / 2)))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def scalar_attributes():
    """
    Gets scalar attributes dict with names paired with descriptions.

    Returns:
    attributes (dict): a dictionary of metadata keys and descriptions.
    """

    attributes = {
        'centroid_x_px': 'X centroid (pixels)',
        'centroid_y_px': 'Y centroid (pixels)',
        'velocity_2d_px': '2D velocity (pixels / frame), note that missing frames are not accounted for',
        'width_px': 'Mouse width (pixels)',
        'length_px': 'Mouse length (pixels)',
        'area_px': 'Mouse area (pixels)',
        'centroid_x_mm': 'X centroid (mm)',
        'centroid_y_mm': 'Y centroid (mm)',
        'velocity_2d_mm': '2D velocity (mm / frame), note that missing frames are not accounted for',
        'velocity_3d_mm': '3D velocity (mm / frame), note that missing frames are not accounted for',
        'width_mm': 'Mouse width (mm)',
        'length_mm': 'Mouse length (mm)',
        'area_mm': 'Mouse area (mm)',
        'height_ave_mm': 'Mouse average height (mm)',
        'angle': 'Angle (radians, unwrapped)',
        'velocity_theta': 'Angular component of velocity (arctan(vel_x, vel_y))'
    }

    return attributes


def convert_raw_to_avi_function(input_file, chunk_size=2000, fps=30, delete=False, threads=3):
    """
    Compress depth file (.dat, '.mkv') to avi file.

    Args:
    input_file (str): path to depth file
    chunk_size (int): size of chunks to process at a time
    fps (int): frames per second
    delete (bool): flag for deleting original depth file
    threads (int): number of threads to write video.

    """

    new_file = f'{splitext(input_file)[0]}.avi'
    print(f'Converting {input_file} to {new_file}')
    # turn into os system call...
    use_kwargs = {
        'output-file': new_file,
        'chunk-size': chunk_size,
        'fps': fps,
        'threads': threads
    }
    use_flags = {
        'delete': delete
    }
    base_command = f'moseq2-extract convert-raw-to-avi {input_file}'
    for k, v in use_kwargs.items():
        base_command += f' --{k} {v}'
    for k, v in use_flags.items():
        if v:
            base_command += f' --{k}'

    print(base_command)
    print()

    os.system(base_command)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    Create subarrays of an array with a given stride and window length.

    Args:
    a (np.ndarray) - original array
    L (int) - Window Length
    S (int) - Stride size

    Returns:
    (np.ndarray) - array of subarrays
    """

    # https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052

    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def dict_to_h5(h5, dic, root='/', annotations=None):
    """
    Save an dict to an h5 file, mounting at root.
    Keys are mapped to group names recursively.

    Args:
    h5 (h5py.File instance): h5py.file object to operate on
    dic (dict): dictionary of data to write
    root (string): group on which to add additional groups and datasets
    annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as dic.
    
    """

    if not root.endswith('/'):
        root = root + '/'

    if annotations is None:
        annotations = {} #empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in dic.items():
        dest = root + key
        try:
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5[dest] = item
            elif isinstance(item, (tuple, list)):
                h5[dest] = np.asarray(item)
            elif isinstance(item, (int, float)):
                h5[dest] = np.asarray([item])[0]
            elif item is None:
                h5.create_dataset(dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str)))
            elif isinstance(item, dict):
                dict_to_h5(h5, item, dest)
            else:
                raise ValueError('Cannot save {} type to key {}'.format(type(item), dest))
        except Exception as e:
            print(e)
            if key != 'inputs':
                print('h5py could not encode key:', key)

        if key in annotations:
            if annotations[key] is None:
                h5[dest].attrs['description'] = ""
            else:
                h5[dest].attrs['description'] = annotations[key]


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """
    Recursively find h5 files, along with yaml files with the same basename

    Args:
    root_dir (str): path to base directory to begin recursive search in.
    ext (str): extension to search for
    yaml_string (str): string for filename formatting when saving data

    Returns:
    h5s (list): list of found h5 files
    dicts (list): list of found metadata files
    yamls (list): list of found yaml files
    """
    if not ext.startswith('.'):
        ext = '.' + ext

    def has_frames(f):
        try:
            with h5py.File(f, 'r') as h5f:
                return 'frames' in h5f
        except OSError:
            warnings.warn(f'Error reading {f}, skipping...')
            return False

    h5s = glob(join(abspath(root_dir), '**', f'*{ext}'), recursive=True)
    h5s = filter(lambda f: exists(yaml_string.format(f.replace(ext, ''))), h5s)
    h5s = list(filter(has_frames, h5s))
    yamls = list(map(lambda f: yaml_string.format(f.replace(ext, '')), h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def escape_path(path):
    """
    Return a path to return to original base directory.

    Args:
    path (str): path to current working dir

    Returns:
    path (str): path to original base_dir
    """

    return re.sub(r'\s', '\ ', path)


def clean_file_str(file_str: str, replace_with: str = '-') -> str:
    """
    Removes invalid characters for a file name from a string.

    Args:
    file_str (str): filename substring to replace
    replace_with (str): value to replace str with

    Returns:
    out (str): cleaned file string
    """

    out = re.sub(r'[ <>:"/\\|?*\']', replace_with, file_str)
    # find any occurrences of `replace_with`, i.e. (--)
    return re.sub(replace_with * 2, replace_with, out)


def load_textdata(data_file, dtype=np.float32):
    """
    Loads timestamp from txt/csv file.

    Args:
    data_file (str): path to timestamp file
    dtype (dtype): data type of timestamps

    Returns:
    data (np.ndarray): timestamp data
    timestamps (numpy.array): the array for the timestamps
    """

    data = []
    timestamps = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            tmp = line.split(' ', 1)
            # appending timestamp value
            timestamps.append(int(float(tmp[0])))

            # append data indicator value
            clean_data = np.fromstring(tmp[1].replace(" ", "").strip(), sep=',', dtype=dtype)
            data.append(clean_data)

    data = np.stack(data, axis=0).squeeze()
    timestamps = np.array(timestamps, dtype=np.int)

    return data, timestamps


def time_str_for_filename(time_str: str) -> str:
    """
    Process the timestamp to be used in the filename.

    Args:
    time_str (str): time str to format

    Returns:
    out (str): formatted timestamp str
    """

    out = time_str.split('.')[0]
    out = out.replace(':', '-').replace('T', '_')
    return out

def build_path(keys: dict, format_string: str, snake_case=True) -> str:
    """
    Produce a new file name using keys collected from extraction h5 files.

    Args:
    keys (dict): dictionary specifying which keys used to produce the new file name
    format_string (str): the string to reformat using the `keys` dictionary i.e. '{subject_name}_{session_name}'.
    snake_case (bool): flag to save the files with snake_case

    Returns:
    out (str): a newly formatted filename useable with any operating system
    """

    if 'start_time' in keys:
        # process the time value
        keys['start_time'] = time_str_for_filename(keys['start_time'])

    if snake_case:
        keys = valmap(camel_to_snake, keys)

    return clean_file_str(format_string.format(**keys))

def read_yaml(yaml_file):
    """
    Read yaml file into a dictionary

    Args:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml contents
    """

    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def mouse_threshold_filter(h5file, thresh=0):
    """
    Filter frames in h5 files by threshold value.

    Args:
    h5file (str): path to h5 file
    thresh (int): threshold at which to apply filter

    Returns:
    (3d-np boolean array): array of regions to include after threshold filter.
    """

    with h5py.File(h5file, 'r') as f:
        # select 1st 1000 frames
        frames = f['frames'][:min(f['frames'].shape[0], 1000)]
    return np.nanmean(frames) > thresh

def _load_h5_to_dict(file: h5py.File, path) -> dict:
    """
    Loads h5 contents to dictionary object.

    Args:
    h5file (h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns:
    ans (dict): a dict with h5 file contents with the same path structure
    """

    ans = {}
    for key, item in file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path) -> dict:
    """
    Load h5 contents to dictionary object.

    Args:
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns:
    out (dict): a dict with h5 file contents with the same path structure
    """

    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, h5py.File):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out

def clean_dict(dct):
    """
    Standardize types of dict value.

    Args:
    dct (dict): dict object with mixed type value objects.

    Returns:
    out (dict): dict object with list value objects.
    """

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            out = np.asscalar(e)
        else:
            out = e
        return out

    return valmap(clean_entry, dct)

_underscorer1: Pattern[str] = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile('([a-z0-9])([A-Z])')

def camel_to_snake(s):
    """
    Convert CamelCase to snake_case

    Args:
    s (str): CamelCase string to convert to snake_case.

    Returns:
    (str): string in snake_case
    """

    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def recursive_find_unextracted_dirs(root_dir=os.getcwd(),
                                    session_pattern=r'session_\d+\.(?:tgz|tar\.gz)',
                                    extension='.dat',
                                    yaml_path='proc/results_00.yaml',
                                    metadata_path='metadata.json',
                                    skip_checks=False):
    """
    Recursively find unextracted (or incompletely extracted) directories

    Args:
    root_dir (str): path to base directory to start recursive search for unextracted folders.
    session_pattern (str): folder name pattern to search for
    extension (str): file extension to search for
    yaml_path (str): path to respective extracted metadata
    metadata_path (str): path to relative metadata.json files
    skip_checks (bool): indicates whether to check if the files exist at the given relative paths

    Returns:
    proc_dirs (1d-list): list of paths to each unextracted session's proc/ directory
    """

    from moseq2_extract.helpers.data import check_completion_status

    session_archive_pattern = re.compile(session_pattern)

    proc_dirs = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension) and not file.startswith("ir"):  # test for uncompressed session
                status_file = join(root, yaml_path)
                metadata_file = join(root, metadata_path)
            elif session_archive_pattern.fullmatch(file):  # test for compressed session
                session_name = basename(file).replace('.tar.gz', '').replace('.tgz', '')
                status_file = join(root, session_name, yaml_path)
                metadata_file = join(root, '{}.json'.format(session_name))
            else:
                continue  # skip this current file as it does not look like session data

            # perform checks, append depth file to list if extraction is missing or incomplete
            if skip_checks or (not check_completion_status(status_file) and exists(metadata_file)):
                proc_dirs.append(join(root, file))

    return proc_dirs

def click_param_annot(click_cmd):
    """
    Return a dict that maps option names to help strings from a click.Command instance.

    Args:
    click_cmd (click.Command): command to annotate

    Returns:
    annotations (dict): dictionary of options and their help messages
    """

    annotations = {}
    for p in click_cmd.params:
        if isinstance(p, click.Option):
            annotations[p.human_readable_name] = p.help
    return annotations

def estimate_depth_range(bground_im, pad_floor=50, bins=200):
    """
    Estimate the arena-floor depth range from a background image.

    Replaces sampling a single (possibly dropped-out) pixel at the image
    centroid. Instead it histograms every valid depth pixel, finds the candidate
    modes, and selects the one whose thresholded pixels form the largest compact
    connected region (favouring a region that covers the frame center). The
    padding around the chosen mode is derived from the actual spread of the
    floor pixels (95% span and MAD) with a floor of ``pad_floor`` mm, so a tilted
    or wide arena that spans more than +/-50 mm is captured.

    Args:
    bground_im (numpy.ndarray): background depth image (mm).
    pad_floor (int): minimum +/- padding (mm) around the detected floor mode.
    bins (int): number of histogram bins.

    Returns:
    depth_range (list): [min, max] depth (mm) bracketing the arena floor.
    """

    finite = bground_im[np.isfinite(bground_im) & (bground_im > 0)]
    if finite.size == 0:
        raise ValueError(
            "Background image has no valid (finite, nonzero) depth pixels to "
            "estimate a floor depth range from."
        )

    counts, edges = np.histogram(finite, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts, np.ones(5) / 5.0, mode="same")

    # candidate floor modes = local maxima of the smoothed histogram
    peaks = [
        i
        for i in range(1, len(smooth) - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > 0
    ]
    if not peaks:
        peaks = [int(np.argmax(smooth))]

    H, W = bground_im.shape
    cy, cx = H // 2, W // 2

    best_score, best_center = -1.0, float(centers[int(np.argmax(smooth))])
    for p in peaks:
        lo, hi = centers[p] - pad_floor, centers[p] + pad_floor
        mask = np.ascontiguousarray(
            ((bground_im >= lo) & (bground_im <= hi)).astype("uint8"))
        # NOTE: pass connectivity and ltype POSITIONALLY. In the pinned
        # opencv-python 4.1.2 the keyword form
        # `connectedComponentsWithStats(mask, connectivity=8)` segfaults on
        # full-size depth masks (verified on a 424x512 frame); supplying ltype
        # explicitly avoids the faulty binding overload.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, 8, cv2.CV_32S
        )
        if n_labels <= 1:
            continue
        # largest connected component that is not the background label (0)
        comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[comp, cv2.CC_STAT_AREA])
        bw, bh = stats[comp, cv2.CC_STAT_WIDTH], stats[comp, cv2.CC_STAT_HEIGHT]
        fill = area / max(1.0, float(bw * bh))  # bounding-box fill => compactness
        # Score how much of a central WINDOW the component covers rather than
        # testing the single centre pixel. A dropped-out centre pixel would
        # otherwise deny every candidate the centre bonus, letting a larger
        # off-centre surface (e.g. the arena walls) win -- the exact
        # single-pixel fragility this function exists to remove.
        half_h, half_w = max(1, H // 8), max(1, W // 8)
        window = labels[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
        centre_frac = float((window == comp).mean()) if window.size else 0.0
        score = area * fill * (1.0 + centre_frac)
        if score > best_score:
            best_score, best_center = score, float(centers[p])

    # derive the pad from the spread of pixels that belong to the chosen mode
    band = finite[np.abs(finite - best_center) <= 3 * pad_floor]
    if band.size:
        lo_pct, hi_pct = np.percentile(band, [2.5, 97.5])
        mad = 1.4826 * np.median(np.abs(band - np.median(band)))
        pad = max(float(pad_floor), (hi_pct - lo_pct) / 2.0, 2.0 * mad)
    else:
        pad = float(pad_floor)

    return [int(round(best_center - pad)), int(round(best_center + pad))]


def plot_depth_range_diagnostic(bground_im, depth_range, output_path, bins=200):
    """
    Save a depth-histogram diagnostic showing the detected floor range.

    Ambiguous scenes (e.g. a small arena with a large room floor behind it)
    cannot be resolved by any unsupervised heuristic, so this writes a figure
    that lets a human confirm the automatically chosen range at a glance.

    Args:
    bground_im (numpy.ndarray): background depth image (mm).
    depth_range (list): the [min, max] depth range that was selected.
    output_path (str): path to write the PNG diagnostic to.
    bins (int): number of histogram bins.

    Returns:
    output_path (str): the path the figure was written to (or None on failure).
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        finite = bground_im[np.isfinite(bground_im) & (bground_im > 0)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(finite.ravel(), bins=bins, color="0.6")
        ax.axvspan(depth_range[0], depth_range[1], color="tab:green", alpha=0.25,
                   label=f"selected range {depth_range}")
        ax.set_xlabel("Depth from camera (mm)")
        ax.set_ylabel("Pixel count")
        ax.set_title("Background depth distribution and detected floor range")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path
    except Exception as e:
        warnings.warn(f"Could not write depth-range diagnostic plot: {e}")
        return None


def get_bucket_center(img, true_depth, threshold=650):
    """
    Find Centroid coordinates of circular bucket.

    Args:
    img (np.ndaarray): original background image.
    true_depth (float): distance value from camera to bucket floor (automatically pre-computed)
    threshold (float): distance values to accept region into detected circle. (used to reduce fall noise interference)

    Returns:
    cX (int): x-coordinate of circle centroid
    cY (int): y-coordinate of circle centroid
    """

    # https://stackoverflow.com/questions/19768508/python-opencv-finding-circle-sun-coordinates-of-center-the-circle-from-pictu
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(img, threshold, true_depth, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY

def make_gradient(width, height, h, k, a, b, theta=0):
    """
    Create gradient around bucket floor representing slanted wall values.

    Args:
    width (int): bounding box width
    height (int) bounding box height
    h (int): centroid x coordinate
    k (int): centroid y coordinate
    a (int): x-radius of drawn ellipse
    b (int): y-radius of drawn ellipse
    theta (float): degree to rotate ellipse in radians. (has no effect if drawing a circle)

    Returns:
    np.ndarray: numpy array with weighted values from 0.08 -> 0.8 representing the proportion of values
    to create a gradient from. 0.8 being the proportioned values closest to the circle wall.
    """

    # https://stackoverflow.com/questions/49829783/draw-a-gradual-change-ellipse-in-skimage/49848093#49848093
    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a ** 2, b ** 2

    # Generate (x,y) coordinate arrays
    y, x = np.mgrid[-k:height - k, -h:width - h]

    # Calculate the weight for each pixel
    weights = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)

    return np.clip(0.98 - weights, 0, 0.81)


def graduate_dilated_wall_area(bground_im, config_data, strel_dilate, output_dir):
    """
    Creates a gradient to represent the dilated (now visible) bucket wall regions.
    Only is used if background is dilated to capture larger rodents in convex shaped buckets (\_/).
    
    Args:
    bground_im (np.ndarray): the computed background image.
    config_data (dict): dictionary containing helper user configuration parameters.
    strel_dilate (cv2.structuringElement): dilation structuring element used to dilate background image.
    output_dir (str): path to save newly computed background to use.

    Returns:
    bground_im (np.ndarray): the new background image with a gradient around the floor from high to low depth values.
    """

    # store old and new backgrounds
    old_bg = deepcopy(bground_im)

    # dilate background size to match ROI size and attribute wall noise to cancel out
    bground_im = cv2.dilate(old_bg, strel_dilate, iterations=config_data.get('dilate_iterations', 5))

    # determine center of bground roi
    width, height = bground_im.shape[1], bground_im.shape[0]  # shape of bounding box

    # getting helper user parameters
    true_depth = config_data['true_depth']
    xoffset = config_data.get('x_bg_offset', -2)
    yoffset = config_data.get('y_bg_offset', 2)
    widen_radius = config_data.get('widen_radius', 0)
    bg_threshold = config_data.get('bg_threshold', np.median(bground_im))

    # getting bground centroid
    cx, cy = get_bucket_center(deepcopy(old_bg), true_depth, threshold=bg_threshold)

    # set up gradient
    h, k = cx + xoffset, cy + yoffset   # centroid of gradient circle
    a, b = cx + widen_radius + 67, cy + widen_radius + 67 # x,y radii of gradient circle
    theta = math.pi/24 # gradient angle; arbitrary - used to rotate ellipses.

    # create slant gradient
    bground_im = np.uint16((make_gradient(width, height, h, k, a, b, theta)) * 255)

    # scale it back to depth
    bground_im = np.uint16((bground_im/bground_im.max())*true_depth)

    # overlay with actual bucket floor distance
    if config_data.get('floor_slant', False):
        ret, thresh = cv2.threshold(old_bg, np.median(old_bg), true_depth, 0)
        contours, _ = cv2.findContours(thresh.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(bground_im.shape, np.uint8)

        cv2.drawContours(mask, contours, -1, (255), -1)

        tmp = np.where(mask == True, bground_im, old_bg)
        bground_im = np.where(tmp == 0, bground_im, tmp)
    else:
        mask = np.ma.equal(old_bg, old_bg.max())
        bground_im = np.where(mask == True, old_bg, bground_im)

    bground_im = cv2.GaussianBlur(bground_im, (7, 7), 7)

    write_image(join(output_dir, 'new_bg.tiff'), bground_im, scale=True)

    return bground_im

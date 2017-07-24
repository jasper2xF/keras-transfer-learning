import os

DATA_ROOT = "/media/jasper/Data/ml-data/planet_ama_kg"
DATA_FORMAT = "jpg"

def get_jpeg_data_files_paths():
    """
    Returns the input file folders path.

    :return: list of strings
        The input file paths as list [train_dir, test_dir, test_additional, train_csv_file]
    """

    data_root_folder = os.path.abspath(DATA_ROOT)
    train_dir = os.path.join(data_root_folder, 'train-jpg')
    test_dir = os.path.join(data_root_folder, 'test-jpg')
    test_additional = os.path.join(data_root_folder, 'test-jpg-additional')

    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
    return [train_dir, test_dir, test_additional, train_csv_file]


def get_tif_data_files_paths():
    """
    Returns the input file folders path

    Note that additional test data (as for jpd data) doesn't exist for tif data. Method will return None for additional
    rest data path.

    :return: list of strings
        The input file paths as list [train_dir, test_dir, None, train_csv_file]
    """

    data_root_folder = os.path.abspath(DATA_ROOT)
    train_dir = os.path.join(data_root_folder, 'train-tif-v2')
    test_dir = os.path.join(data_root_folder, 'test-tif')
    #test_additional = os.path.join(data_root_folder, 'test-tif-additional')

    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
    return [train_dir, test_dir, None, train_csv_file]

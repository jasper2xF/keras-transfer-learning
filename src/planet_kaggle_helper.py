import os

# Data parameters
DATA_ROOT = "/media/jasper/Data/ml-data/planet_ama_kg"
DATA_FORMAT = "jpg"


def get_data(competition_name, destination_path, is_datasets_present, test, test_additional, test_additional_u,
             test_labels, test_u, train, train_u):
    """
    Check whether competition data exists or download.

    DEPRECATED - functionality not confirmed.
    """
    # If the folders already exists then the files may already be extracted
    # This is a bit hacky but it's sufficient for our needs
    datasets_path = planet_kaggle_helper.get_jpeg_data_files_paths()
    for dir_path in datasets_path:
        if os.path.exists(dir_path):
            is_datasets_present = True

    if not is_datasets_present:
        # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
        downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

        train_output_path = downloader.download_dataset(train, destination_path)
        downloader.decompress(train_output_path, destination_path)  # Outputs a tar file
        downloader.decompress(destination_path + train_u,
                              destination_path)  # Extract the content of the previous tar file
        os.remove(train_output_path)  # Removes the 7z file
        os.remove(destination_path + train_u)  # Removes the tar file

        test_output_path = downloader.download_dataset(test, destination_path)
        downloader.decompress(test_output_path, destination_path)  # Outputs a tar file
        downloader.decompress(destination_path + test_u,
                              destination_path)  # Extract the content of the previous tar file
        os.remove(test_output_path)  # Removes the 7z file
        os.remove(destination_path + test_u)  # Removes the tar file

        test_add_output_path = downloader.download_dataset(test_additional, destination_path)
        downloader.decompress(test_add_output_path, destination_path)  # Outputs a tar file
        downloader.decompress(destination_path + test_additional_u,
                              destination_path)  # Extract the content of the previous tar file
        os.remove(test_add_output_path)  # Removes the 7z file
        os.remove(destination_path + test_additional_u)  # Removes the tar file

        test_labels_output_path = downloader.download_dataset(test_labels, destination_path)
        downloader.decompress(test_labels_output_path, destination_path)  # Outputs a csv file
        os.remove(test_labels_output_path)  # Removes the zip file
    else:
        print("All datasets are present.")
    return None


def get_proccessed_data_paths():
    """
    Returns the processed numpy file folders path.

    :return: list of strings
        The input file paths as list [train_processed_dir, test_processed_dir]
    """
    train_processed_dir = os.path.join(DATA_ROOT, "preprocessing", DATA_FORMAT, "train")
    test_processed_dir = os.path.join(DATA_ROOT, "preprocessing", DATA_FORMAT, "test")
    return train_processed_dir, test_processed_dir


def get_data_files_paths():
    """
    Returns the input file folders path.

    Note that additional test data (as for jpg data) doesn't exist for tif data. Method will return None for additional
    test data path.

    :return: list of strings
        The input file paths as list [train_dir, test_dir, test_additional, train_csv_file]
    """
    if DATA_FORMAT == "jpg":
        train_dir, test_dir, test_additional, train_csv_file = get_jpeg_data_files_paths()
    elif DATA_FORMAT == "tif":
        train_dir, test_dir, test_additional, train_csv_file = get_tif_data_files_paths()
    else:
        raise ValueError("Invalid DATA_FORMAT.")
    return [train_dir, test_dir, test_additional, train_csv_file]


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

    Note that additional test data (as for jpg data) doesn't exist for tif data. Method will return None for additional
    test data path.

    :return: list of strings
        The input file paths as list [train_dir, test_dir, None, train_csv_file]
    """

    data_root_folder = os.path.abspath(DATA_ROOT)
    train_dir = os.path.join(data_root_folder, 'train-tif-v2')
    test_dir = os.path.join(data_root_folder, 'test-tif')
    #test_additional = os.path.join(data_root_folder, 'test-tif-additional')

    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
    return [train_dir, test_dir, None, train_csv_file]

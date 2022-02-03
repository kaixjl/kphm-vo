from functools import reduce
import os
import os.path
import datetime
import argparse
import shutil
from params import param

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_tag", "-a",   type=str, help="specify the archive tag to be used")
    parser.add_argument("--config_name", "-c",   type=str, default=param.config_name, help="specify the tag of the running to be archived")
    parser.add_argument("--remove_original", "-r", help="remove original records.", action='store_true')
    parser.add_argument("--no_archive", "-n", help="do not archive.", action='store_true')
    args0 = parser.parse_args()
    return args0

def main():
    args0 = parse_arg()
    if args0.archive_tag is None:
        datetime_now = datetime.datetime.now()
        args0.archive_tag = "ar{}_{}_{}".format(datetime_now.strftime("%Y%m%d"), args0.config_name, datetime_now.strftime("%H%M%S"))
    archive(args0.config_name, args0.archive_tag, args0.remove_original, args0.no_archive)

def archive(config_name, archive_tag, remove_original=False, no_archive=False):
    path_sw_dir = os.path.join("runs", config_name)
    path_records_dir = os.path.join("records", config_name)
    path_result_dir = os.path.join("result", config_name)

    if not no_archive:
        path_archive_dir = os.path.join("archive", archive_tag)
        if os.path.exists(path_archive_dir):
            shutil.rmtree(path_archive_dir)

        os.makedirs(path_archive_dir)

        if os.path.exists(path_sw_dir):
            shutil.copytree(path_sw_dir, os.path.join(path_archive_dir, path_sw_dir))
        if os.path.exists(path_records_dir):
            shutil.copytree(path_records_dir, os.path.join(path_archive_dir, path_records_dir))
        if os.path.exists(path_result_dir):
            shutil.copytree(path_result_dir, os.path.join(path_archive_dir, path_result_dir))
        if os.path.exists("nohup.out"):
            shutil.copy("nohup.out", path_archive_dir)

    if remove_original:
        if os.path.exists(path_sw_dir):
            shutil.rmtree(path_sw_dir, ignore_errors=True)
        if os.path.exists(path_records_dir):
            shutil.rmtree(path_records_dir, ignore_errors=True)
        if os.path.exists(path_result_dir):
            shutil.rmtree(path_result_dir, ignore_errors=True)
        if os.path.exists("nohup.out"):
            os.remove("nohup.out")

    pass

if __name__=="__main__":
    main()

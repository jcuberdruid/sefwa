import os
import json
import shutil
import sys
import base64
from datetime import datetime
from .. import paths

class ExperimentManager:
    def __init__(self):
        self.data_dir = paths.dataDir
        self.experiment_dir = paths.experimentDir
        self.analysis_dir = paths.analysisDir
        self.results_dir = paths.resultDir
        self.archive_dir = paths.archiveDir
        self.experiment_template_dir = os.path.join(parent_dir, "../templates/experiment/")
        self.analysis_template_dir = os.path.join(parent_dir, "../templates/analysis/")

    def new_experiment(self, name, description):
        uid = self.generateUID()
        print(uid)
        exp_path = os.path.join(self.experiment_dir, uid)
        os.makedirs(exp_path)

        # Copy template contents to the new experiment directory
        self._copy_template_contents(self.experiment_template_dir, exp_path)
        self._write_info_file(exp_path, name, description, uid)
        os.makedirs(os.path.join(self.results_dir, ("results_"+uid)))

    def fork_experiment(self, existing_id):
        self._fork_item(self.experiment_dir, existing_id)

    def new_analysis(self, name, description):
        uid = self.generateUID()
        print(uid)
        analysis_path = os.path.join(self.analysis_dir, uid)
        os.makedirs(analysis_path)

        # Copy template contents to the new analysis directory
        self._copy_template_contents(self.analysis_template_dir, analysis_path)
        self._write_info_file(analysis_path, name, description, uid)

    def fork_analysis(self, existing_id):
        self._fork_item(self.analysis_dir, existing_id)

    def new_data(self, name, description):
        uid = self.generateUID()
        print(uid)
        data_path = os.path.join(self.data_dir, uid)
        os.makedirs(data_path)

        self._write_info_file(data_path, name, description, uid)

    def archive(self, ID, item_type):
        if item_type == 'experiment':
            self._archive_item(self.experiment_dir, self.results_dir, ID)
        elif item_type == 'analysis':
            self._archive_item(self.analysis_dir, None, ID)
        elif item_type == 'data':
            self._archive_item(self.data_dir, None, ID)
        else:
            print("Invalid item type for archiving.")

    def unarchive(self, ID, item_type):
        archive_path = os.path.join(self.archive_dir, ID + '.zip')
        if not os.path.exists(archive_path):
            print("Archive not found.")
            return

        shutil.unpack_archive(archive_path, self.archive_dir)

        if item_type == 'experiment':
            shutil.move(os.path.join(self.archive_dir, ID), self.experiment_dir)
            shutil.move(os.path.join(self.archive_dir, "results_" + ID), self.results_dir)
        elif item_type == 'analysis':
            shutil.move(os.path.join(self.archive_dir, ID), self.analysis_dir)
        elif item_type == 'data':
            shutil.move(os.path.join(self.archive_dir, ID), self.data_dir)

    def mostRecent(self, item_type):
        dir_path = None
        if item_type == 'experiment':
            dir_path = self.experiment_dir
        elif item_type == 'analysis':
            dir_path = self.analysis_dir
        elif item_type == 'data':
            dir_path = self.data_dir

        if not dir_path:
            print("Invalid item type.")
            return

        latest, latest_time = self._find_most_recent(dir_path)
        if latest:
            with open(os.path.join(dir_path, latest, "info.json")) as f:
                info = json.load(f)
            print(f"Name: {info['name']}\nID: {info['ID']}\nDescription: {info['description']}\nLast Modified: {datetime.fromtimestamp(latest_time)}")
            print(f"Path: {os.path.join(dir_path, latest)}")

    def generateUID(self):
        while True:
            uid = base64.urlsafe_b64encode(os.urandom(6)).decode('utf-8')  # Generates 8 char semi-UID
            if not self._is_duplicate_uid(uid):
                return uid

    # Private helper methods
    def _is_duplicate_uid(self, uid):
        for directory in [self.experiment_dir, self.analysis_dir, self.data_dir, self.archive_dir]:
            if uid in os.listdir(directory):
                return True
        return False

    def _copy_template_contents(self, template_dir, dest_dir):
        for item in os.listdir(template_dir):
            s = os.path.join(template_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, None)
            else:
                shutil.copy2(s, d)

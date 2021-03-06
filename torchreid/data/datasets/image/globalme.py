"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import, division, print_function
import glob
import os.path as osp
import re
import warnings

from ..dataset import ImageDataset


class GlobalMe(ImageDataset):
    """GlobalMe.

    Dataset statistics:
        - identities: 1610.
        - images: 0 (train) + 8450 (query) + 41107 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'globalme-reid'
    dataset_subdir = 'GlobalMe-reID'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, self.dataset_subdir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"{}".'.format(self.dataset_subdir))

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(GlobalMe, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def process_dir(dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data


class InternalWildtrack(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'wildtrack'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalWildtrack, self).__init__(root, market1501_500k, **kwargs)


class InternalAirport(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'airport'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalAirport, self).__init__(root, market1501_500k, **kwargs)


class InternalCameraTampering(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'camera_tampering'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalCameraTampering, self).__init__(root, market1501_500k, **kwargs)


class InternalGlobalMe(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'globalme'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalGlobalMe, self).__init__(root, market1501_500k, **kwargs)


class InternalMall(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'mall'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalMall, self).__init__(root, market1501_500k, **kwargs)


class InternalPSVIndoor(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'psv_indoor'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalPSVIndoor, self).__init__(root, market1501_500k, **kwargs)


class InternalPSVOutdoor(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'psv_outdoor'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalPSVOutdoor, self).__init__(root, market1501_500k, **kwargs)


class InternalSSPlatform(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'ss_platform'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalSSPlatform, self).__init__(root, market1501_500k, **kwargs)


class InternalSSStreet(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'ss_street'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalSSStreet, self).__init__(root, market1501_500k, **kwargs)


class InternalSSTicket(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'ss_ticket'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalSSTicket, self).__init__(root, market1501_500k, **kwargs)


class MarketTrainOnly(GlobalMe):
    dataset_dir = 'market1501'
    dataset_subdir = 'Train-only'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(MarketTrainOnly, self).__init__(root, market1501_500k, **kwargs)

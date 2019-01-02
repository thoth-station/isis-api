#!/usr/bin/env python3
# Isis
# Copyright(C) 2018 Fridolin Pokorny
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""This is TODO as it is missing a docstring."""

import os
import sys
import logging
import yaml

from thoth.common import init_logging

# Configure global application logging using Thoth's init_logging.
init_logging(logging_env_var_start="ISIS_LOG_")

_LOGGER = logging.getLogger(__name__)
_MODEL_PATH = os.path.join(os.getcwd(), "model")


#################################################################################
# TODO: move to storages
from thoth.storages.result_base import ResultStorageBase


class TaggingModelStore(ResultStorageBase):
    """Adapter for persisting and retrieving tagging model."""

    # TODO check in which way we can utilize KubeFlow or... for this

    RESULT_TYPE = "tagging"
    SCHEMA = None

    _METADATA_FILE = "metadata.tsv"
    _MODEL_FILE = "model.ckpt"

    def retrieve(self, dst_path: str) -> None:
        """Retrieve model."""
        for object_key in self.ceph.get_document_listing():
            content = self.ceph.retrieve_blob(object_key)
            with open(os.path.join(dst_path, object_key), "wb") as output_file:
                output_file.write(content)

        # Fix path to checkpoints based on path supplied.
        checkpoint_path = os.path.join(dst_path, "checkpoint")
        if not os.path.isfile(checkpoint_path):
            return

        with open(checkpoint_path, "r") as checkpoint_file:
            checkpoint_content = yaml.load(checkpoint_file)

        if "model_checkpoint_path" in checkpoint_content:
            checkpoint_content["model_checkpoint_path"] = os.path.join(
                dst_path,
                checkpoint_content["model_checkpoint_path"].rsplit("/", maxsplit=1)[-1],
            )

        if "all_model_checkpoint_paths" in checkpoint_content:
            checkpoint_content["all_model_checkpoint_paths"] = os.path.join(
                os.getcwd(),
                checkpoint_content["all_model_checkpoint_paths"].rsplit(
                    "/", maxsplit=1
                )[-1],
            )

        with open(checkpoint_path, "w") as checkpoint_file:
            for key, value in checkpoint_content.items():
                checkpoint_file.write(f'{key}: "{value}"\n')

    def store(self):
        """Store model to disk."""
        # TODO: persist model on creation
        raise NotImplementedError


#################################################################################


def download_model():
    """Download tagging model to desired path."""
    tagging = TaggingModelStore()
    tagging.connect()
    _LOGGER.info("Downloading tagging model to %r", _MODEL_PATH)
    tagging.retrieve(_MODEL_PATH)
    _LOGGER.info("Tagging model downloaded successfully to %r", _MODEL_PATH)


if __name__ == "__main__":
    sys.exit(download_model())

#!/usr/bin/env python3
# Isis
# Copyright(C) 2018, 2019 Fridolin Pokorny
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

"""Implementation of API v1."""

import logging
import heapq
import math

import numpy as np
from numpy.linalg import norm
from thoth.worker.storages import Project2VecModelStore
from thoth.worker.storages import PerformanceMaskStore

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SIMILAR_COUNT = 10
_MAX_SIMILAR_COUNT = 100

# Pre-loaded model.
_PROJECTS_BY_NAME = {}
_PROJECTS_BY_IDX = []
_VECTOR_SPACE = []
_PERFORMANCE_MASK = []


def get_python_similar_projects(
    project_name: str, count: int = _DEFAULT_SIMILAR_COUNT
) -> tuple:
    """Get similar projects to the given project."""
    parameters = locals()

    if project_name not in _PROJECTS_BY_NAME:
        return (
            {
                "parameters": parameters,
                "error": f"No project with name {project_name} found",
            },
            404,
        )

    count = min(count, _MAX_SIMILAR_COUNT)
    project_idx = _PROJECTS_BY_NAME[project_name]
    project_vector = _VECTOR_SPACE[project_idx]

    heap = []
    for idx, other_project_vector in enumerate(_VECTOR_SPACE):
        if idx == project_idx:
            continue

        distance = np.inner(project_vector, other_project_vector) / (
            norm(project_vector) * norm(other_project_vector)
        )

        if math.isnan(distance):
            _LOGGER.error(
                f"Computed distance between vectors on indexes {project_idx} and {idx} is NaN; skipped from listing"
            )
            continue

        distance = -distance

        if len(heap) < count:
            heapq.heappush(heap, (distance, idx))
        else:
            heapq.heappushpop(heap, (distance, idx))

    result = []
    for distance, idx in sorted(heap, reverse=True):
        result.append(
            {"project_name": _PROJECTS_BY_IDX[idx], "distance": float(-distance)}
        )

    return {"parameters": parameters, "result": result}, 200


def get_python_performance_impact(project_name: str) -> tuple:
    """Check if the given project can have performance impact."""
    parameters = locals()

    if project_name not in _PROJECTS_BY_NAME:
        return (
            {
                "parameters": parameters,
                "error": f"No project with name {project_name} found",
            },
            404,
        )

    project_idx = _PROJECTS_BY_NAME[project_name]
    project_vector = _VECTOR_SPACE[project_idx]
    performance_vector = np.bitwise_and(project_vector, _PERFORMANCE_MASK)

    return (
        {
            "parameters": parameters,
            "result": {
                "performance_impact": sum(performance_vector) / sum(_PERFORMANCE_MASK)
            },
        },
        200,
    )


def get_python_list_projects(prefix: str = None):
    """List Python projects."""
    parameters = locals()

    result = []
    # As we use Python 3.6+, keys in dictionary preserve insert order - we insert projects by sorted project name.
    for project_name in _PROJECTS_BY_NAME.keys():
        if prefix and project_name.startswith(prefix):
            result.append(project_name)
        elif not prefix:
            result.append(project_name)

    return {
        "parameters": parameters,
        "projects": result
    }


def _load_model():
    """Load model to wsgi server worker."""
    global _PROJECTS_BY_NAME
    global _VECTOR_SPACE
    global _PERFORMANCE_MASK
    global _PROJECTS_BY_IDX

    # Load project2vec.

    model_store = Project2VecModelStore()
    model_store.connect()

    projects, _VECTOR_SPACE = model_store.retrieve_model()

    # Transform array into dict to have O(1).
    for idx, project_name in enumerate(projects):
        if project_name in _PROJECTS_BY_NAME:
            raise ValueError(f"Got multiple projects with same name {project_name}")

        _PROJECTS_BY_NAME[project_name] = idx
        _PROJECTS_BY_IDX.append(project_name)

    for idx, vector in enumerate(_VECTOR_SPACE[1:]):
        if len(vector) != len(_VECTOR_SPACE[0]):
            raise ValueError(
                f"Vectors in vector space have different dimensions - index 0 has "
                f"{len(_VECTOR_SPACE[0])}, index {idx} has {len(vector)}"
            )

        if sum(vector) == 0:
            raise ValueError(
                f"Found vector with size of zero at index {idx} (project {_PROJECTS_BY_IDX[idx]})"
            )

    # Load performance mask.

    performance_mask_store = PerformanceMaskStore()
    performance_mask_store.connect()
    _PERFORMANCE_MASK = performance_mask_store.retrieve_mask()

    if len(_PERFORMANCE_MASK) != len(_VECTOR_SPACE[0]):
        raise ValueError(
            "Performance mask dimension does not conform to vector space dimension"
        )


# Load model on start.
_load_model()

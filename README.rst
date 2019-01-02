Isis
====

A service exposing package tags and package categories to Thoth's
recommendation engine.


project2vec
###########

Isis exposes functionality on top of project2vec - description of a package using a vector. The vector consists of features that the given project provides. These features are aggregated based on keywords found in the Python ecosystem and subsequently they are extracted from project descriptions and other free text descriptions of a project (README files on linked GitHub repos).

.. figure:: https://raw.githubusercontent.com/thoth-station/isis-api/master/example/tb.gif
   :alt: TensorBoard project2vec visualization
   :align: center


Deployment
##########

The service is built using OpenShift's s2i. On deployment, there is first run
an init container that downloads model from Ceph/S3 (created by one of the
flows defined by `selinon-worker
<https;//github.com/thoth-station/selinon-worker>`_ flows) and then there are
run two applications - an API service exposing REST endpoints and an
TensorBoard that serves the given model for debugging and exploration.

